import itertools
import json
from pathlib import Path
import struct
import numpy as np
import pytest
from fastapi.testclient import TestClient
from penta_format import (pack_pentanary, unpack_pentanary, read_directory,
                          write_directory, TensorInfo, PACKED, FP16)
from quantize import quantize, states, SafeSource
from native import NativeModel
from runtime import Runtime
from synthetic import create_checkpoint, write_safetensors
from server import create_app


def decode(path):
    start, entries = read_directory(path)
    result = {}
    with open(path, 'rb') as f:
        for name, t in entries.items():
            f.seek(start + t.offset)
            b = f.read(t.nbytes)
            result[name] = ((unpack_pentanary(b, t.size).astype(np.float32) * t.alpha) if t.kind == PACKED
                            else np.frombuffer(b, '<f2').astype(np.float32)).reshape(t.shape)
    return result


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp('checkpoint')
    create_checkpoint(path)
    quantize(path, path / 'model_pentanary.bin', chunk_elements=128)
    return path


def engine(path, **kwargs):
    return Runtime(path / 'model_pentanary.bin', path / 'config.json', path / 'tokenizer.json', **kwargs)


def test_pack_exhaustive_and_tails():
    triplets = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=np.int8)
    assert set(pack_pentanary(triplets)) == set(range(125))
    for count in [1, 2, 3, 4, 374, 375]:
        q = triplets.ravel()[:count]
        np.testing.assert_array_equal(unpack_pentanary(pack_pentanary(q).tobytes(), count), q)
    with pytest.raises(ValueError):
        unpack_pentanary(b'\xff', 3)
    with pytest.raises(ValueError):
        pack_pentanary([0.5])


def test_thresholds_and_structural_zeros():
    x = np.array([-2, -1, .1, .2, 2, 1, -.1, -.2], dtype=np.float32)
    np.testing.assert_array_equal(states(x, 1), [-2, -1, 0, 0, 2, 1, 0, 0])
    np.testing.assert_array_equal(states(np.array([-.49, .5, 1.49, 1.5]), 1, False), [0, 1, 1, 2])
    q = states(np.zeros(8), 1)
    assert not q.any()


def test_quantize_scale_sparsity_and_chunk_equivalence(checkpoint, tmp_path):
    report = quantize(checkpoint, tmp_path / 'other.bin', chunk_elements=512)
    a = decode(checkpoint / 'model_pentanary.bin')
    b = decode(tmp_path / 'other.bin')
    assert a.keys() == b.keys()
    for name, x in a.items():
        np.testing.assert_array_equal(x, b[name])
        if x.ndim == 2:
            assert np.all((x.reshape(-1, 4) == 0).sum(axis=1) == 2)
    assert all(np.isfinite(t['alpha']) for t in report['tensors'])


def test_native_matvec_and_rows(checkpoint):
    rng = np.random.default_rng(7)
    reference = decode(checkpoint / 'model_pentanary.bin')
    with NativeModel(checkpoint / 'model_pentanary.bin') as m:
        for name, matrix in reference.items():
            if matrix.ndim == 1:
                np.testing.assert_array_equal(m.row(name), matrix)
                continue
            for row in [0, 1, matrix.shape[0] - 1]:
                np.testing.assert_array_equal(m.row(name, row), matrix[row])
            for x in [rng.normal(size=matrix.shape[1]).astype(np.float32), np.zeros(matrix.shape[1], np.float32),
                      np.full(matrix.shape[1], -32767, np.float32)]:
                scale = np.max(np.abs(x)).astype(np.float64) / 32767 if x.any() else 1
                # C++ std::round is ties-away-from-zero.
                quant_x = np.sign(x) * np.floor(np.abs(x.astype(np.float64) / scale) + .5)
                ref = matrix.astype(np.float64) @ (quant_x * scale)
                np.testing.assert_allclose(m.matvec(name, x), ref, rtol=1e-5, atol=1e-4)
                bound = np.abs(matrix).sum(axis=1) * scale / 2 + 1e-3
                assert np.all(np.abs(m.matvec(name, x) - matrix.astype(np.float64) @ x) <= bound)
        with pytest.raises(ValueError):
            m.matvec('lm_head.weight', np.full(32, np.nan, np.float32))
        with pytest.raises(ValueError):
            m.matvec('lm_head.weight', np.zeros(31, np.float32))


@pytest.mark.parametrize('shape', [(5, 4), (3, 5), (1, 1)])
def test_native_unaligned_rows_and_padding(tmp_path, shape):
    q = (np.arange(np.prod(shape)) % 5 - 2).astype(np.int8).reshape(shape)
    path = tmp_path / 'tail.bin'
    with path.open('wb') as f:
        write_directory(f, [TensorInfo('w', shape, PACKED, .5)])
        f.write(pack_pentanary(q).tobytes())
    with NativeModel(path) as m:
        for i in range(shape[0]):
            np.testing.assert_array_equal(m.row('w', i), q[i] * .5)
        x = np.full(shape[1], -2, dtype=np.float32)
        np.testing.assert_array_equal(m.matvec('w', x), q @ x * .5)


def test_bf16_and_shards(tmp_path):
    x = np.arange(32, dtype=np.float32).reshape(4, 8) / 8
    write_safetensors(tmp_path / 'a.safetensors', {'a': x}, bf16=True)
    write_safetensors(tmp_path / 'b.safetensors', {'b': x / 2}, bf16=True)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'a.safetensors', 'b': 'b.safetensors'}}))
    got = np.concatenate(list(SafeSource(tmp_path / 'a.safetensors').chunks('a', 16)))
    np.testing.assert_array_equal(got, x.ravel())
    quantize(tmp_path, tmp_path / 'q.bin')
    assert set(read_directory(tmp_path / 'q.bin')[1]) == {'a', 'b'}


def test_pytorch_optional(tmp_path):
    torch = pytest.importorskip('torch')
    torch.save({'w': torch.arange(32, dtype=torch.float32).reshape(4, 8)}, tmp_path / 'source.pt')
    quantize(tmp_path / 'source.pt', tmp_path / 'q.bin')
    assert decode(tmp_path / 'q.bin')['w'].shape == (4, 8)


def test_reject_corruption(checkpoint, tmp_path):
    original = (checkpoint / 'model_pentanary.bin').read_bytes()
    for payload in [original[:10], b'BADMAGIC' + original[8:], original[:-3]]:
        path = tmp_path / 'bad.bin'
        path.write_bytes(payload)
        with pytest.raises(ValueError):
            NativeModel(path)
        # Exercise native parser too, independent of the Python validation.
        with NativeModel(checkpoint / 'model_pentanary.bin') as m:
            handle = m.lib.penta_open(str(path).encode())
            assert not handle
    path = tmp_path / 'badpayload.bin'
    path.write_bytes(original)
    start, tensors = read_directory(path)
    with path.open('r+b') as f:
        f.seek(start + tensors['lm_head.weight'].offset)
        f.write(b'\xff')
    with NativeModel(path) as m:
        with pytest.raises(ValueError, match='base-5'):
            m.matvec('lm_head.weight', np.ones(32, np.float32))


def test_invalid_quantization_inputs(tmp_path):
    for x in [np.full((4, 4), np.nan), np.ones((4, 5))]:
        write_safetensors(tmp_path / 'source.safetensors', {'w': x})
        with pytest.raises(ValueError):
            quantize(tmp_path / 'source.safetensors', tmp_path / 'q.bin')
        assert not (tmp_path / 'q.bin').exists()


def test_runtime_context_reset_and_memory(checkpoint):
    rt = engine(checkpoint, max_context=64)
    first = rt.forward(10)
    second = rt.forward(11)
    assert first.shape == (128,) and np.isfinite(second).all()
    rt.reset()
    np.testing.assert_array_equal(rt.forward(10), first)
    np.testing.assert_array_equal(rt.forward(11), second)
    assert rt.kv_bytes == 2 * 2 * 64 * 2 * 8 * 2
    with pytest.raises(MemoryError):
        engine(checkpoint, memory_budget=1)


def test_api_stream_nonstream_and_errors(checkpoint):
    rt = engine(checkpoint, max_context=64)
    with TestClient(create_app(rt)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/v1/models').json()['data'][0]['id'] == 'pentanary'
        request = {'model': 'pentanary', 'messages': [{'role': 'user', 'content': 'hello world'}], 'max_tokens': 4}
        response = client.post('/v1/chat/completions', json=request)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result['choices'][0]['finish_reason'] in ['stop', 'length']
        usage = result['usage']
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        streamed = client.post('/v1/chat/completions', json={**request, 'stream': True, 'stream_options': {'include_usage': True}})
        assert streamed.headers['content-type'].startswith('text/event-stream')
        frames = [line[6:] for line in streamed.text.splitlines() if line.startswith('data: ')]
        assert frames[-1] == '[DONE]'
        chunks = [json.loads(x) for x in frames[:-1]]
        text = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
        assert text == result['choices'][0]['message']['content']
        assert chunks[-1]['usage'] == usage
        assert not rt.lock.locked()
        assert client.post('/v1/chat/completions', json={**request, 'model': 'absent'}).status_code == 404
        assert client.post('/v1/chat/completions', json={**request, 'max_tokens': 64}).status_code == 400
        assert client.post('/v1/chat/completions', json={**request, 'temperature': -1}).status_code == 422
        assert client.post('/v1/chat/completions', json={**request, 'stop': ''}).status_code == 400
        assert client.post('/v1/chat/completions', content=b'x' * ((1 << 20) + 1)).status_code == 413
        rt.lock.acquire()
        try:
            assert client.post('/v1/chat/completions', json=request).status_code == 429
        finally:
            rt.lock.release()


def test_llama3_and_tied_embedding(tmp_path):
    create_checkpoint(tmp_path, family='llama')
    config = json.loads((tmp_path / 'config.json').read_text())
    config['rope_scaling'] = {'rope_type': 'llama3', 'factor': 8, 'low_freq_factor': 1,
                             'high_freq_factor': 4, 'original_max_position_embeddings': 8192}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    quantize(tmp_path, tmp_path / 'model_pentanary.bin')
    rt = engine(tmp_path, max_context=64)
    assert np.isfinite(rt.forward(10)).all()
    assert rt.encode_messages([{'role': 'user', 'content': 'hello'}])
    # A separate tied checkpoint genuinely omits lm_head.weight.
    _, weights = create_checkpoint(tmp_path, family='llama')
    weights.pop('lm_head.weight')
    write_safetensors(tmp_path / 'model.safetensors', weights)
    config['tie_word_embeddings'] = True
    (tmp_path / 'config.json').write_text(json.dumps(config))
    quantize(tmp_path, tmp_path / 'model_pentanary.bin')
    tied = engine(tmp_path, max_context=64)
    assert tied.output_weight == 'model.embed_tokens.weight'
    assert np.isfinite(tied.forward(10)).all()
