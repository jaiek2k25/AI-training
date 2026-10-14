"""Independent dense NumPy transformer oracle for multiple causal positions."""
import json
import numpy as np
import pytest
from runtime import Runtime
from quantize import quantize
from synthetic import create_checkpoint
from penta_format import read_directory, unpack_pentanary, PACKED


@pytest.mark.parametrize('family', ['qwen2', 'llama'])
def test_decoder_against_dense_reference(tmp_path, family):
    config, _ = create_checkpoint(tmp_path, family=family)
    path = tmp_path / 'model_pentanary.bin'
    quantize(tmp_path, path)
    start, entries = read_directory(path)
    weights = {}
    with path.open('rb') as f:
        for name, t in entries.items():
            f.seek(start + t.offset)
            raw = f.read(t.nbytes)
            x = unpack_pentanary(raw, t.size).astype(np.float64) * t.alpha if t.kind == PACKED else np.frombuffer(raw, '<f2').astype(np.float64)
            weights[name] = x.reshape(t.shape)
    runtime = Runtime(path, tmp_path / 'config.json', tmp_path / 'tokenizer.json', max_context=64)
    layers, heads, kv_heads, dim = 2, 4, 2, 8
    cache_k, cache_v = [[] for _ in range(layers)], [[] for _ in range(layers)]

    def norm(x, name):
        return (x / np.sqrt(np.mean(x * x) + config['rms_norm_eps']) * weights[name]).astype(np.float32)

    def linear(x, name):
        # Independent expression for dynamic activation quantization, including
        # float32 outputs between operators as in the native implementation.
        x = x.astype(np.float64)
        scale = np.max(np.abs(x)) / 32767 if x.any() else 1
        fixed = np.sign(x) * np.floor(np.abs(x) / scale + .5)
        result = (weights[name + '.weight'] @ (fixed * scale)).astype(np.float32)
        if name + '.bias' in weights:
            result += weights[name + '.bias'].astype(np.float32)
        return result

    for pos, token in enumerate([10, 11, 16, 8]):
        x = weights['model.embed_tokens.weight'][token].astype(np.float32)
        for layer in range(layers):
            p = f'model.layers.{layer}.'
            normalized = norm(x, p + 'input_layernorm.weight')
            q = linear(normalized, p + 'self_attn.q_proj').reshape(heads, dim)
            k = linear(normalized, p + 'self_attn.k_proj').reshape(kv_heads, dim)
            v = linear(normalized, p + 'self_attn.v_proj').reshape(kv_heads, dim)
            frequencies = config['rope_theta'] ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
            angles = np.tile(pos * frequencies, 2)
            def rotate(a):
                rotated = np.concatenate([-a[:, dim // 2:], a[:, :dim // 2]], axis=1)
                return (a * np.cos(angles) + rotated * np.sin(angles)).astype(np.float32)
            q, k = rotate(q), rotate(k)
            cache_k[layer].append(k.astype(np.float16))
            cache_v[layer].append(v.astype(np.float16))
            keys = np.stack(cache_k[layer]).astype(np.float32)
            values = np.stack(cache_v[layer]).astype(np.float32)
            # Explicit GQA broadcast and dense score matrix, different from
            # runtime's per-head loop and without calling runtime helpers.
            keys = np.repeat(keys, heads // kv_heads, axis=1)
            values = np.repeat(values, heads // kv_heads, axis=1)
            scores = np.einsum('hd,thd->ht', q, keys) / np.sqrt(dim)
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            attention = np.einsum('ht,thd->hd', probs, values).astype(np.float32)
            x += linear(attention.ravel(), p + 'self_attn.o_proj')
            normalized = norm(x, p + 'post_attention_layernorm.weight')
            gate = linear(normalized, p + 'mlp.gate_proj')
            up = linear(normalized, p + 'mlp.up_proj')
            activated = gate / (1 + np.exp(-gate)) * up
            x += linear(activated, p + 'mlp.down_proj')
        expected = linear(norm(x, 'model.norm.weight'), 'lm_head')
        np.testing.assert_allclose(runtime.forward(token), expected, rtol=2e-3, atol=2e-4)


def test_stop_strings_do_not_leak_across_tokens(tmp_path):
    create_checkpoint(tmp_path)
    quantize(tmp_path, tmp_path / 'model_pentanary.bin')
    runtime = Runtime(tmp_path / 'model_pentanary.bin', tmp_path / 'config.json', tmp_path / 'tokenizer.json', max_context=64)
    # Deterministic logits with real tokenizer decoding: "hello world".
    tokens = iter([runtime.tokenizer.token_to_id('hello'), runtime.tokenizer.token_to_id('world')])
    def forward(*args, **kwargs):
        logits = np.full(runtime.vocab, -1e5, np.float32)
        logits[next(tokens)] = 0
        return logits
    runtime.forward = forward
    chunks = list(runtime.generate([1], max_tokens=2, stop=['lo world']))
    assert ''.join(c[0] for c in chunks) == 'hel'
    assert chunks[-1][2] == 'stop'
