"""Run with the optional training environment: pytest -q tests/test_qat.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from torch import nn
from qat import (PentanaryQuantizer, PentanaryWeight, ActivationQuantizer,
                 install_qat, export_qat)
from penta_format import read_directory, unpack_pentanary, PACKED
from quantize import states, quantize
from native import NativeModel


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(42)
    torch.set_num_threads(1)


def decode(path):
    start, entries = read_directory(path)
    arrays = {}
    with open(path, 'rb') as f:
        for name, t in entries.items():
            f.seek(start + t.offset)
            raw = f.read(t.nbytes)
            x = (unpack_pentanary(raw, t.size).astype(np.float32) * t.alpha if t.kind == PACKED
                 else np.frombuffer(raw, '<f2').astype(np.float32))
            arrays[name] = x.reshape(t.shape)
    return arrays


@pytest.mark.parametrize('sparse', [True, False])
def test_ste_and_thresholds(sparse):
    x = torch.tensor([[-.5, .5, -1.5, 1.5], [2., -2., 0., 0.], [10., -10., .1, -.1]], requires_grad=True)
    alpha = torch.tensor(1., requires_grad=True)
    out = PentanaryQuantizer.apply(x, alpha, sparse)
    np.testing.assert_array_equal(out.detach().numpy(), states(x.detach().numpy().ravel(), 1., sparse).reshape(x.shape))
    out.sum().backward()
    # Identity gradient includes pruned/zero/saturated entries; alpha is fixed.
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    assert alpha.grad is None


@pytest.mark.parametrize('alpha', [0, -1, float('inf'), float('nan'), 70000])
def test_invalid_scale(alpha):
    with pytest.raises(ValueError):
        PentanaryWeight(alpha)


def test_activation_ste_native_rounding():
    x = torch.tensor([[32767., .5, -.5, 1.5], [0., 0., 0., 0.]], requires_grad=True)
    y = ActivationQuantizer.apply(x)
    torch.testing.assert_close(y, torch.tensor([[32767., 1., -1., 2.], [0., 0., 0., 0.]]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize('sparse,shape', [(True, (5, 4)), (False, (3, 5))])
def test_exact_export_and_native_matvec(tmp_path, sparse, shape):
    model = nn.Linear(shape[1], shape[0])
    state = install_qat(model, sparse=sparse, chunk_elements=8)
    original_scale = state.bindings['weight'].quantizer.alpha.item()
    # Move the master enough that refitting would change the scale. Export must NOT refit.
    with torch.no_grad():
        state.bindings['weight'].master.mul_(1.7)
        state.bindings['bias'].master.add_(.0123456)
    path = tmp_path / 'qat.bin'
    export_qat(state, path, chunk_elements=8)
    arrays = decode(path)
    np.testing.assert_array_equal(arrays['weight'], model.weight.detach().numpy())
    np.testing.assert_array_equal(arrays['bias'], model.bias.detach().numpy())
    assert read_directory(path)[1]['weight'].alpha == original_scale
    if sparse:
        assert np.all((arrays['weight'].reshape(-1, 4) == 0).sum(axis=1) >= 2)
    x = torch.randn(3, shape[1])
    expected = model(x).detach().numpy()
    with NativeModel(path) as native:
        for row in range(3):
            got = native.matvec('weight', x[row].numpy()) + arrays['bias']
            np.testing.assert_allclose(got, expected[row], rtol=1e-5, atol=2e-6)
    second = tmp_path / 'other.bin'
    export_qat(state, second, chunk_elements=1000)
    assert path.read_bytes() == second.read_bytes()


def test_tied_weights_share_master_and_scale(tmp_path):
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16, bias=False)
            self.head.weight = self.embed.weight
        def forward(self, ids):
            return self.head(self.embed(ids))
    model = Tied()
    state = install_qat(model)
    a, b = state.bindings['embed.weight'], state.bindings['head.weight']
    assert a.master is b.master
    assert a.quantizer is b.quantizer
    assert len(list(model.parameters())) == 1
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    model(torch.tensor([1, 2])).square().mean().backward()
    optimizer.step()
    export_qat(state, tmp_path / 'tied.bin')
    arrays = decode(tmp_path / 'tied.bin')
    np.testing.assert_array_equal(arrays['embed.weight'], arrays['head.weight'])


def test_real_optimizer_updates_and_reduces_toy_loss():
    model = nn.Linear(8, 4, bias=False)
    state = install_qat(model)
    x = torch.randn(32, 8)
    target = torch.zeros(32, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=.08)
    before = float((model(x) - target).square().mean().detach())
    master_before = state.bindings['weight'].master.detach().clone()
    for _ in range(50):
        optimizer.zero_grad()
        (model(x) - target).square().mean().backward()
        optimizer.step()
    after = float((model(x) - target).square().mean().detach())
    assert not torch.equal(master_before, state.bindings['weight'].master)
    assert after < before


def test_reject_nonfinite_export_without_replacing_file(tmp_path):
    model = nn.Linear(4, 4)
    state = install_qat(model)
    path = tmp_path / 'q.bin'
    export_qat(state, path)
    old = path.read_bytes()
    with torch.no_grad():
        state.bindings['weight'].master[0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        export_qat(state, path)
    assert path.read_bytes() == old


def test_preflight_and_double_install():
    with pytest.raises(ValueError, match='FP32'):
        install_qat(nn.Linear(4, 4).half())
    with pytest.raises(ValueError, match='2:4'):
        install_qat(nn.Linear(5, 4))
    model = nn.Linear(4, 4)
    install_qat(model)
    with pytest.raises(ValueError, match='already parametrized'):
        install_qat(model)


def test_pytorch_ptq_input_now_available(tmp_path):
    torch.save({'w': torch.arange(32, dtype=torch.float32).reshape(4, 8)}, tmp_path / 'source.pt')
    quantize(tmp_path / 'source.pt', tmp_path / 'q.bin')
    assert decode(tmp_path / 'q.bin')['w'].shape == (4, 8)


def test_data_batching_and_token_weighted_evaluation(tmp_path):
    from train_qat import collate, evaluate, target_count, batches
    from synthetic import create_checkpoint
    from tokenizers import Tokenizer
    create_checkpoint(tmp_path)
    tokenizer = Tokenizer.from_file(str(tmp_path / 'tokenizer.json'))
    data = tmp_path / 'data.jsonl'
    data.write_text(json.dumps({'text': 'hello world hello world hello'}) + '\n')
    got = list(batches(data, tokenizer, 3, 2, 0))
    assert got[0]['input_ids'].shape == (2, 3)
    assert got[0]['labels'][1, -1] == -100
    assert target_count(got[0]) == 3
    class Fake(nn.Module):
        def forward(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(loss=torch.tensor(float(kwargs['input_ids'].shape[-1])))
    result = evaluate(Fake(), iter([collate([[1, 2]], 0), collate([[1, 2, 3, 4]], 0)]), 2, 'cpu')
    assert result['target_tokens'] == 4
    assert result['nll'] == (2 * 1 + 4 * 3) / 4
    data.write_text('{"text":"hello"}\n')
    with pytest.raises(ValueError, match='at least two'):
        list(batches(data, tokenizer, 8, 1, 0))


@pytest.mark.parametrize('family,tied', [('qwen2', False), ('llama', False), ('qwen2', True)])
def test_hf_training_export_and_native_decoder(tmp_path, family, tied):
    pytest.importorskip('transformers')
    from transformers import AutoModelForCausalLM
    from synthetic import create_checkpoint
    from train_qat import collate, train_steps
    create_checkpoint(tmp_path, family=family)
    if tied:
        config = json.loads((tmp_path / 'config.json').read_text())
        config['tie_word_embeddings'] = True
        (tmp_path / 'config.json').write_text(json.dumps(config))
    model = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True, torch_dtype=torch.float32,
                                               attn_implementation='eager')
    state = install_qat(model, chunk_elements=128)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    batch = collate([[10, 11, 16, 8], [10, 11, 12]], 0)
    losses = train_steps(model, optimizer, iter([batch] * 6), 3, 'cpu', grad_accum=2)
    assert all(np.isfinite(losses))
    assert all(p.grad is not None for p in model.parameters())
    model.eval()
    path = tmp_path / 'model_pentanary.bin'
    export_qat(state, path, chunk_elements=128, omit_tied_output=tied)
    if tied:
        assert 'lm_head.weight' not in read_directory(path)[1]
    with torch.no_grad():
        expected = model(input_ids=torch.tensor([[10, 11, 16, 8]]), use_cache=False).logits[0].numpy()
    np.save(tmp_path / 'expected.npy', expected)
    # Inference must not import torch: launch a fresh process to keep its memory
    # admission meaningful. This also checks the actual exported deployment files.
    script = '''
import sys
from pathlib import Path
import numpy as np
from runtime import Runtime
p=Path(sys.argv[1])
m=Runtime(p/'model_pentanary.bin', p/'config.json', p/'tokenizer.json', max_context=64)
actual=np.stack([m.forward(t) for t in [10,11,16,8]])
expected=np.load(p/'expected.npy')
np.testing.assert_allclose(actual, expected, rtol=.01, atol=.003)
print('native/HF max absolute logit error:', float(np.max(np.abs(actual-expected))))
'''
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=env, check=True)


def test_train_cli(tmp_path):
    from synthetic import create_checkpoint
    source = tmp_path / 'source'
    create_checkpoint(source)
    train, heldout = tmp_path / 'train.jsonl', tmp_path / 'heldout.jsonl'
    train.write_text('\n'.join(json.dumps({'text': s}) for s in ['hello world hello world', 'world hello test hello']))
    heldout.write_text(json.dumps({'text': 'hello test world hello'}) + '\n')
    output = tmp_path / 'deployment'
    command = [sys.executable, 'train_qat.py', '--checkpoint', str(source), '--train-data', str(train),
               '--eval-data', str(heldout), '--output', str(output), '--device', 'cpu', '--steps', '3',
               '--sequence-length', '16', '--eval-batches', '1', '--gradient-checkpointing']
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    subprocess.run(command, env=env, check=True)
    metrics = json.loads((output / 'qat_metrics.json').read_text())
    assert len(metrics['train_losses']) == 3
    assert all(metrics[k]['target_tokens'] == 3 for k in ['baseline_fp32', 'before_qat', 'after_qat'])
    assert (output / 'tokenizer.json').exists() and (output / 'config.json').exists()
    assert json.loads((output / 'model_pentanary.bin.json').read_text())['method'] == 'QAT-STE'
