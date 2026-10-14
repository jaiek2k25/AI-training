import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import numpy as np
import pytest
import memory_limits as limits
from memory_plan import memory_plan
from penta_format import TensorInfo, PACKED, FP16, write_directory, pack_pentanary
from run_450 import scope_command


def config():
    return {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'vocab_size': 128, 'max_position_embeddings': 512}


def test_packing_arithmetic_and_no_sparsity_discount(tmp_path):
    assert (1_500_000_000 + 2) // 3 == 500_000_000
    assert (1_350_000_000 + 2) // 3 == 450_000_000
    assert (1_300_000_000 + 2) // 3 == 433_333_334
    reports = []
    for value in [0, 2]:
        path = tmp_path / f'w{value}.bin'
        with path.open('wb') as f:
            write_directory(f, [TensorInfo('w', (4, 4), PACKED), TensorInfo('norm', (4,), FP16)])
            f.write(pack_pentanary(np.full(16, value)).tobytes())
            f.write(np.ones(4, dtype='<f2').tobytes())
        reports.append(memory_plan(path, config(), context=64))
    assert reports[0] == reports[1]
    assert reports[0]['storage']['zero_skipping_storage_discount_bytes'] == 0
    assert reports[0]['storage']['fp16_vector_bytes'] == 8
    assert not reports[0]['current_streamed_loader']['hard_limit_or_success_guaranteed_by_this_report']


def test_large_file_is_not_a_resident_fit(tmp_path):
    # Sparse filesystem file exercises metadata arithmetic, not model accuracy;
    # no 450MB array is allocated and the directory scan never reads its payload.
    path = tmp_path / 'large.bin'
    t = TensorInfo('w', (337500, 4000), PACKED)
    with path.open('wb') as f:
        start = write_directory(f, [t])
        f.truncate(start + t.nbytes)
    report = memory_plan(path, config(), context=64)
    assert report['storage']['packed_weight_bytes'] == 450_000_000
    assert not report['hypothetical_fully_resident']['fits_admission_estimate']
    assert report['hypothetical_fully_resident']['estimated_bytes'] > 450_000_000
    assert report['current_streamed_loader']['fits_process_admission_estimate']
    assert not report['current_streamed_loader']['filesystem_page_cache_budgeted']


def test_kv_formula_not_ring_buffer():
    assert limits.kv_cache_bytes(config(), 64) == 8192
    assert limits.kv_cache_bytes(config(), 128) == 16384
    assert limits.scratch_reserve_bytes(8, 128, 64) > 64_000_000
    with pytest.raises(ValueError):
        limits.kv_cache_bytes(config(), 513)
    with pytest.raises(ValueError):
        limits.kv_cache_bytes({**config(), 'num_key_value_heads': 3}, 64)


def fake_cgroup(tmp_path, root='/', member='/tenant/worker'):
    mount = tmp_path / 'mount space'
    relative = Path(member).relative_to(root)
    leaf = mount / relative
    leaf.mkdir(parents=True)
    for name, value in {'cgroup.procs': '123\n', 'memory.max': 'max\n',
                        'memory.swap.max': 'max\n', 'memory.current': '10000\n',
                        'memory.peak': '20000\n', 'memory.events': 'oom 0\noom_kill 0\n'}.items():
        (leaf / name).write_text(value)
    proc = tmp_path / 'proc_cgroup'
    proc.write_text(f'0::{member}\n')
    info = tmp_path / 'mountinfo'
    escaped = str(mount).replace(' ', r'\040')
    info.write_text(f'30 23 0:25 {root} {escaped} rw - cgroup2 cgroup2 rw\n')
    return proc, info, leaf, mount


def test_cgroup_ancestors_and_escaped_mount(tmp_path):
    proc, info, leaf, mount = fake_cgroup(tmp_path)
    (leaf.parent / 'memory.max').write_text('400000000')
    (leaf.parent / 'memory.swap.max').write_text('0')
    result = limits.cgroup_memory(proc, info)
    assert result['available']
    assert result['memory_max_bytes'] is None
    assert result['effective_memory_max_bytes'] == 400000000
    assert result['effective_swap_max_bytes'] == 0
    assert result['peak_bytes'] == 20000
    assert result['events']['oom_kill'] == 0


def test_cgroup_subtree_mount_and_failures(tmp_path):
    proc, info, leaf, mount = fake_cgroup(tmp_path, root='/tenant')
    assert limits.cgroup_directory(proc, info) == (leaf, mount)
    (leaf / 'memory.max').write_text('garbage')
    assert not limits.cgroup_memory(proc, info)['available']
    proc.write_text('0::/../../hidden\n')
    assert not limits.cgroup_memory(proc, info)['available']
    proc.write_text('1:memory:/legacy\n')
    assert not limits.cgroup_memory(proc, info)['available']


@pytest.mark.parametrize('memory,swap,available,accepted', [
    (450000000, 0, True, True), (400000000, 0, True, True),
    (450000001, 0, True, False), (None, 0, True, False),
    (450000000, None, True, False), (450000000, 1000, True, False),
    (450000000, 0, False, False)])
def test_hard_guard(monkeypatch, memory, swap, available, accepted):
    monkeypatch.setattr(limits, 'cgroup_memory', lambda: {
        'available': available, 'effective_memory_max_bytes': memory, 'effective_swap_max_bytes': swap})
    if accepted:
        assert limits.require_hard_limit()['effective_memory_max_bytes'] <= 450000000
    else:
        with pytest.raises(RuntimeError, match='No soft fallback'):
            limits.require_hard_limit()


def test_launcher_dry_run_and_scope_properties():
    command = scope_command(['python', 'server.py', '--strict-450mb'])
    assert '--user' in command and '--scope' in command
    aligned = limits.page_aligned_limit()
    assert aligned <= 450000000 and aligned % os.sysconf('SC_PAGE_SIZE') == 0
    assert f'--property=MemoryMax={aligned}' in command
    assert '--property=MemorySwapMax=0' in command
    assert '--property=OOMPolicy=kill' in command
    result = subprocess.run([sys.executable, 'run_450.py', '--dry-run', '--', '--model', 'a file.bin'],
                            capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert 'a file.bin' in report['command']  # no shell splitting
    assert '--strict-450mb' in report['command']
    assert report['environment_overrides']['OPENBLAS_NUM_THREADS'] == '1'
    assert os.path.abspath(sys.executable) in report['command']  # venv symlink preserved


def test_server_refuses_strict_without_cap(monkeypatch):
    import server
    def refuse():
        raise RuntimeError('No soft fallback')
    monkeypatch.setattr(server, 'require_hard_limit', refuse)
    monkeypatch.setattr(sys, 'argv', ['server.py', '--strict-450mb', '--model', 'missing',
                                     '--config', 'missing', '--tokenizer', 'missing'])
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2


def test_server_refuses_excessive_strict_admission(monkeypatch):
    import server
    monkeypatch.setattr(sys, 'argv', ['server.py', '--strict-450mb', '--memory-budget-mb', '450',
                                     '--model', 'missing', '--config', 'missing', '--tokenizer', 'missing'])
    with pytest.raises(SystemExit) as exc:
        server.main()
    assert exc.value.code == 2


def system_scope(script, *args):
    if os.environ.get('PENTA_TEST_SYSTEMD') != '1':
        pytest.skip('set PENTA_TEST_SYSTEMD=1 for isolated privileged cgroup integration tests')
    probe = subprocess.run(['sudo', '-n', 'true'], capture_output=True)
    if probe.returncode:
        pytest.skip('noninteractive privilege unavailable for temporary system scope')
    username = pwd.getpwuid(os.getuid()).pw_name
    command = scope_command(['runuser', '-u', username, '--', 'env',
                             'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'MALLOC_ARENA_MAX=2',
                             os.path.abspath(sys.executable), '-c', script, *map(str, args)], user=False)
    return subprocess.run(['sudo', '-n', *command], capture_output=True, text=True, timeout=45,
                          env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})


def test_live_hard_cap_api_and_full_tiny_context(tmp_path):
    if os.environ.get('PENTA_TEST_SYSTEMD') != '1':
        pytest.skip('opt-in kernel test')
    from synthetic import create_checkpoint
    from quantize import quantize
    create_checkpoint(tmp_path)
    quantize(tmp_path, tmp_path / 'model_pentanary.bin')
    script = '''
import json, sys
from pathlib import Path
from memory_limits import require_hard_limit
require_hard_limit()
from runtime import Runtime
from server import create_app
from fastapi.testclient import TestClient
p=Path(sys.argv[1])
m=Runtime(p/'model_pentanary.bin',p/'config.json',p/'tokenizer.json',max_context=64,memory_budget=420000000)
m.strict_450mb=True
for _ in range(64): m.forward(10,logits=False)
with TestClient(create_app(m)) as c:
    payload={'messages':[{'role':'user','content':'hello world'}],'max_tokens':4}
    r=c.post('/v1/chat/completions',json=payload); r.raise_for_status()
    r=c.post('/v1/chat/completions',json={**payload,'stream':True}); r.raise_for_status()
    assert 'data: [DONE]' in r.text
    health=c.get('/health').json()
    assert health['strict_450mb_profile']
    assert health['cgroup_memory']['effective_swap_max_bytes']==0
    assert health['cgroup_memory']['events']['oom_kill']==0
    print(json.dumps(health),flush=True)
'''
    result = system_scope(script, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report['cgroup_memory']['effective_memory_max_bytes'] <= 450000000
    directory = Path('artifacts/memory450')
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'kernel-smoke.json').write_text(json.dumps(report, indent=2) + '\n')


def test_kernel_kills_oversized_allocation():
    script = '''
import json
from memory_limits import require_hard_limit
print(json.dumps({'guard':require_hard_limit()}),flush=True)
# Deliberately exceed the test scope, not the parent session's cgroup.
x=bytearray(600000000)
print('UNEXPECTED_ALLOCATION_SUCCESS',flush=True)
'''
    result = system_scope(script)
    assert result.returncode in (137, -9), result.stdout + result.stderr
    guard = json.loads(result.stdout)['guard']
    assert guard['effective_memory_max_bytes'] <= 450000000
    assert guard['effective_swap_max_bytes'] == 0
    assert 'UNEXPECTED_ALLOCATION_SUCCESS' not in result.stdout
    unit = Path(guard['cgroup']).name
    assert unit.startswith('run-') and unit.endswith('.scope')
    try:
        status = subprocess.run(['sudo', '-n', 'systemctl', 'show', unit, '--property=Result', '--value'],
                                capture_output=True, text=True, check=True)
        assert status.stdout.strip() == 'oom-kill'
        directory = Path('artifacts/memory450')
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'oom-test.json').write_text(json.dumps({'returncode': result.returncode,
                                                           'systemd_result': status.stdout.strip(),
                                                           'guard_before_allocation': guard,
                                                           'stderr': result.stderr}, indent=2) + '\n')
    finally:
        subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', unit], check=True,
                       capture_output=True, text=True)
