"""Linux cgroup-v2 accounting and admission estimates (stdlib only).

A cgroup limit is kernel-accounted memory, NOT an unconditional successful-run
promise or a byte-exact cap on every instantaneous RSS measurement. Filesystem
cache may be charged here; shared-page attribution follows Linux cgroup rules.
"""
import os
from pathlib import Path, PurePosixPath
import re

STRICT_LIMIT_BYTES = 450_000_000       # decimal MB, not MiB
STRICT_ADMISSION_BYTES = 420_000_000   # reserve headroom below the kernel limit
NATIVE_BUFFER_BYTES = 65_536


def page_aligned_limit(limit=STRICT_LIMIT_BYTES):
    page = os.sysconf('SC_PAGE_SIZE')
    if not isinstance(limit, int) or limit < page:
        raise ValueError('memory limit must be at least one page')
    return limit // page * page


def kv_cache_bytes(config, context):
    def dimension(key, default=None):
        value = config.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'invalid {key}')
        return value
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        raise ValueError('context must be a positive integer')
    layers = dimension('num_hidden_layers')
    heads = dimension('num_attention_heads')
    hidden = dimension('hidden_size')
    kv_heads = dimension('num_key_value_heads', heads)
    if 'head_dim' not in config and hidden % heads:
        raise ValueError('hidden_size must be divisible by attention heads without explicit head_dim')
    head_dim = dimension('head_dim', hidden // heads)
    if heads % kv_heads:
        raise ValueError('attention heads must be divisible by KV heads')
    if context > dimension('max_position_embeddings', 2048):
        raise ValueError('context exceeds model positional range')
    return 2 * layers * context * kv_heads * head_dim * 2


def scratch_reserve_bytes(head_dim, vocab, context):
    # Matches the decoder admission policy; includes generic allocator/HTTP
    # headroom, per-head attention copies, vocabulary logits/sampling buffers.
    if min(head_dim, vocab, context) <= 0:
        raise ValueError('invalid scratch dimensions')
    return (64 << 20) + context * head_dim * 8 + vocab * 32


def _unescape_mount(text):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), text)


def cgroup_directory(proc_cgroup='/proc/self/cgroup', mountinfo='/proc/self/mountinfo'):
    """Resolve the real process membership, including subtree-mounted hierarchies.

    Never trust an environment variable or an arbitrary asserted cgroup path.
    Paths with '..' (unresolved namespace ancestry) fail closed.
    """
    memberships = [line.split(':', 2)[2] for line in Path(proc_cgroup).read_text().splitlines()
                   if line.startswith('0::')]
    if len(memberships) != 1:
        raise RuntimeError('unified cgroup-v2 process membership not available')
    membership = PurePosixPath(memberships[0])
    if not membership.is_absolute() or '..' in membership.parts:
        raise RuntimeError('unresolvable cgroup namespace path')
    candidates = []
    for line in Path(mountinfo).read_text().splitlines():
        before, separator, after = line.partition(' - ')
        if not separator or not after.startswith('cgroup2 '):
            continue
        fields = before.split()
        root = PurePosixPath(_unescape_mount(fields[3]))
        mount = Path(_unescape_mount(fields[4]))
        if not root.is_absolute() or not mount.is_absolute() or '..' in root.parts or '..' in mount.parts:
            continue
        if membership.is_relative_to(root):
            leaf = mount / membership.relative_to(root)
            if (leaf / 'cgroup.procs').is_file():
                candidates.append((len(root.parts), leaf, mount))
    if not candidates:
        raise RuntimeError('no visible cgroup-v2 mount matches process membership')
    _, leaf, mount = max(candidates, key=lambda item: item[0])
    return leaf, mount


def _limit(path):
    text = path.read_text().strip()
    if text == 'max':
        return None
    value = int(text)
    if value < 0:
        raise ValueError(f'invalid cgroup limit in {path.name}')
    return value


def cgroup_memory(proc_cgroup='/proc/self/cgroup', mountinfo='/proc/self/mountinfo'):
    """Return observable limits, including the tightest visible ancestor limits."""
    try:
        leaf, mount = cgroup_directory(proc_cgroup, mountinfo)
        # The leaf must expose the controller; an inaccessible ancestor alone
        # is not enough evidence for the strict-mode guard.
        own_limit = _limit(leaf / 'memory.max')
        own_swap = _limit(leaf / 'memory.swap.max')
        limits, swaps = [], []
        parent = leaf
        while True:
            for name, values in [('memory.max', limits), ('memory.swap.max', swaps)]:
                path = parent / name
                if path.exists():
                    value = _limit(path)
                    if value is not None:
                        values.append(value)
            if parent == mount:
                break
            parent = parent.parent
        current = _limit(leaf / 'memory.current')
        peak = _limit(leaf / 'memory.peak') if (leaf / 'memory.peak').exists() else None
        events = {}
        if (leaf / 'memory.events').exists():
            events = {line.split()[0]: int(line.split()[1])
                      for line in (leaf / 'memory.events').read_text().splitlines()}
        return {'available': True, 'cgroup': str(leaf), 'memory_max_bytes': own_limit,
                'swap_max_bytes': own_swap, 'effective_memory_max_bytes': min(limits) if limits else None,
                'effective_swap_max_bytes': min(swaps) if swaps else None,
                'current_bytes': current, 'peak_bytes': peak, 'events': events}
    except (OSError, ValueError, IndexError, RuntimeError) as exc:
        return {'available': False, 'reason': str(exc)}


def require_hard_limit(limit=STRICT_LIMIT_BYTES):
    status = cgroup_memory()
    effective = status.get('effective_memory_max_bytes')
    if (not status['available'] or effective is None or effective > limit
            or status.get('effective_swap_max_bytes') != 0):
        raise RuntimeError('strict memory mode requires an observable cgroup-v2 memory.max '
                           f'<= {limit:,} bytes and memory.swap.max=0; use run_450.py '
                           'or an equivalent administrator-provisioned scope. No soft fallback. '
                           f'Observed: {status}')
    return status
