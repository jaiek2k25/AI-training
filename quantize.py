#!/usr/bin/env python3
"""Chunked, scale-aware 2:4 pentanary quantization of LOCAL HF checkpoints.

Safetensors are read directly (including BF16), without torch or full-tensor copies.
PyTorch .bin/.pt input uses optional torch.load(weights_only=True, mmap=True).
Offline quantization is not covered by the inference memory budget.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import tempfile
import os
import numpy as np
from penta_format import TensorInfo, PACKED, FP16, pack_pentanary, write_directory


class SafeSource:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('truncated safetensors header')
            n, = struct.unpack('<Q', raw)
            if not 2 <= n <= 64 << 20:
                raise ValueError('invalid safetensors header size')
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop('__metadata__', None)
        length = self.path.stat().st_size
        ranges = []
        for name, h in self.header.items():
            dtype = h['dtype']
            if dtype not in ('F16', 'F32', 'BF16'):
                raise ValueError(f'{name}: unsupported dtype {dtype}')
            count = math.prod(h['shape'])
            lo, hi = h['data_offsets']
            if (not h['shape'] or min(h['shape']) <= 0 or lo < 0 or hi < lo
                    or hi - lo != count * (4 if dtype == 'F32' else 2) or self.base + hi > length):
                raise ValueError(f'{name}: invalid safetensors offsets/shape')
            ranges.append((lo, hi))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('overlapping safetensors tensors')

    def entries(self):
        return [(n, tuple(h['shape'])) for n, h in self.header.items()]

    def chunks(self, name, chunk_elements):
        h = self.header[name]
        width = h['shape'][-1] if len(h['shape']) >= 2 else 1
        step = max(width, chunk_elements // width * width)
        count = math.prod(h['shape'])
        dtype = '<f4' if h['dtype'] == 'F32' else '<u2' if h['dtype'] == 'BF16' else '<f2'
        # pread-style bounded reads, not mmap: no accumulating resident mapped pages.
        with self.path.open('rb') as f:
            f.seek(self.base + h['data_offsets'][0])
            for pos in range(0, count, step):
                size = min(step, count - pos)
                data = f.read(size * np.dtype(dtype).itemsize)
                if len(data) != size * np.dtype(dtype).itemsize:
                    raise ValueError('checkpoint truncated during conversion')
                x = np.frombuffer(data, dtype=dtype)
                if h['dtype'] == 'BF16':
                    x = (x.astype(np.uint32) << 16).view(np.float32)
                else:
                    x = x.astype(np.float32)
                if not np.all(np.isfinite(x)):
                    raise ValueError(f'{name}: NaN or infinity')
                yield x


class TorchSource:
    def __init__(self, path):
        try:
            import torch
        except ImportError as e:
            raise RuntimeError('PyTorch checkpoints require optional dependency: pip install torch') from e
        self.torch = torch
        self.state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if 'state_dict' in self.state:
            self.state = self.state['state_dict']

    def entries(self):
        return [(n, tuple(t.shape)) for n, t in self.state.items()]

    def chunks(self, name, chunk_elements):
        t = self.state[name]
        if not t.is_floating_point() or not t.is_contiguous():
            raise ValueError(f'{name}: expected contiguous floating tensor')
        width = t.shape[-1] if t.ndim >= 2 else 1
        step = max(width, chunk_elements // width * width)
        flat = t.view(-1)
        for i in range(0, flat.numel(), step):
            x = flat[i:i + step].float().numpy()
            if not np.all(np.isfinite(x)):
                raise ValueError(f'{name}: NaN or infinity')
            yield x


def structural_mask(x):
    """Keep two largest magnitudes per contiguous four; stable tie breaking."""
    groups = np.abs(x).reshape(-1, 4)
    indices = np.argsort(groups, axis=1, kind='stable')[:, 2:]
    mask = np.zeros_like(groups, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=1)
    return mask.ravel()


def states(x, alpha, sparse=True):
    if sparse:
        mask = structural_mask(x)
        # Pruned entries are zero. Nonzero survivors choose 1 or 2 at 1.5*alpha.
        # True zero survivors remain zero: zero sparsity is AT LEAST 50%.
        q = np.where(np.abs(x) >= 1.5 * alpha, 2, 1) * np.sign(x)
        return np.where(mask, q, 0).astype(np.int8)
    # Nearest of five states, thresholds 0.5 alpha and 1.5 alpha.
    return (np.minimum(2, np.floor(np.abs(x) / alpha + 0.5)) * np.sign(x)).astype(np.int8)


def fit_alpha(chunks, sparse, iterations):
    total, count = 0.0, 0
    for x in chunks():
        active = np.abs(x[structural_mask(x)]) if sparse else np.abs(x)
        total += active.sum(dtype=np.float64)
        count += active.size
    alpha = max(total / max(count, 1) / 1.5, 2 ** -24)
    for _ in range(iterations):
        numerator, denominator = 0.0, 0.0
        for x in chunks():
            q = states(x, alpha, sparse)
            numerator += np.sum(x * q, dtype=np.float64)
            denominator += np.sum(q.astype(np.int32) ** 2, dtype=np.int64)
        if denominator:
            alpha = max(numerator / denominator, 2 ** -24)
    if not np.isfinite(alpha) or alpha > 65504:
        raise ValueError('scale cannot be represented in FP16')
    return float(np.float16(alpha))


def discover(path):
    path = Path(path)
    if path.is_file():
        return [path]
    for index in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        if (path / index).exists():
            names = sorted(set(json.loads((path / index).read_text())['weight_map'].values()))
            files = [(path / n).resolve() for n in names]
            if any(not p.is_relative_to(path.resolve()) for p in files):
                raise ValueError('checkpoint index escapes source directory')
            return files
    files = sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
    if not files:
        raise ValueError('no checkpoint shards found')
    return files


def quantize(source, output, chunk_elements=262144, iterations=3, sparse=True):
    if chunk_elements < 4 or not 0 <= iterations <= 20:
        raise ValueError('invalid chunk size or iteration count')
    files = discover(source)
    output = Path(output)
    if output.resolve() in [p.resolve() for p in files]:
        raise ValueError('output cannot overwrite checkpoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Spool payload to disk; never accumulate quantized layers in RAM.
    tensors, seen, statistics = [], set(), []
    tmpname = None
    try:
        with tempfile.TemporaryFile(dir=output.parent) as payload:
            for path in files:
                reader = SafeSource(path) if path.suffix == '.safetensors' else TorchSource(path)
                for name, shape in reader.entries():
                    if (name in seen or not shape or len(shape) > 8 or min(shape) <= 0
                            or len(name.encode()) > 4096 or '\0' in name):
                        raise ValueError(f'invalid/duplicate tensor {name}')
                    seen.add(name)
                    kind = PACKED if len(shape) >= 2 else FP16
                    if kind == PACKED and sparse and shape[-1] % 4:
                        raise ValueError(f'{name}: 2:4 sparsity requires last dimension divisible by four; use --dense')
                    chunks = lambda: reader.chunks(name, chunk_elements)
                    alpha = fit_alpha(chunks, sparse, iterations) if kind == PACKED else 1.0
                    tensors.append(TensorInfo(name, shape, kind, alpha, int(sparse and kind == PACKED)))
                    zeros, count, square_error, square_signal = 0, 0, 0.0, 0.0
                    carry = np.empty(0, dtype=np.int8)
                    for x in chunks():
                        if kind == FP16:
                            if np.any(np.abs(x) > 65504):
                                raise ValueError(f'{name}: FP16 overflow')
                            payload.write(x.astype('<f2').tobytes())
                            continue
                        q = states(x, alpha, sparse)
                        zeros += int(np.count_nonzero(q == 0))
                        count += q.size
                        square_error += np.sum((x - alpha * q) ** 2, dtype=np.float64)
                        square_signal += np.sum(x ** 2, dtype=np.float64)
                        joined = np.concatenate((carry, q))
                        limit = joined.size // 3 * 3
                        payload.write(pack_pentanary(joined[:limit]).tobytes())
                        carry = joined[limit:].copy()
                    if carry.size:
                        payload.write(pack_pentanary(carry).tobytes())
                    statistics.append(dict(name=name, shape=shape, alpha=alpha,
                                           zero_fraction=zeros / count if count else None,
                                           relative_rmse=math.sqrt(square_error / square_signal) if square_signal else 0))
                    print(f'{name}: {shape}, alpha={alpha:g}, zeros={zeros}/{count}', flush=True)
                del reader
            if not tensors:
                raise ValueError('empty checkpoint')
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as out:
                tmpname = out.name
                write_directory(out, tensors)
                payload.seek(0)
                while block := payload.read(1 << 20):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmpname, output)
            tmpname = None
    finally:
        if tmpname:
            os.unlink(tmpname)
    report = {'format': 'PENTA5-v1', 'bytes': output.stat().st_size, 'tensors': statistics}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', help='local safetensors / torch file or HF checkpoint directory')
    p.add_argument('-o', '--output', default='model_pentanary.bin')
    p.add_argument('--chunk-elements', type=int, default=262144)
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--dense', action='store_true', help='disable 2:4 pruning')
    args = p.parse_args()
    quantize(args.checkpoint, args.output, args.chunk_elements, args.iterations, not args.dense)


if __name__ == '__main__':
    main()
