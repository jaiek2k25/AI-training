"""PENTA5 v1 little-endian container and reference packing."""
from dataclasses import dataclass
import math
import os
import struct
import numpy as np

MAGIC = b'PENTA5\0\0'
HEADER = struct.Struct('<8sIIQ')
RECORD = struct.Struct('<HBBHHQQ')
PACKED, FP16 = 1, 2


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    kind: int
    alpha: float = 1.0
    flags: int = 0
    offset: int = 0
    nbytes: int = 0

    @property
    def size(self):
        return math.prod(self.shape)


def pack_pentanary(values):
    q = np.asarray(values).ravel()
    if not np.all(np.isin(q, [-2, -1, 0, 1, 2])):
        raise ValueError('weights must be integers in [-2, 2]')
    digits = np.full(((q.size + 2) // 3) * 3, 2, dtype=np.uint8)
    digits[:q.size] = q.astype(np.int8) + 2
    d = digits.reshape(-1, 3)
    return d[:, 0] + 5 * d[:, 1] + 25 * d[:, 2]


def unpack_pentanary(data, count):
    b = np.frombuffer(data, dtype=np.uint8)
    if count < 0 or count > b.size * 3 or np.any(b >= 125):
        raise ValueError('invalid packed payload')
    return (np.stack((b % 5, b // 5 % 5, b // 25), axis=1)
            .astype(np.int8).ravel()[:count] - 2)


def write_directory(f, tensors):
    start = HEADER.size + sum(RECORD.size + 8 * len(t.shape) + len(t.name.encode()) for t in tensors)
    start = (start + 63) // 64 * 64
    f.seek(0)
    f.write(HEADER.pack(MAGIC, 1, len(tensors), start))
    offset = 0
    for t in tensors:
        t.offset = offset
        t.nbytes = (t.size + 2) // 3 if t.kind == PACKED else t.size * 2
        name = t.name.encode('utf-8')
        alpha_bits = struct.unpack('<H', struct.pack('<e', t.alpha))[0]
        f.write(RECORD.pack(len(name), t.kind, len(t.shape), alpha_bits, t.flags, offset, t.nbytes))
        f.write(struct.pack('<' + 'Q' * len(t.shape), *t.shape))
        f.write(name)
        offset += t.nbytes
    f.write(b'\0' * (start - f.tell()))
    return start


def read_directory(path):
    def exact(f, n):
        data = f.read(n)
        if len(data) != n:
            raise ValueError('truncated container')
        return data
    length = os.path.getsize(path)
    with open(path, 'rb') as f:
        magic, version, count, start = HEADER.unpack(exact(f, HEADER.size))
        if magic != MAGIC or version != 1 or not 0 < count <= 100000 or not HEADER.size <= start <= min(length, 64 << 20):
            raise ValueError('invalid PENTA5 header')
        tensors = {}
        end = 0
        for _ in range(count):
            n, kind, rank, alpha, flags, offset, size = RECORD.unpack(exact(f, RECORD.size))
            if not 0 < n <= 4096 or not 1 <= rank <= 8 or kind not in (PACKED, FP16) or flags & ~1:
                raise ValueError('invalid tensor descriptor')
            shape = struct.unpack('<' + 'Q' * rank, exact(f, 8 * rank))
            name = exact(f, n).decode('utf-8')
            scale = struct.unpack('<e', struct.pack('<H', alpha))[0]
            t = TensorInfo(name, shape, kind, scale, flags, offset, size)
            expected = (t.size + 2) // 3 if kind == PACKED else t.size * 2
            if (name in tensors or '\0' in name or min(shape) == 0 or t.size > (1 << 40)
                    or not math.isfinite(scale) or scale <= 0 or size != expected
                    or offset < end or start + offset + size > length or f.tell() > start):
                raise ValueError('invalid tensor bounds, scale, or duplicate name')
            end = offset + size
            tensors[name] = t
    return start, tensors
