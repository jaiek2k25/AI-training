"""ctypes interface: no torch, no dense weight materialization."""
import ctypes as C
from pathlib import Path
import numpy as np
from penta_format import read_directory


class NativeModel:
    def __init__(self, path, library=None):
        self.handle = None
        self.data_start, self.tensors = read_directory(path)
        self.lib = C.CDLL(str(Path(library or Path(__file__).with_name('libpentanary.so')).resolve()))
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
        self.lib.penta_open.argtypes = [C.c_char_p]
        self.lib.penta_open.restype = C.c_void_p
        self.lib.penta_close.argtypes = [C.c_void_p]
        self.lib.penta_close.restype = None
        self.lib.penta_error.restype = C.c_char_p
        self.lib.penta_error.argtypes = []
        self.lib.penta_matvec.argtypes = [C.c_void_p, C.c_char_p, pointer, C.c_uint64, pointer, C.c_uint64]
        self.lib.penta_matvec.restype = C.c_int
        self.lib.penta_row.argtypes = [C.c_void_p, C.c_char_p, C.c_uint64, pointer, C.c_uint64]
        self.lib.penta_row.restype = C.c_int
        self.handle = self.lib.penta_open(str(path).encode())
        if not self.handle:
            raise ValueError(self.lib.penta_error().decode())

    def _check(self, code):
        if code:
            raise ValueError(self.lib.penta_error().decode())

    def matvec(self, name, x):
        shape = self.tensors[name].shape
        x = np.ascontiguousarray(x, dtype=np.float32)
        if len(shape) != 2 or x.ndim != 1 or x.size != shape[1]:
            raise ValueError('invalid matvec input shape')
        out = np.empty(shape[0], dtype=np.float32)
        self._check(self.lib.penta_matvec(self.handle, name.encode(), x, x.size, out, out.size))
        return out

    def row(self, name, index=0):
        shape = self.tensors[name].shape
        out = np.empty(shape[-1], dtype=np.float32)
        self._check(self.lib.penta_row(self.handle, name.encode(), index, out, out.size))
        return out

    def close(self):
        if self.handle:
            self.lib.penta_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
