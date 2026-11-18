#!/usr/bin/env python3
"""
glm53_torch_compat.py
=====================
Zero-overhead import helper that imports native `torch` when installed (e.g. PyTorch 2.11+cu128
on the user's Windows 11 RTX 5090 workstation), and provides a NumPy-backed `torch` API
execution shim ONLY in environments where PyTorch is not installed (such as a CPU-only CI
or sandbox container) so every single line, shape, index, and equation of the PyTorch
codebase can be executed and verified end-to-end without dead code branches.
"""

from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import scipy.linalg

try:
    import torch as _real_torch
    import torch.nn as _real_nn
    import torch.nn.functional as _real_F
    HAS_REAL_TORCH = True
except ImportError:
    _real_torch = None
    _real_nn = None
    _real_F = None
    HAS_REAL_TORCH = False


if HAS_REAL_TORCH:
    torch = _real_torch
    nn = _real_nn
    F = _real_F
else:
    class _DType:
        def __init__(self, name: str, np_dtype: Any, itemsize: int, is_floating: bool = True):
            self.name = name
            self.np_dtype = np.dtype(np_dtype)
            self.itemsize = itemsize
            self.is_floating_point = is_floating

        def __repr__(self) -> str:
            return f"torch.{self.name}"

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, _DType):
                return self.name == other.name
            return False

        def __hash__(self) -> int:
            return hash(self.name)

    float32 = _DType("float32", np.float32, 4, True)
    float16 = _DType("float16", np.float16, 2, True)
    bfloat16 = _DType("bfloat16", np.float32, 2, True)       # Stored in fp32 math, 2-byte accounting
    float8_e4m3fn = _DType("float8_e4m3fn", np.uint8, 1, True)
    int64 = _DType("int64", np.int64, 8, False)
    long = int64
    int32 = _DType("int32", np.int32, 4, False)
    int16 = _DType("int16", np.int16, 2, False)
    int8 = _DType("int8", np.int8, 1, False)
    uint16 = _DType("uint16", np.uint16, 2, False)
    uint8 = _DType("uint8", np.uint8, 1, False)
    bool_dt = _DType("bool", np.bool_, 1, False)

    _NP_TO_DT = {
        np.dtype(np.float32): float32,
        np.dtype(np.float64): float32,
        np.dtype(np.float16): float16,
        np.dtype(np.int64): int64,
        np.dtype(np.int32): int32,
        np.dtype(np.int16): int16,
        np.dtype(np.int8): int8,
        np.dtype(np.uint16): uint16,
        np.dtype(np.uint8): uint8,
        np.dtype(np.bool_): bool_dt,
    }

    class device:
        def __init__(self, dev: Union[str, "device"] = "cpu"):
            if isinstance(dev, device):
                self.type = dev.type
            else:
                s = str(dev).split(":")[0].lower()
                self.type = "cpu" if s in ("cpu", "cuda") else s

        def __repr__(self) -> str:
            return f"device(type='{self.type}')"

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, device):
                return self.type == other.type
            if isinstance(other, str):
                return self.type == other.split(":")[0]
            return False

    class Generator:
        def __init__(self, device: Any = "cpu"):
            self.rng = np.random.default_rng(0)

        def manual_seed(self, seed: int) -> "Generator":
            self.rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
            return self

    class _TopKResult:
        def __init__(self, values: "Tensor", indices: "Tensor"):
            self.values = values
            self.indices = indices

        def __iter__(self):
            yield self.values
            yield self.indices

    def _fp8_e4m3_encode_np(x_f32: np.ndarray) -> np.ndarray:
        """Vectorized IEEE-like E4M3 (1 sign, 4 exp bias 7, 3 mantissa) encoder into uint8."""
        x = np.asarray(x_f32, dtype=np.float32)
        sign = (x < 0).astype(np.uint8) << 7
        ax = np.clip(np.abs(x), 0.0, 448.0)
        out = np.zeros(x.shape, dtype=np.uint8)
        nz = ax > 0
        if np.any(nz):
            v = ax[nz]
            exp_unbiased = np.floor(np.log2(np.maximum(v, 2.0 ** -9))).astype(np.int32)
            exp_biased = np.clip(exp_unbiased + 7, 0, 15)
            sub = exp_biased == 0
            mant = np.zeros_like(exp_biased)
            if np.any(sub):
                mant[sub] = np.clip(np.round(v[sub] / (2.0 ** -9)).astype(np.int32), 0, 7)
            if np.any(~sub):
                scale = np.power(2.0, exp_biased[~sub] - 7)
                m_f = (v[~sub] / scale - 1.0) * 8.0
                m_r = np.round(m_f).astype(np.int32)
                carry = m_r >= 8
                exp_b = exp_biased[~sub].copy()
                exp_b[carry] += 1
                m_r[carry] = 0
                exp_b = np.clip(exp_b, 0, 15)
                exp_biased[~sub] = exp_b
                mant[~sub] = np.clip(m_r, 0, 7)
            out[nz] = ((exp_biased.astype(np.uint8) << 3) | mant.astype(np.uint8))
        return sign | out

    def _fp8_e4m3_decode_np(u8: np.ndarray) -> np.ndarray:
        u = np.asarray(u8, dtype=np.uint8)
        sign = np.where((u >> 7) != 0, -1.0, 1.0).astype(np.float32)
        exp = ((u >> 3) & 0x0F).astype(np.int32)
        mant = (u & 0x07).astype(np.float32)
        val = np.where(
            exp == 0,
            (mant / 8.0) * (2.0 ** -6),
            (1.0 + mant / 8.0) * np.power(2.0, exp - 7),
        ).astype(np.float32)
        return sign * val

    class Tensor:
        def __init__(self, arr: Any, dtype: Optional[_DType] = None, dev: Any = "cpu"):
            if isinstance(arr, Tensor):
                self._arr = arr._arr
                self.dtype = dtype or arr.dtype
                self.device = device(dev) if dev is not None else arr.device
                return
            np_a = np.asarray(arr)
            if dtype is None:
                if np_a.dtype == np.float64:
                    np_a = np_a.astype(np.float32)
                dtype = _NP_TO_DT.get(np_a.dtype, float32)
            self.dtype = dtype
            self.device = device(dev)
            if dtype == bfloat16:
                # Truncate mantissa to 7 bits (bfloat16 precision) while keeping float32 array
                f32 = np_a.astype(np.float32)
                u32 = f32.view(np.uint32)
                u32_bf16 = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) & 0xFFFF0000)
                self._arr = u32_bf16.view(np.float32).copy()
            elif dtype == float8_e4m3fn:
                if np_a.dtype == np.uint8:
                    self._arr = np_a.copy()
                else:
                    self._arr = _fp8_e4m3_encode_np(np_a)
            else:
                self._arr = np_a.astype(dtype.np_dtype, copy=False)

        @property
        def shape(self) -> Tuple[int, ...]:
            return self._arr.shape

        @property
        def ndim(self) -> int:
            return self._arr.ndim

        @property
        def is_cuda(self) -> bool:
            return False

        @property
        def T(self) -> "Tensor":
            return Tensor(self._arr.T, dtype=self.dtype, dev=self.device)

        def t(self) -> "Tensor":
            return self.T

        def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
            return self.shape if dim is None else self.shape[dim]

        def numel(self) -> int:
            return int(self._arr.size)

        def element_size(self) -> int:
            return self.dtype.itemsize

        def item(self) -> Any:
            return self._arr.item()

        def tolist(self) -> Any:
            return self._arr.tolist()

        def numpy(self) -> np.ndarray:
            if self.dtype == float8_e4m3fn:
                return self._arr
            return self._arr

        def detach(self) -> "Tensor":
            return self

        def cpu(self) -> "Tensor":
            return self

        def pin_memory(self) -> "Tensor":
            return self

        def contiguous(self) -> "Tensor":
            return Tensor(np.ascontiguousarray(self._arr), dtype=self.dtype, dev=self.device)

        def clone(self) -> "Tensor":
            return Tensor(self._arr.copy(), dtype=self.dtype, dev=self.device)

        def to(self, *args: Any, **kwargs: Any) -> "Tensor":
            target_dt = kwargs.get("dtype", None)
            target_dev = kwargs.get("device", self.device)
            for a in args:
                if isinstance(a, _DType):
                    target_dt = a
                elif isinstance(a, (str, device)):
                    target_dev = a
            if target_dt is None or target_dt == self.dtype:
                return Tensor(self._arr, dtype=self.dtype, dev=target_dev)
            if self.dtype == float8_e4m3fn:
                dec = _fp8_e4m3_decode_np(self._arr)
                return Tensor(dec, dtype=target_dt, dev=target_dev)
            if target_dt == float8_e4m3fn:
                return Tensor(_fp8_e4m3_encode_np(self._arr.astype(np.float32)), dtype=float8_e4m3fn, dev=target_dev)
            return Tensor(self._arr, dtype=target_dt, dev=target_dev)

        def float(self) -> "Tensor":
            return self.to(float32)

        def half(self) -> "Tensor":
            return self.to(float16)

        def bfloat16(self) -> "Tensor":
            return self.to(bfloat16)

        def long(self) -> "Tensor":
            return self.to(int64)

        def view(self, *shape: Any) -> "Tensor":
            if len(shape) == 1 and isinstance(shape[0], _DType):
                dt = shape[0]
                if dt == float8_e4m3fn and self.dtype == uint8:
                    return Tensor(self._arr, dtype=float8_e4m3fn, dev=self.device)
                if dt == uint8:
                    if self.dtype == bfloat16:
                        u16 = (self._arr.view(np.uint32) >> 16).astype(np.uint16)
                        return Tensor(np.ascontiguousarray(u16).view(np.uint8), dtype=uint8, dev=self.device)
                    return Tensor(np.ascontiguousarray(self._arr).view(np.uint8), dtype=uint8, dev=self.device)
                raw = np.ascontiguousarray(self._arr).view(dt.np_dtype)
                return Tensor(raw, dtype=dt, dev=self.device)
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(self._arr.reshape(*shape), dtype=self.dtype, dev=self.device)

        def reshape(self, *shape: Any) -> "Tensor":
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(self._arr.reshape(*shape), dtype=self.dtype, dev=self.device)

        def permute(self, *dims: Any) -> "Tensor":
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = tuple(dims[0])
            return Tensor(np.transpose(self._arr, dims), dtype=self.dtype, dev=self.device)

        def transpose(self, dim0: int, dim1: int) -> "Tensor":
            axes = list(range(self.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            return Tensor(np.transpose(self._arr, axes), dtype=self.dtype, dev=self.device)

        def unsqueeze(self, dim: int) -> "Tensor":
            return Tensor(np.expand_dims(self._arr, axis=dim), dtype=self.dtype, dev=self.device)

        def squeeze(self, dim: Optional[int] = None) -> "Tensor":
            if dim is None:
                return Tensor(np.squeeze(self._arr), dtype=self.dtype, dev=self.device)
            if self.shape[dim] == 1:
                return Tensor(np.squeeze(self._arr, axis=dim), dtype=self.dtype, dev=self.device)
            return self

        def expand(self, *sizes: Any) -> "Tensor":
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            target = [self.shape[i] if s == -1 else s for i, s in enumerate(sizes)]
            return Tensor(np.broadcast_to(self._arr, target), dtype=self.dtype, dev=self.device)

        def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
            if start_dim < 0:
                start_dim += self.ndim
            if end_dim < 0:
                end_dim += self.ndim
            new_shape = list(self.shape[:start_dim]) + [-1] + list(self.shape[end_dim + 1:])
            return self.reshape(*new_shape)

        def flip(self, dims: Union[int, List[int], Tuple[int, ...]]) -> "Tensor":
            return Tensor(np.flip(self._arr, axis=dims).copy(), dtype=self.dtype, dev=self.device)

        def repeat_interleave(self, repeats: int, dim: Optional[int] = None) -> "Tensor":
            return Tensor(np.repeat(self._arr, repeats, axis=dim), dtype=self.dtype, dev=self.device)

        def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "Tensor":
            return Tensor(np.clip(self._arr, min, max), dtype=self.dtype, dev=self.device)

        def clamp_min(self, min_val: float) -> "Tensor":
            return Tensor(np.maximum(self._arr, min_val), dtype=self.dtype, dev=self.device)

        def abs(self) -> "Tensor":
            return Tensor(np.abs(self._arr), dtype=self.dtype, dev=self.device)

        def pow(self, exp: float) -> "Tensor":
            return Tensor(np.power(self._arr.astype(np.float32), exp), dtype=float32 if self.dtype.is_floating_point else self.dtype)

        def sqrt(self) -> "Tensor":
            return Tensor(np.sqrt(self._arr.astype(np.float32)), dtype=self.dtype if self.dtype.is_floating_point else float32)

        def exp(self) -> "Tensor":
            return Tensor(np.exp(self._arr.astype(np.float32)), dtype=self.dtype if self.dtype.is_floating_point else float32)

        def sum(self, dim: Optional[Any] = None, keepdim: bool = False, dtype: Optional[_DType] = None) -> "Tensor":
            arr = self._arr.astype(np.float32) if self.dtype == float16 else self._arr
            res = np.sum(arr, axis=dim, keepdims=keepdim)
            out_dt = dtype or (int64 if not self.dtype.is_floating_point else self.dtype)
            return Tensor(res, dtype=out_dt, dev=self.device)

        def mean(self, dim: Optional[Any] = None, keepdim: bool = False) -> "Tensor":
            res = np.mean(self._arr.astype(np.float32), axis=dim, keepdims=keepdim)
            return Tensor(res, dtype=self.dtype if self.dtype.is_floating_point else float32, dev=self.device)

        def amax(self, dim: Optional[Any] = None, keepdim: bool = False) -> "Tensor":
            res = np.amax(self._arr, axis=dim, keepdims=keepdim)
            return Tensor(res, dtype=self.dtype, dev=self.device)

        def norm(self, dim: Optional[Any] = None, keepdim: bool = False) -> "Tensor":
            res = np.linalg.norm(self._arr.astype(np.float32), axis=dim, keepdims=keepdim)
            return Tensor(res, dtype=float32, dev=self.device)

        def argmax(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
            res = np.argmax(self._arr, axis=dim)
            if keepdim and dim is not None:
                res = np.expand_dims(res, axis=dim)
            return Tensor(res, dtype=int64, dev=self.device)

        def argmin(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
            res = np.argmin(self._arr, axis=dim)
            if keepdim and dim is not None:
                res = np.expand_dims(res, axis=dim)
            return Tensor(res, dtype=int64, dev=self.device)

        def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> _TopKResult:
            return topk(self, k=k, dim=dim, largest=largest, sorted=sorted)

        def remainder(self, divisor: Any) -> "Tensor":
            d = divisor._arr if isinstance(divisor, Tensor) else divisor
            return Tensor(np.remainder(self._arr, d), dtype=self.dtype, dev=self.device)

        def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
            out = self._arr.copy()
            out[mask._arr.astype(bool)] = value
            return Tensor(out, dtype=self.dtype, dev=self.device)

        def gather(self, dim: int, index: "Tensor") -> "Tensor":
            return gather(self, dim, index)

        def scatter_(self, dim: int, index: "Tensor", src: Any) -> "Tensor":
            s_val = src._arr if isinstance(src, Tensor) else src
            np.put_along_axis(self._arr, index._arr.astype(np.int64), s_val, axis=dim)
            return self

        def index_select(self, dim: int, index: "Tensor") -> "Tensor":
            return Tensor(np.take(self._arr, index._arr.astype(np.int64), axis=dim), dtype=self.dtype, dev=self.device)

        def index_add_(self, dim: int, index: "Tensor", source: "Tensor") -> "Tensor":
            idx = index._arr.astype(np.int64)
            if dim == 0:
                np.add.at(self._arr, idx, source._arr.astype(self._arr.dtype))
            else:
                slc = [slice(None)] * self.ndim
                slc[dim] = idx
                np.add.at(self._arr, tuple(slc), source._arr.astype(self._arr.dtype))
            return self

        def mul_(self, other: Any) -> "Tensor":
            o = other._arr if isinstance(other, Tensor) else other
            self._arr[...] = self._arr * o
            return self

        def add_(self, other: Any) -> "Tensor":
            o = other._arr if isinstance(other, Tensor) else other
            self._arr[...] = self._arr + o
            return self

        def addmm_(self, mat1: "Tensor", mat2: "Tensor") -> "Tensor":
            self._arr[...] = self._arr + np.matmul(mat1._arr.astype(np.float32), mat2._arr.astype(np.float32))
            return self

        def copy_(self, src: "Tensor", non_blocking: bool = False) -> "Tensor":
            self._arr[...] = src._arr
            return self

        def new_zeros(self, size: Any, dtype: Optional[_DType] = None) -> "Tensor":
            return zeros(size, dtype=dtype or self.dtype, device=self.device)

        def new_empty(self, size: Any, dtype: Optional[_DType] = None) -> "Tensor":
            return empty(size, dtype=dtype or self.dtype, device=self.device)

        def any(self) -> "Tensor":
            return Tensor(np.any(self._arr), dtype=bool_dt, dev=self.device)

        def all(self, dim: Optional[int] = None) -> "Tensor":
            return Tensor(np.all(self._arr, axis=dim), dtype=bool_dt, dev=self.device)

        def __bool__(self) -> bool:
            return bool(self._arr.item())

        def __len__(self) -> int:
            return len(self._arr)

        def _conv_idx(self, idx: Any) -> Any:
            if isinstance(idx, Tensor):
                return idx._arr
            if isinstance(idx, tuple):
                return tuple(i._arr if isinstance(i, Tensor) else i for i in idx)
            return idx

        def __getitem__(self, idx: Any) -> "Tensor":
            return Tensor(self._arr[self._conv_idx(idx)], dtype=self.dtype, dev=self.device)

        def __setitem__(self, idx: Any, val: Any) -> None:
            v = val._arr if isinstance(val, Tensor) else val
            self._arr[self._conv_idx(idx)] = v

        def _binop(self, other: Any, op: Any) -> "Tensor":
            o = other._arr if isinstance(other, Tensor) else other
            res = op(self._arr, o)
            out_dt = self.dtype
            if isinstance(other, Tensor) and other.dtype == float32:
                out_dt = float32
            elif isinstance(res, np.ndarray) and res.dtype == np.bool_:
                out_dt = bool_dt
            return Tensor(res, dtype=out_dt, dev=self.device)

        def __add__(self, o: Any) -> "Tensor": return self._binop(o, np.add)
        def __radd__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.add(b, a))
        def __sub__(self, o: Any) -> "Tensor": return self._binop(o, np.subtract)
        def __rsub__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.subtract(b, a))
        def __mul__(self, o: Any) -> "Tensor": return self._binop(o, np.multiply)
        def __rmul__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.multiply(b, a))
        def __truediv__(self, o: Any) -> "Tensor": return self._binop(o, np.true_divide)
        def __rtruediv__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.true_divide(b, a))
        def __floordiv__(self, o: Any) -> "Tensor": return self._binop(o, np.floor_divide)
        def __rfloordiv__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.floor_divide(b, a))
        def __mod__(self, o: Any) -> "Tensor": return self._binop(o, np.mod)
        def __rmod__(self, o: Any) -> "Tensor": return self._binop(o, lambda a, b: np.mod(b, a))
        def __pow__(self, o: Any) -> "Tensor": return self._binop(o, np.power)
        def __neg__(self) -> "Tensor": return Tensor(-self._arr, dtype=self.dtype, dev=self.device)
        def __invert__(self) -> "Tensor": return Tensor(~self._arr, dtype=self.dtype, dev=self.device)
        def __and__(self, o: Any) -> "Tensor": return self._binop(o, np.logical_and if self.dtype == bool_dt else np.bitwise_and)
        def __or__(self, o: Any) -> "Tensor": return self._binop(o, np.logical_or if self.dtype == bool_dt else np.bitwise_or)
        def __lt__(self, o: Any) -> "Tensor": return self._binop(o, np.less)
        def __le__(self, o: Any) -> "Tensor": return self._binop(o, np.less_equal)
        def __gt__(self, o: Any) -> "Tensor": return self._binop(o, np.greater)
        def __ge__(self, o: Any) -> "Tensor": return self._binop(o, np.greater_equal)
        def __eq__(self, o: Any) -> "Tensor":  # type: ignore[override]
            return self._binop(o, np.equal)
        def __ne__(self, o: Any) -> "Tensor":  # type: ignore[override]
            return self._binop(o, np.not_equal)
        def __matmul__(self, o: "Tensor") -> "Tensor":
            return matmul(self, o)

    def no_grad():
        class _NG:
            def __enter__(self): return None
            def __exit__(self, *a): return False
            def __call__(self, fn): return fn
        return _NG()

    def inference_mode():
        return no_grad()

    def tensor(data: Any, dtype: Optional[_DType] = None, device: Any = "cpu") -> Tensor:
        return Tensor(np.array(data), dtype=dtype, dev=device)

    def from_numpy(arr: np.ndarray) -> Tensor:
        dt = _NP_TO_DT.get(arr.dtype, float32)
        t = Tensor.__new__(Tensor)
        t._arr = arr
        t.dtype = dt
        t.device = device("cpu")
        return t

    def zeros(size: Any, *more: int, dtype: _DType = float32, device: Any = "cpu") -> Tensor:
        shp = (size, *more) if isinstance(size, int) else tuple(size)
        return Tensor(np.zeros(shp, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def ones(size: Any, *more: int, dtype: _DType = float32, device: Any = "cpu") -> Tensor:
        shp = (size, *more) if isinstance(size, int) else tuple(size)
        return Tensor(np.ones(shp, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def empty(size: Any, *more: int, dtype: _DType = float32, device: Any = "cpu", pin_memory: bool = False) -> Tensor:
        shp = (size, *more) if isinstance(size, int) else tuple(size)
        return Tensor(np.empty(shp, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def full(size: Any, fill_value: Any, dtype: _DType = float32, device: Any = "cpu") -> Tensor:
        shp = (size,) if isinstance(size, int) else tuple(size)
        return Tensor(np.full(shp, fill_value, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def zeros_like(t: Tensor, dtype: Optional[_DType] = None, device: Any = None, **kwargs: Any) -> Tensor:
        dt = dtype or t.dtype
        return Tensor(np.zeros(t.shape, dtype=dt.np_dtype), dtype=dt, dev=device or t.device)

    def ones_like(t: Tensor, dtype: Optional[_DType] = None, device: Any = None, **kwargs: Any) -> Tensor:
        dt = dtype or t.dtype
        return Tensor(np.ones(t.shape, dtype=dt.np_dtype), dtype=dt, dev=device or t.device)

    def empty_like(t: Tensor, dtype: Optional[_DType] = None, device: Any = None, **kwargs: Any) -> Tensor:
        dt = dtype or t.dtype
        return Tensor(np.empty(t.shape, dtype=dt.np_dtype), dtype=dt, dev=device or t.device)

    def full_like(t: Tensor, fill_value: Any, dtype: Optional[_DType] = None, device: Any = None, **kwargs: Any) -> Tensor:
        dt = dtype or t.dtype
        return Tensor(np.full(t.shape, fill_value, dtype=dt.np_dtype), dtype=dt, dev=device or t.device)

    def eye(n: int, dtype: _DType = float32, device: Any = "cpu") -> Tensor:
        return Tensor(np.eye(n, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def arange(*args: Any, dtype: _DType = int64, device: Any = "cpu") -> Tensor:
        return Tensor(np.arange(*args, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def randn(size: Any, *more: int, generator: Optional[Generator] = None, device: Any = "cpu", dtype: _DType = float32) -> Tensor:
        shp = (size, *more) if isinstance(size, int) else tuple(size)
        rng = generator.rng if generator is not None else np.random.default_rng()
        return Tensor(rng.standard_normal(shp).astype(np.float32), dtype=dtype, dev=device)

    def rand(size: Any, *more: int, generator: Optional[Generator] = None, device: Any = "cpu", dtype: _DType = float32) -> Tensor:
        shp = (size, *more) if isinstance(size, int) else tuple(size)
        rng = generator.rng if generator is not None else np.random.default_rng()
        return Tensor(rng.random(shp).astype(np.float32), dtype=dtype, dev=device)

    def randint(low: int, high: int, size: Tuple[int, ...], generator: Optional[Generator] = None, dtype: _DType = int64, device: Any = "cpu") -> Tensor:
        rng = generator.rng if generator is not None else np.random.default_rng()
        return Tensor(rng.integers(low, high, size=size, dtype=dtype.np_dtype), dtype=dtype, dev=device)

    def randperm(n: int, generator: Optional[Generator] = None, device: Any = "cpu", dtype: _DType = int64) -> Tensor:
        rng = generator.rng if generator is not None else np.random.default_rng()
        return Tensor(rng.permutation(n).astype(dtype.np_dtype), dtype=dtype, dev=device)

    def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
        return Tensor(np.concatenate([t._arr for t in tensors], axis=dim), dtype=tensors[0].dtype, dev=tensors[0].device)

    def stack(tensors: Union[List[Tensor], Tuple[Tensor, ...]], dim: int = 0) -> Tensor:
        return Tensor(np.stack([t._arr for t in tensors], axis=dim), dtype=tensors[0].dtype, dev=tensors[0].device)

    def matmul(a: Tensor, b: Tensor) -> Tensor:
        af = a._arr.astype(np.float32) if a.dtype == float16 else a._arr
        bf = b._arr.astype(np.float32) if b.dtype == float16 else b._arr
        res = np.matmul(af, bf)
        return Tensor(res, dtype=a.dtype, dev=a.device)

    def mm(a: Tensor, b: Tensor) -> Tensor:
        return matmul(a, b)

    def mv(a: Tensor, v: Tensor) -> Tensor:
        return matmul(a, v)

    def bmm(a: Tensor, b: Tensor) -> Tensor:
        return matmul(a, b)

    def einsum(eq: str, *operands: Tensor) -> Tensor:
        arrs = [op._arr.astype(np.float32) for op in operands]
        res = np.einsum(eq, *arrs, optimize=True)
        return Tensor(res, dtype=operands[0].dtype, dev=operands[0].device)

    def sum(t: Tensor, dim: Optional[Any] = None, keepdim: bool = False) -> Tensor:
        return t.sum(dim=dim, keepdim=keepdim)

    def max(t: Tensor) -> Tensor:
        return Tensor(np.max(t._arr), dtype=t.dtype, dev=t.device)

    def abs(t: Tensor) -> Tensor:
        return t.abs()

    def sqrt(t: Tensor) -> Tensor:
        return t.sqrt()

    def rsqrt(t: Tensor) -> Tensor:
        return Tensor(1.0 / np.sqrt(t._arr.astype(np.float32)), dtype=t.dtype, dev=t.device)

    def exp(t: Tensor) -> Tensor:
        return t.exp()

    def sigmoid(t: Tensor) -> Tensor:
        x = np.clip(t._arr.astype(np.float32), -80.0, 80.0)
        return Tensor(1.0 / (1.0 + np.exp(-x)), dtype=t.dtype, dev=t.device)

    def softmax(t: Tensor, dim: int = -1) -> Tensor:
        x = t._arr.astype(np.float32)
        mx = np.max(x, axis=dim, keepdims=True)
        mx = np.where(np.isinf(mx), 0.0, mx)
        ex = np.exp(x - mx)
        sm = np.sum(ex, axis=dim, keepdims=True)
        return Tensor(ex / np.maximum(sm, 1e-30), dtype=t.dtype, dev=t.device)

    def relu(t: Tensor) -> Tensor:
        return Tensor(np.maximum(t._arr, 0), dtype=t.dtype, dev=t.device)

    def nan_to_num(t: Tensor, nan: float = 0.0) -> Tensor:
        return Tensor(np.nan_to_num(t._arr, nan=nan), dtype=t.dtype, dev=t.device)

    def argsort(t: Tensor, dim: int = -1, descending: bool = False) -> Tensor:
        arr = -t._arr if descending else t._arr
        return Tensor(np.argsort(arr, axis=dim, kind="stable"), dtype=int64, dev=t.device)

    def sort(t: Tensor, dim: int = -1, descending: bool = False) -> Tuple[Tensor, Tensor]:
        idx = argsort(t, dim=dim, descending=descending)
        vals = gather(t, dim=dim, index=idx)
        return vals, idx

    def topk(t: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> _TopKResult:
        vals, idx = sort(t, dim=dim, descending=largest)
        slc = [slice(None)] * t.ndim
        slc[dim] = slice(0, k)
        return _TopKResult(vals[tuple(slc)], idx[tuple(slc)])

    def argmax(t: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return t.argmax(dim=dim, keepdim=keepdim)

    def argmin(t: Tensor, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
        return t.argmin(dim=dim, keepdim=keepdim)

    def gather(inp: Tensor, dim: int, index: Tensor) -> Tensor:
        res = np.take_along_axis(inp._arr, index._arr.astype(np.int64), axis=dim)
        return Tensor(res, dtype=inp.dtype, dev=inp.device)

    def index_select(inp: Tensor, dim: int, index: Tensor, out: Optional[Tensor] = None) -> Tensor:
        res = np.take(inp._arr, index._arr.astype(np.int64), axis=dim)
        if out is not None:
            out._arr[...] = res
            return out
        return Tensor(res, dtype=inp.dtype, dev=inp.device)

    def where(condition: Tensor, x: Optional[Any] = None, y: Optional[Any] = None) -> Any:
        if x is None and y is None:
            pts = np.nonzero(condition._arr)
            return tuple(Tensor(p, dtype=int64, dev=condition.device) for p in pts)
        xv = x._arr if isinstance(x, Tensor) else x
        yv = y._arr if isinstance(y, Tensor) else y
        dt = x.dtype if isinstance(x, Tensor) else (y.dtype if isinstance(y, Tensor) else float32)
        return Tensor(np.where(condition._arr, xv, yv), dtype=dt, dev=condition.device)

    def unique(inp: Tensor, sorted: bool = True, return_inverse: bool = False) -> Any:
        if return_inverse:
            u, inv = np.unique(inp._arr, return_inverse=True)
            return Tensor(u, dtype=inp.dtype, dev=inp.device), Tensor(inv.reshape(inp.shape), dtype=int64, dev=inp.device)
        u = np.unique(inp._arr)
        return Tensor(u, dtype=inp.dtype, dev=inp.device)

    def bincount(inp: Tensor, minlength: int = 0) -> Tensor:
        return Tensor(np.bincount(inp._arr.reshape(-1).astype(np.int64), minlength=minlength), dtype=int64, dev=inp.device)

    def cumsum(inp: Tensor, dim: int = 0, out: Optional[Tensor] = None) -> Tensor:
        res = np.cumsum(inp._arr, axis=dim)
        if out is not None:
            out._arr[...] = res
            return out
        return Tensor(res, dtype=inp.dtype, dev=inp.device)

    def equal(a: Tensor, b: Tensor) -> bool:
        return bool(np.array_equal(a._arr, b._arr))

    class finfo:
        def __init__(self, dt: _DType):
            self.min = -3.4e38 if dt in (float32, bfloat16) else -65504.0

    class linalg:
        @staticmethod
        def eigh(A: Tensor) -> Tuple[Tensor, Tensor]:
            w, v = np.linalg.eigh(A._arr.astype(np.float64))
            return Tensor(w.astype(np.float32), dtype=float32, dev=A.device), Tensor(v.astype(np.float32), dtype=float32, dev=A.device)

        @staticmethod
        def solve(A: Tensor, B: Tensor) -> Tensor:
            res = np.linalg.solve(A._arr.astype(np.float64), B._arr.astype(np.float64))
            return Tensor(res.astype(np.float32), dtype=float32, dev=A.device)

        @staticmethod
        def solve_triangular(A: Tensor, B: Tensor, upper: bool = False, unitriangular: bool = False) -> Tensor:
            a_np = A._arr.astype(np.float64)
            b_np = B._arr.astype(np.float64)
            orig_b_shape = b_np.shape
            L = a_np.shape[-1]
            a_flat = a_np.reshape(-1, L, L)
            b_flat = b_np.reshape(-1, L, orig_b_shape[-1])
            out = np.empty_like(b_flat)
            for i in range(a_flat.shape[0]):
                out[i] = scipy.linalg.solve_triangular(
                    a_flat[i], b_flat[i], lower=not upper, unit_diagonal=unitriangular
                )
            return Tensor(out.reshape(orig_b_shape).astype(np.float32), dtype=float32, dev=A.device)

        @staticmethod
        def norm(A: Tensor, dim: Optional[Any] = None, keepdim: bool = False) -> Tensor:
            return A.norm(dim=dim, keepdim=keepdim)

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def synchronize(dev: Any = None) -> None:
            pass

        @staticmethod
        def empty_cache() -> None:
            pass

        @staticmethod
        def memory_allocated(dev: Any = None) -> int:
            return 0

        @staticmethod
        def max_memory_allocated(dev: Any = None) -> int:
            return 0

    class _FFuncs:
        @staticmethod
        def silu(x: Tensor) -> Tensor:
            xf = x._arr.astype(np.float32)
            sig = 1.0 / (1.0 + np.exp(-np.clip(xf, -80.0, 80.0)))
            return Tensor(xf * sig, dtype=x.dtype, dev=x.device)

        @staticmethod
        def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
            res = np.matmul(x._arr.astype(np.float32), weight._arr.astype(np.float32).T)
            if bias is not None:
                res = res + bias._arr.astype(np.float32)
            return Tensor(res, dtype=x.dtype, dev=x.device)

        @staticmethod
        def layer_norm(
            x: Tensor,
            normalized_shape: Tuple[int, ...],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            eps: float = 1e-5,
        ) -> Tensor:
            xf = x._arr.astype(np.float32)
            mean = xf.mean(axis=-1, keepdims=True)
            var = ((xf - mean) ** 2).mean(axis=-1, keepdims=True)
            out = (xf - mean) / np.sqrt(var + eps)
            if weight is not None:
                out = out * weight._arr.astype(np.float32)
            if bias is not None:
                out = out + bias._arr.astype(np.float32)
            return Tensor(out, dtype=x.dtype, dev=x.device)

        @staticmethod
        def normalize(x: Tensor, p: float = 2.0, dim: int = -1, eps: float = 1e-12) -> Tensor:
            xf = x._arr.astype(np.float32)
            nrm = np.linalg.norm(xf, ord=p, axis=dim, keepdims=True)
            return Tensor(xf / np.maximum(nrm, eps), dtype=x.dtype, dev=x.device)

        @staticmethod
        def conv1d(inp: Tensor, weight: Tensor, groups: int = 1) -> Tensor:
            # Depthwise causal conv1d [B, D, S_pad] with weight [D, 1, K] and groups=D
            x_np = inp._arr.astype(np.float32)
            w_np = weight._arr.astype(np.float32)
            B, D, S_pad = x_np.shape
            K = w_np.shape[2]
            S = S_pad - K + 1
            out = np.zeros((B, D, S), dtype=np.float32)
            for k in range(K):
                out += x_np[:, :, k : k + S] * w_np[:, 0, k : k + 1]
            return Tensor(out, dtype=inp.dtype, dev=inp.device)

        @staticmethod
        def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
            lg = logits._arr.astype(np.float64)
            mx = np.max(lg, axis=-1, keepdims=True)
            log_probs = (lg - mx) - np.log(np.sum(np.exp(lg - mx), axis=-1, keepdims=True))
            idx = targets._arr.astype(np.int64)
            nll = -log_probs[np.arange(lg.shape[0]), idx]
            return Tensor(np.mean(nll).astype(np.float32), dtype=float32, dev=logits.device)

    class _NNModule:
        def __init__(self):
            pass

        def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
            setattr(self, name, tensor)

    class _NNNamespace:
        Module = _NNModule
        functional = _FFuncs

    class _TorchShim:
        Tensor = Tensor
        Size = tuple
        dtype = _DType
        float32 = float32
        float16 = float16
        bfloat16 = bfloat16
        float8_e4m3fn = float8_e4m3fn
        int64 = int64
        long = long
        int32 = int32
        int16 = int16
        int8 = int8
        uint16 = uint16
        uint8 = uint8
        bool = bool_dt
        device = device
        Generator = Generator
        no_grad = staticmethod(no_grad)
        inference_mode = staticmethod(inference_mode)
        tensor = staticmethod(tensor)
        from_numpy = staticmethod(from_numpy)
        zeros = staticmethod(zeros)
        ones = staticmethod(ones)
        empty = staticmethod(empty)
        full = staticmethod(full)
        zeros_like = staticmethod(zeros_like)
        ones_like = staticmethod(ones_like)
        empty_like = staticmethod(empty_like)
        full_like = staticmethod(full_like)
        eye = staticmethod(eye)
        arange = staticmethod(arange)
        randn = staticmethod(randn)
        rand = staticmethod(rand)
        randint = staticmethod(randint)
        randperm = staticmethod(randperm)
        cat = staticmethod(cat)
        stack = staticmethod(stack)
        matmul = staticmethod(matmul)
        mm = staticmethod(mm)
        mv = staticmethod(mv)
        bmm = staticmethod(bmm)
        einsum = staticmethod(einsum)
        sum = staticmethod(sum)
        max = staticmethod(max)
        abs = staticmethod(abs)
        sqrt = staticmethod(sqrt)
        rsqrt = staticmethod(rsqrt)
        exp = staticmethod(exp)
        sigmoid = staticmethod(sigmoid)
        softmax = staticmethod(softmax)
        relu = staticmethod(relu)
        nan_to_num = staticmethod(nan_to_num)
        argsort = staticmethod(argsort)
        sort = staticmethod(sort)
        topk = staticmethod(topk)
        argmax = staticmethod(argmax)
        argmin = staticmethod(argmin)
        gather = staticmethod(gather)
        index_select = staticmethod(index_select)
        where = staticmethod(where)
        unique = staticmethod(unique)
        bincount = staticmethod(bincount)
        cumsum = staticmethod(cumsum)
        equal = staticmethod(equal)
        finfo = finfo
        linalg = linalg
        cuda = cuda
        nn = _NNNamespace
        __version__ = "2.11.0-numpy-compat-shim"

    torch = _TorchShim()
    nn = _NNNamespace
    F = _FFuncs
