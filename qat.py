"""Optional PyTorch QAT with a PENTA5-v1-compatible export path.

FP32 master parameters are optimized. Forward uses fixed, FP16-representable
scales and dynamic 2:4 masks; backward uses an identity STE, including pruned
entries (allowing regrowth). No gradient is assigned to alpha. This is NOT a
promise of quality recovery, and training is not constrained to 500 MB.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from penta_format import TensorInfo, PACKED, FP16, pack_pentanary, write_directory
from quantize import fit_alpha, states


@torch.no_grad()
def pentanary_states(weight, alpha, sparse=True):
    """Match quantize.states, including stable tie-breaking and half-away rounding."""
    magnitude = weight.abs()
    if sparse:
        if weight.ndim != 2 or weight.shape[-1] % 4:
            raise ValueError('2:4 QAT requires matrix widths divisible by four')
        grouped = magnitude.reshape(-1, 4)
        indices = torch.argsort(grouped, dim=-1, stable=True)[:, 2:]
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(1, indices, True)
        q = torch.where(magnitude >= 1.5 * alpha, 2, 1) * weight.sign()
        return torch.where(mask.reshape_as(weight), q, 0)
    return torch.clamp(torch.floor(magnitude / alpha + 0.5), max=2) * weight.sign()


class PentanaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, sparse):
        return pentanary_states(weight, alpha, sparse) * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def apply_pentanary_ste(weight, alpha, sparse=True):
    return PentanaryQuantizer.apply(weight, alpha, sparse)


class ActivationQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # The native kernel uses one scale per token and double intermediates.
        # Use the same rule, including ties away from zero (not torch.round).
        d = x.double()
        peak = d.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(peak == 0, torch.ones_like(peak), peak / 32767)
        fixed = d.sign() * torch.floor(d.abs() / scale + 0.5)
        return (fixed * scale).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class HalfSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.half().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class PentanaryWeight(nn.Module):
    def __init__(self, alpha, sparse=True, chunk_elements=262144):
        super().__init__()
        if not math.isfinite(alpha) or not 2**-24 <= alpha <= 65504:
            raise ValueError('alpha must be finite, positive, and representable in FP16')
        self.register_buffer('alpha', torch.tensor(float(np.float16(alpha)), dtype=torch.float32))
        self.sparse, self.chunk_elements = sparse, chunk_elements

    def forward(self, weight):
        # Bound temporary sorting/mask tensors. Autograd concatenates outputs;
        # the full fake-quantized matrix still exists during training.
        rows = max(1, self.chunk_elements // weight.shape[-1])
        return torch.cat([apply_pentanary_ste(part, self.alpha, self.sparse)
                          for part in weight.split(rows, dim=0)], dim=0)


class HalfWeight(nn.Module):
    def forward(self, weight):
        return HalfSTE.apply(weight)


@dataclass
class Binding:
    module: nn.Module
    parameter: str
    quantizer: nn.Module

    @property
    def master(self):
        return getattr(self.module.parametrizations, self.parameter).original


@dataclass
class QATState:
    bindings: dict
    hooks: list
    sparse: bool
    activation_quantization: bool


def _activation_hook(module, args, kwargs):
    if args:
        return (ActivationQuantizer.apply(args[0]), *args[1:]), kwargs
    if 'input' not in kwargs:
        raise ValueError('QAT linear expects positional input or input=')
    return args, {**kwargs, 'input': ActivationQuantizer.apply(kwargs['input'])}


def install_qat(model, sparse=True, activation_quantization=True,
                chunk_elements=262144, iterations=3):
    """Parametrize every Linear/Embedding weight, and FP16-round all vectors.

    Tied parameters retain one shared FP32 master and one scale. Optimizers must
    be constructed AFTER this call. Do not call model.half()/bfloat16() afterwards.
    """
    if chunk_elements < 4 or not 0 <= iterations <= 20:
        raise ValueError('invalid chunk size or scale-fitting iteration count')
    modules = list(model.named_modules())
    # Preflight without mutating the model. Avoid silently ignoring custom weights.
    for prefix, module in modules:
        if parametrize.is_parametrized(module):
            raise ValueError('model is already parametrized')
        for name, p in module.named_parameters(recurse=False):
            if p.dtype != torch.float32 or not p.is_contiguous():
                raise ValueError('QAT requires contiguous FP32 master parameters')
            if p.ndim != 1 and not (p.ndim == 2 and name == 'weight' and isinstance(module, (nn.Linear, nn.Embedding))):
                raise ValueError(f'unsupported parameter: {prefix}.{name}')
            if p.numel() == 0 or (p.ndim == 2 and sparse and p.shape[-1] % 4):
                raise ValueError(f'invalid shape for 2:4 QAT: {prefix}.{name}')
    bindings, shared, hooks = {}, {}, []
    for prefix, module in modules:
        parameters = list(module.named_parameters(recurse=False))
        for name, p in parameters:
            key = f'{prefix}.{name}' if prefix else name
            if id(p) not in shared:
                if p.ndim == 2:
                    def chunks():
                        rows = max(1, chunk_elements // p.shape[-1])
                        for part in p.detach().split(rows, dim=0):
                            x = part.cpu().numpy().ravel()
                            if not np.all(np.isfinite(x)):
                                raise ValueError(f'{key}: nonfinite weight')
                            yield x
                    alpha = fit_alpha(chunks, sparse, iterations)
                    quantizer = PentanaryWeight(alpha, sparse, chunk_elements).to(p.device)
                else:
                    if not torch.isfinite(p).all() or (p.abs() > 65504).any():
                        raise ValueError(f'{key}: cannot represent vector in FP16')
                    quantizer = HalfWeight()
                shared[id(p)] = quantizer
            quantizer = shared[id(p)]
            parametrize.register_parametrization(module, name, quantizer)
            bindings[key] = Binding(module, name, quantizer)
        if isinstance(module, nn.Linear) and activation_quantization:
            hooks.append(module.register_forward_pre_hook(_activation_hook, with_kwargs=True))
    if not bindings:
        raise ValueError('model has no trainable parameters')
    return QATState(bindings, hooks, sparse, activation_quantization)


@torch.no_grad()
def export_qat(state, output, chunk_elements=262144, omit_tied_output=False):
    """Export trained masters using EXISTING alpha/masks, never refit via PTQ.

    Reads at most a row-aligned chunk of each master onto CPU. Dynamic masks
    are reconstructed exactly, using the same stable tie-breaking as forward.
    Vectors use the same FP16 rounding as the QAT parametrizations.
    """
    if chunk_elements < 4:
        raise ValueError('chunk_elements must be >=4')
    if omit_tied_output:
        embed = state.bindings.get('model.embed_tokens.weight')
        head = state.bindings.get('lm_head.weight')
        if embed is None or head is None or embed.master is not head.master or embed.quantizer is not head.quantizer:
            raise ValueError('cannot omit output head: embedding/master/scale are not tied')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors, summary = [], []
    temporary = None
    try:
        with tempfile.TemporaryFile(dir=output.parent) as payload:
            for name, binding in state.bindings.items():
                if omit_tied_output and name == 'lm_head.weight':
                    continue
                master, quantizer = binding.master, binding.quantizer
                if master.dtype != torch.float32 or not master.is_contiguous():
                    raise ValueError('FP32 masters required; do not cast model after installing QAT')
                kind = PACKED if isinstance(quantizer, PentanaryWeight) else FP16
                alpha = float(quantizer.alpha.item()) if kind == PACKED else 1.0
                if not math.isfinite(alpha) or alpha <= 0 or alpha != float(np.float16(alpha)):
                    raise ValueError('scale is not an unchanged, valid FP16 scale')
                if not name or len(name.encode()) > 4096 or '\0' in name:
                    raise ValueError('invalid tensor name')
                tensors.append(TensorInfo(name, tuple(master.shape), kind, alpha,
                                          int(kind == PACKED and quantizer.sparse)))
                width = master.shape[-1] if kind == PACKED else 1
                step = max(width, chunk_elements // width * width)
                flat = master.detach().view(-1)
                carry = np.empty(0, np.int8)
                zeros = 0
                for offset in range(0, flat.numel(), step):
                    x = flat[offset:offset + step].cpu().numpy()
                    if not np.isfinite(x).all():
                        raise ValueError(f'{name}: nonfinite trained parameter')
                    if kind == FP16:
                        if (np.abs(x) > 65504).any():
                            raise ValueError(f'{name}: FP16 vector overflow')
                        payload.write(x.astype('<f2').tobytes())
                    else:
                        q = states(x, alpha, quantizer.sparse)
                        zeros += int(np.count_nonzero(q == 0))
                        joined = np.concatenate((carry, q))
                        limit = joined.size // 3 * 3
                        payload.write(pack_pentanary(joined[:limit]).tobytes())
                        carry = joined[limit:].copy()
                if carry.size:
                    payload.write(pack_pentanary(carry).tobytes())
                summary.append({'name': name, 'alpha': alpha,
                                'zero_fraction': zeros / master.numel() if kind == PACKED else None})
            if not tensors:
                raise ValueError('no QAT weights to export')
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as f:
                temporary = f.name
                write_directory(f, tensors)
                payload.seek(0)
                while block := payload.read(1 << 20):
                    f.write(block)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, output)
            temporary = None
    finally:
        if temporary:
            os.unlink(temporary)
    report = {'format': 'PENTA5-v1', 'method': 'QAT-STE', 'fixed_fp16_scales': True,
              'sparse_2_4': state.sparse, 'activation_fake_quantization': state.activation_quantization,
              'omitted_tied_output': omit_tied_output,
              'bytes': output.stat().st_size, 'tensors': summary}
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
