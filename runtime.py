"""Single-sequence reference Llama / Qwen2 decoder with streamed native weights.

Weight GEMVs use the shift kernel. RMSNorm, RoPE, attention, SiLU and sampling
remain floating point. This is a correctness-oriented CPU runtime, not a fast
production serving engine. Post-training 2:4 pruning is NOT quality-preserving.
"""
import json
from pathlib import Path
import threading
import numpy as np
from tokenizers import Tokenizer
from native import NativeModel
from penta_format import PACKED, FP16
from memory_limits import kv_cache_bytes, scratch_reserve_bytes


def rss_bytes():
    # Linux current resident set; does not count global filesystem page cache.
    import os
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


class Runtime:
    def __init__(self, model_path, config_path, tokenizer_path, max_context=512,
                 memory_budget=480_000_000):
        self.lock = threading.Lock()
        self.config = c = json.loads(Path(config_path).read_text())
        self.family = c.get('model_type')
        if self.family not in ('llama', 'qwen2'):
            raise ValueError('only model_type llama and qwen2 are implemented')
        if c.get('hidden_act', 'silu') != 'silu' or c.get('use_sliding_window', False):
            raise ValueError('only SiLU and full causal attention are supported')
        self.hidden = int(c['hidden_size'])
        self.layers = int(c['num_hidden_layers'])
        self.heads = int(c['num_attention_heads'])
        self.kv_heads = int(c.get('num_key_value_heads', self.heads))
        self.head_dim = int(c.get('head_dim', self.hidden // self.heads))
        self.intermediate = int(c['intermediate_size'])
        self.vocab = int(c['vocab_size'])
        if (min(self.hidden, self.layers, self.heads, self.kv_heads, self.head_dim, self.intermediate, self.vocab) <= 0
                or self.heads % self.kv_heads or self.head_dim % 2
                or self.hidden > 65536 or self.layers > 256 or self.heads > 512 or self.head_dim > 1024
                or self.intermediate > 262144 or self.vocab > 1000000):
            raise ValueError('unsupported transformer dimensions')
        if not 1 <= max_context <= int(c.get('max_position_embeddings', 2048)):
            raise ValueError('max_context outside checkpoint positional range')
        if not 0 < memory_budget < 500_000_000:
            raise ValueError('memory budget must be positive and below 500,000,000 bytes')
        self.max_context, self.memory_budget = max_context, memory_budget
        self.eps = float(c.get('rms_norm_eps', 1e-6))
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError('invalid RMS norm epsilon')
        self.model = NativeModel(model_path)
        self._validate_weights()
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        if self.tokenizer.get_vocab_size(with_added_tokens=True) > self.vocab:
            raise ValueError('tokenizer vocabulary exceeds embedding vocabulary')
        markers = (['<|im_start|>', '<|im_end|>'] if self.family == 'qwen2' else
                   ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'])
        if any(self.tokenizer.token_to_id(marker) is None for marker in markers):
            raise ValueError('tokenizer lacks the supported Qwen2/Llama3 chat markers')
        eos = c.get('eos_token_id', [])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.eos.discard(None)
        self.inv_freq = self._rope_frequencies(c)
        # K and V each: layers * context * KV heads * head_dim * sizeof(fp16).
        self.kv_bytes = kv_cache_bytes(c, max_context)
        # Reserve for allocator overhead, per-head attention copies, tokenizer,
        # vocabulary logits, HTTP requests. This is an admission estimate, NOT
        # an OS-enforced memory limit; use a cgroup for a hard limit.
        scratch = scratch_reserve_bytes(self.head_dim, self.vocab, max_context)
        self.admission_estimate = rss_bytes() + self.kv_bytes + scratch
        if self.admission_estimate >= memory_budget:
            raise MemoryError(f'estimated resident memory {self.admission_estimate:,} exceeds budget {memory_budget:,}; reduce context')
        shape = (self.layers, max_context, self.kv_heads, self.head_dim)
        self.keys = np.empty(shape, dtype=np.float16)
        self.values = np.empty(shape, dtype=np.float16)
        self.position = 0

    def _validate_weights(self):
        expected = {'model.embed_tokens.weight': (self.vocab, self.hidden),
                    'model.norm.weight': (self.hidden,)}
        for i in range(self.layers):
            p = f'model.layers.{i}.'
            expected[p + 'input_layernorm.weight'] = (self.hidden,)
            expected[p + 'post_attention_layernorm.weight'] = (self.hidden,)
            for key, rows, cols in [('self_attn.q_proj', self.heads * self.head_dim, self.hidden),
                                    ('self_attn.k_proj', self.kv_heads * self.head_dim, self.hidden),
                                    ('self_attn.v_proj', self.kv_heads * self.head_dim, self.hidden),
                                    ('self_attn.o_proj', self.hidden, self.heads * self.head_dim),
                                    ('mlp.gate_proj', self.intermediate, self.hidden),
                                    ('mlp.up_proj', self.intermediate, self.hidden),
                                    ('mlp.down_proj', self.hidden, self.intermediate)]:
                expected[p + key + '.weight'] = (rows, cols)
                if p + key + '.bias' in self.model.tensors:
                    expected[p + key + '.bias'] = (rows,)
        self.output_weight = 'lm_head.weight'
        if self.output_weight not in self.model.tensors and self.config.get('tie_word_embeddings', False):
            self.output_weight = 'model.embed_tokens.weight'
        expected[self.output_weight] = (self.vocab, self.hidden)
        for name, shape in expected.items():
            t = self.model.tensors.get(name)
            if t is None or t.shape != shape or t.kind != (PACKED if len(shape) == 2 else FP16):
                raise ValueError(f'missing or incompatible tensor: {name}; expected {shape}')
        unsupported = set(self.model.tensors) - set(expected)
        # Some older exports contain precomputed rotary buffers; unused safely.
        unsupported = {n for n in unsupported if not n.endswith('rotary_emb.inv_freq')}
        if unsupported:
            raise ValueError(f'unsupported checkpoint tensors: {sorted(unsupported)[:5]}')

    def _rope_frequencies(self, c):
        theta = float(c.get('rope_theta', 10000.0))
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError('invalid rope_theta')
        inv = 1.0 / theta ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim)
        scaling = c.get('rope_scaling') or {}
        kind = scaling.get('rope_type', scaling.get('type', 'default'))
        if kind == 'llama3':
            factor = float(scaling['factor'])
            low, high = float(scaling['low_freq_factor']), float(scaling['high_freq_factor'])
            old = float(scaling['original_max_position_embeddings'])
            if not (factor > 0 and high > low > 0 and old > 0):
                raise ValueError('invalid llama3 RoPE scaling')
            wavelength = 2 * np.pi / inv
            smooth = (old / wavelength - low) / (high - low)
            middle = (1 - smooth) * inv / factor + smooth * inv
            inv = np.where(wavelength > old / low, inv / factor,
                           np.where(wavelength < old / high, inv, middle))
        elif kind != 'default':
            raise ValueError(f'unsupported RoPE scaling: {kind}')
        return inv.astype(np.float32)

    def _rope(self, x, position):
        angle = self.inv_freq * position
        cos, sin = np.cos(angle), np.sin(angle)
        left, right = np.split(x, 2, axis=-1)
        return np.concatenate((left * cos - right * sin, right * cos + left * sin), axis=-1)

    def _norm(self, x, name):
        return x / np.sqrt(np.mean(x * x) + self.eps) * self.model.row(name)

    def _linear(self, prefix, x):
        result = self.model.matvec(prefix + '.weight', x)
        if prefix + '.bias' in self.model.tensors:
            result += self.model.row(prefix + '.bias')
        return result

    def reset(self):
        # Stale cache entries cannot be attended to because all slices end at position.
        self.position = 0

    def forward(self, token, logits=True):
        if self.position >= self.max_context or not 0 <= token < self.vocab:
            raise ValueError('token or position out of bounds')
        if rss_bytes() >= self.memory_budget:
            raise MemoryError('resident memory budget exceeded')
        pos = self.position
        x = self.model.row('model.embed_tokens.weight', token)
        for layer in range(self.layers):
            p = f'model.layers.{layer}.'
            n = self._norm(x, p + 'input_layernorm.weight')
            q = self._linear(p + 'self_attn.q_proj', n).reshape(self.heads, self.head_dim)
            k = self._linear(p + 'self_attn.k_proj', n).reshape(self.kv_heads, self.head_dim)
            v = self._linear(p + 'self_attn.v_proj', n).reshape(self.kv_heads, self.head_dim)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(v))
                    or np.max(np.abs(k)) > 65504 or np.max(np.abs(v)) > 65504):
                raise FloatingPointError('KV cache FP16 overflow')
            self.keys[layer, pos], self.values[layer, pos] = k, v
            attention = np.empty((self.heads, self.head_dim), dtype=np.float32)
            for head in range(self.heads):
                kv_head = head // (self.heads // self.kv_heads)
                keys = self.keys[layer, :pos + 1, kv_head].astype(np.float32)
                values = self.values[layer, :pos + 1, kv_head].astype(np.float32)
                scores = keys @ q[head] / np.sqrt(self.head_dim)
                attention[head] = softmax(scores) @ values
            x += self._linear(p + 'self_attn.o_proj', attention.ravel())
            n = self._norm(x, p + 'post_attention_layernorm.weight')
            gate = self._linear(p + 'mlp.gate_proj', n)
            up = self._linear(p + 'mlp.up_proj', n)
            silu = gate / (1.0 + np.exp(-np.clip(gate, -80, 80)))
            x += self._linear(p + 'mlp.down_proj', silu * up)
        self.position += 1
        if not logits:
            return None
        result = self.model.matvec(self.output_weight, self._norm(x, 'model.norm.weight'))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError('nonfinite logits')
        return result

    def encode_messages(self, messages):
        # Explicit model-family templates; not arbitrary Jinja chat templates.
        if self.family == 'qwen2':
            text = ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)
            text += '<|im_start|>assistant\n'
        else:
            text = '<|begin_of_text|>'
            text += ''.join(f'<|start_header_id|>{m["role"]}<|end_header_id|>\n\n{m["content"]}<|eot_id|>' for m in messages)
            text += '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or len(ids) >= self.max_context:
            raise ValueError('prompt must leave room for generation within max_context')
        return ids

    def generate(self, prompt, max_tokens=32, temperature=0.0, top_p=1.0, seed=None, stop=()):
        """Yield (new_text, completion_token_count, finish_reason).

        Caller holds lock for entire generator lifetime. Stop strings are held
        back across token boundaries and never sent in SSE content.
        """
        self.reset()
        rng = np.random.default_rng(seed)
        for i, token in enumerate(prompt):
            scores = self.forward(token, logits=i == len(prompt) - 1)
        generated, emitted = [], ''
        limit = min(max_tokens, self.max_context - len(prompt))
        for i in range(limit):
            if temperature == 0:
                token = int(np.argmax(scores))
            else:
                probabilities = softmax(scores.astype(np.float64) / max(temperature, 1e-6))
                if top_p < 1:
                    order = np.argsort(-probabilities)
                    ranked = probabilities[order]
                    ranked[np.cumsum(ranked) - ranked >= top_p] = 0
                    ranked /= ranked.sum()
                    token = int(rng.choice(order, p=ranked))
                else:
                    token = int(rng.choice(self.vocab, p=probabilities))
            generated.append(token)
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            ends = [text.find(s) for s in stop if s in text]
            reason = 'stop' if token in self.eos or ends else 'length' if i + 1 == limit else None
            if ends:
                text = text[:min(ends)]
            if reason is None:
                hold = 0
                for s in stop:
                    for n in range(1, min(len(s), len(text) + 1)):
                        if text.endswith(s[:n]):
                            hold = max(hold, n)
                if hold:
                    text = text[:-hold]
                text = text.rstrip('\ufffd')  # incomplete byte-fallback UTF-8
            if not text.startswith(emitted):
                raise ValueError('tokenizer decoder is not prefix-stable; unsupported streaming decoder')
            delta = text[len(emitted):]
            emitted = text
            yield delta, len(generated), reason
            if reason:
                break
            scores = self.forward(token)
