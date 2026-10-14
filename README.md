# Pentanary CPU inference runtime

A working **experimental** base-5 weight pipeline, bounded-buffer C++ inference
backend, Llama/Qwen2 reference decoder, and OpenAI-style chat API.

**Status:** compiled and tested on Linux/x86-64 with GCC 12 and Python 3.11.
Synthetic tests exercise the complete pipeline. **No pretrained 3B checkpoint
was supplied or downloaded; 3B output quality and full-model RSS are unverified.**
The bundled/generated demo has tiny random weights and produces meaningless text.

**QAT extension:** [QAT.md](QAT.md) documents the optional cloud training path:
`qat.py` supplies compatible STE parametrizations and exact export;
`train_qat.py` fine-tunes local HF weights and compares held-out loss before/after.
QAT can help recover quality, but is not guaranteed to recover all capabilities.
It does not change the ~1 GB packed size of 3 billion weights in this format.

**450 MB deployment:** [MEMORY.md](MEMORY.md) gives the corrected budget and the
new `run_450.py` cgroup launcher. A nominal 1.5B packed model needs ~500 MB on disk;
zero skipping does not reduce that storage. The strict profile caps kernel-accounted
memory and refuses an unverified soft fallback, but can still fail by OOM.

## Build and verify

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make                         # g++ -O3 -march=native, shared library + native tests
make test
make sanitize                # AddressSanitizer + UndefinedBehaviorSanitizer
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make benchmark
```

`requirements-lock.txt` records the exact Python environment used for verification;
install it instead of `requirements.txt` to reproduce those dependency versions.
PyTorch is optional and intentionally not a default runtime dependency. Install
`torch` separately to convert `.pt` / PyTorch `.bin` checkpoints. Only tensor
state dictionaries are accepted, using `weights_only=True, mmap=True`; legacy
non-mmap-compatible serialization should first be re-exported as Safetensors.

`-march=native` produces machine-specific code. Rebuild on the deployment host;
use `make CXXFLAGS='-O3 -std=c++17 -Wall -Wextra -Wpedantic'` for a portable build.

## Real checkpoint workflow

Use a **local** original HF checkpoint directory containing `config.json`,
`tokenizer.json`, and its weight shards. Obtain model access/licenses separately.
There is no automatic download or remote-code execution.

```bash
# Supports a single file, shard directory, or HF weight_map index.
.venv/bin/python quantize.py checkpoints/Qwen2.5-3B-Instruct \
  -o artifacts/model_pentanary.bin

# Requires a user systemd manager with cgroup-v2 memory delegation.
# A 3B model remains an unverified deployment candidate, not a fit guarantee.
.venv/bin/python run_450.py -- \
  --model artifacts/model_pentanary.bin \
  --config checkpoints/Qwen2.5-3B-Instruct/config.json \
  --tokenizer checkpoints/Qwen2.5-3B-Instruct/tokenizer.json \
  --max-context 512 --port 8000
```

Supported architecture/configuration subset:

- HF `model_type: llama` or `qwen2`, dense decoder-only models; **not MoE**.
- RMSNorm, SiLU gated MLP, causal grouped-query attention, FP16 KV cache.
- Default RoPE and Llama-3 `llama3` frequency scaling (Llama-3.2 configs).
- Projection biases (including Qwen Q/K/V biases), tied or separate output head.
- Canonical Llama-3-Instruct and Qwen2/2.5-Instruct text chat formatting using
  the original `tokenizer.json`. Arbitrary Jinja templates, older Llama chat
  templates, multimodal input, sliding-window attention, YaRN/dynamic RoPE,
  adapters, and speculative decoding are **not** implemented.
- Unexpected tensor names/shapes and unsupported architecture options fail early
  rather than silently ignoring new model components.

A family-compatible config is not a validation of a specific released checkpoint.
Run a full-checkpoint smoke test and perplexity/task evaluation before relying on
outputs. Cache/context is reset for each request; all supplied messages are
prefilled token-by-token. There is no persistent conversation store.

## Synthetic end-to-end demo

```bash
.venv/bin/python synthetic.py --output artifacts/demo
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python server.py \
  --model artifacts/demo/model_pentanary.bin \
  --config artifacts/demo/config.json \
  --tokenizer artifacts/demo/tokenizer.json \
  --max-context 64
```

The server binds `0.0.0.0:8000`; `/` redirects to interactive `/docs`.
`/health` exposes current RSS and cache size; `/v1/models` lists `pentanary`.
No frontend calls to sandbox-local hosts are necessary: use `/docs` in the live
preview, or run this command **on the same host as the server**:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"pentanary","messages":[{"role":"user","content":"hello world"}],
       "max_tokens":16,"temperature":0,"stream":true,
       "stream_options":{"include_usage":true}}'
```

Remove `stream` for ordinary JSON. Streaming emits assistant-role, text-delta,
finish-reason, optional usage chunks, then `data: [DONE]`. Supported options are
`max_tokens`, `temperature`, `top_p`, `seed`, up to four `stop` strings, and `n=1`.
Stop strings are buffered across tokens so they do not leak into streamed text.
Tools, logprobs, image input, multiple choices, and other unknown fields are
rejected. This is a **chat-completions subset**, not the entire OpenAI API.

A single model instance permits one active generation. Busy requests get HTTP
429 instead of accumulating KV caches. A disconnected stream can remain busy
until its current native step completes; retry after it releases the model.
Requests have a 1 MiB body limit and a fixed context limit. Errors after SSE
headers are sent appear as error events followed by `[DONE]`.

**Security:** intended for a trusted local environment. No authentication, TLS,
per-user quotas, or production abuse protection is provided. Put an authenticated
reverse proxy in front before exposing the service outside the workspace.

## Quantization and execution

### Scale-aware pentanary quantization (`quantize.py`)

For each matrix, retain the two largest absolute weights in every group of four
along the last dimension. Ties use a stable ordering. Pruned weights become zero;
nonzero retained weights choose magnitude 1 or 2 at threshold `1.5 * alpha`.
True zero inputs stay zero, so sparsity is **at least 50%** (exactly 50% for the
continuous random fixtures). Matrix widths must be divisible by four in this mode.
Vectors such as normalization weights and biases stay FP16 to avoid needless
additional damage.

A positive scale is fitted per matrix by alternating state assignment and the
least-squares update:

```
alpha = sum(w * q) / sum(q * q)
```

The initial scale is mean retained magnitude / 1.5. Three refinement iterations
are used by default; the final scale is rounded to FP16 and states are reassigned.
All-zero matrices use a positive minimum scale and all-zero states. `--dense`
disables structural pruning and uses nearest-state thresholds `0.5 * alpha` and
`1.5 * alpha`. NaNs, infinities, invalid shapes, and FP16 scale overflow are rejected.

Safetensors F16, BF16, and F32 data are read in bounded row-aligned chunks, with
no full-layer materialization. Chunk packing carries the final one or two states
across chunk boundaries. Output is atomically replaced on success. Payload
spooling needs temporary disk space of roughly **twice the final packed file**
in addition to the input checkpoint. Offline quantization is not included in the
inference RSS target. `<output>.json` records per-tensor scale, zero fraction, and
relative reconstruction RMSE.

### Packing and loading (`pentanary_loader.cpp`)

For `w` in `{-2,-1,0,1,2}`, define digit `d = w + 2`. Pack:

```
b = d0 + 5*d1 + 25*d2         # 0 <= b < 125
w0 = b % 5 - 2
w1 = (b / 5) % 5 - 2
w2 = b / 25 - 2               # integer division
```

The C++ `unpack_pentanary_byte(uint8_t)` returns three `int8_t` weights. Bytes
125–255 are invalid. The final tensor triplet is padded with **zero weights**
(digit 2). Packing is tensor-contiguous, not row-padded. Tests include rows that
begin or end mid-byte. Shape products, offsets, sizes, duplicate names, scales,
and truncated payloads are checked. Payload bytes are checked when consumed,
not scanned at startup.

The loader uses `pread` with a **64 KiB buffer**, not a full-file mapping or an
in-memory dequantized layer. Embedding lookup reads only one packed row.

### Integer weight kernel (`kernel.cpp`)

Activations are dynamically rounded to the symmetric int16 range
`[-32767,32767]` and held in int32 storage. Weight accumulation is int64:

- `0`: skip activation load and accumulator arithmetic;
- `+1 / -1`: addition / subtraction;
- `+2 / -2`: shift an **unsigned magnitude** left by one, restore sign, add/subtract.

Shifting a negative signed C++ integer is avoided. Tensor limits keep the
accumulator safely within int64. Only the final activation/weight scale
application uses floating multiplication in the weight GEMV. Activation rounding
introduces another small approximation in addition to weight quantization.

**Zero is not literally zero-cost:** byte decoding, iteration, and branching
still occur. This is zero-arithmetic skipping, not a sparse indexed format or
hardware sparse-matmul implementation. Attention dot products, RMSNorm, RoPE,
SiLU, softmax, and sampling remain floating point; the full transformer is not
multiply-free. This scalar CPU kernel prioritizes clarity and correctness over
SIMD throughput.

## Memory budgets: legacy soft admission and strict 450 MB profile

The optional `run_450.py` launcher sets a page-aligned cgroup limit at or below
450,000,000 bytes, disables swap and selects a 420 MB admission budget. The server
verifies the real kernel limits before loading the model. See [MEMORY.md](MEMORY.md)
for requirements, tested enforcement, and the distinction between cgroup memory,
RSS, and successful inference. No literal every-instant RSS guarantee is made.

The original soft-budget behavior remains available when running `server.py`
directly without `--strict-450mb`, as described below.

Three billion weights at **three weights per byte require ~1,000,000,000 bytes
on disk**, before headers. Exactly 50% zero weights do not shrink this fixed
packing format further. An all-resident 3B checkpoint cannot fit in 500 MB here.

The implementation instead targets **process resident RAM**, using streamed
weights, FP16 KV cache, single-sequence generation, startup memory admission,
and current-RSS checks between tokens. Default budget: **480,000,000 bytes**;
startup rejects estimates above the budget. Cache bytes are:

```
2 * num_layers * max_context * num_kv_heads * head_dim * 2
```

For illustration, 28 layers, 8 KV heads, head dimension 128, and context 512 need
58,720,256 KV bytes. Admission also reserves scratch memory and reads current
RSS. These estimates and periodic checks are **not an OS-enforced hard cap**.
Python/native allocator behavior and tokenizer sizes vary. Filesystem page cache
is not in process RSS and can count against container/cgroup limits. For a hard
limit use an external Linux cgroup/container memory limit, monitor OOM behavior,
and validate with the actual model at its maximum context. Small synthetic tests
cannot prove a 3B deployment will stay under budget.

Streaming most model weights for every token is expensive. If the OS cannot
cache the packed file, disk bandwidth becomes a major bottleneck. 2:4 pruning
plus per-matrix five-level quantization is also extremely lossy. `quantize.py`
is PTQ only; the optional [QAT workflow](QAT.md) adds full-parameter STE fine-tuning,
not a guarantee of recovery. Calibration, distillation, and a pretrained 3B QAT
run are not included. Useful 3B language quality is **not claimed**.

## Verified results

See [VERIFICATION.md](VERIFICATION.md) for recorded measurements and caveats.
The independent decoder tests compare several causal positions for tiny Llama
and Qwen models against a separately expressed dense NumPy oracle. These tests
verify mechanics, not parity with an original unquantized HF 3B model.

## Files

| File | Purpose |
|---|---|
| `quantize.py` | Local checkpoint conversion and scale fitting |
| `qat.py`, `train_qat.py` | Optional cloud QAT, evaluation, and exact export |
| `requirements-qat.txt`, `QAT.md` | Separate training environment and workflow |
| `penta_format.py` | Binary directory and reference base-5 codec |
| `pentanary.h`, `pentanary_loader.cpp` | Validated loader, packing helpers, C ABI |
| `kernel.cpp` | Streamed integer-shift weight GEMV |
| `native.py` | ctypes binding |
| `runtime.py` | Llama/Qwen2 autoregressive decoder and KV cache |
| `server.py` | HTTP chat API, SSE, strict-profile guard and memory health |
| `run_450.py`, `memory_limits.py` | Fail-closed cgroup-v2 450 MB launch and inspection |
| `memory_plan.py`, `MEMORY.md` | File/resident/streaming budget report and deployment guide |
| `synthetic.py`, `benchmark.py` | Reproducible fixtures and benchmarks |
| `tests/` | Native, numerical, decoder, format, and API tests |
| `FORMAT.md` | Byte-level PENTA5 v1 specification |

Generated weights, executables, shared libraries, environments, and benchmark
artifacts are ignored by Git. `make` and the demo commands regenerate them.
