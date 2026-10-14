# Verification record

Verified in this workspace on **2026-10-14**, Linux x86-64, Python 3.11.2,
GCC 12.2.0. Dependency versions are in `requirements-lock.txt`.

## Commands and results

| Check | Result |
|---|---|
| `make` | Shared library and native test binary compiled with `-O3 -march=native` |
| `make test` | Native tests passed; **31 Python tests passed, 4 skipped** in the inference-only environment |
| `make test-qat` | **20 optional QAT tests passed** in the separate PyTorch environment |
| `PENTA_TEST_SYSTEMD=1 … pytest tests/test_memory_budget.py` | **17 passed**, including isolated real cgroup cap/OOM tests |
| `make sanitize` | Native loader, truncated-file handling, codec and kernel tests passed under AddressSanitizer + UndefinedBehaviorSanitizer |
| Python `compileall` | Passed |
| `git diff --check` | Passed |
| `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 make benchmark` | Numerical and RSS assertions passed |
| Live Uvicorn HTTP test, port 8000 | Health/docs, nonstream JSON, SSE, optional usage, `[DONE]`, and disconnect recovery passed |

The inference-only environment does not install torch, so it skips optional
PyTorch checkpoint input and the QAT test module. **Both PyTorch paths were tested
in the separate training environment** (PyTorch 2.6.0 / Transformers 4.48.3).
Two privileged live memory-limit tests also skip in the default suite; they were
run separately with explicit opt-in in the inference environment. Safetensors
FP16 and BF16 input and shard-index loading were exercised. The FP32 reader path
is implemented but not separately benchmarked against a full checkpoint. No
original pretrained 3B checkpoint was loaded; no pretrained quality or full-model
RSS result is claimed. See [QAT.md](QAT.md) for the new training/export checks.

## Test coverage

- All 125 valid three-weight states, all 131 invalid byte values.
- Signed activation extremes including `INT_MIN`, ±1 and ±2 arithmetic,
  zero skipping, and positive/subnormal FP16 conversion.
- Global tensor packing, row boundaries inside packed bytes, and final padding.
- 2:4 zero masks, scale-aware thresholds, deterministic chunk-size equivalence.
- F16/BF16 Safetensors, sharded weight maps, NaN rejection, invalid sparse widths.
- File truncation, wrong magic, malformed payload bytes, invalid matvec dimensions.
- Every synthetic checkpoint tensor's native rows and GEMVs versus dequantized
  NumPy reference, including all-zero and negative activations.
- Multiple causal positions against an independent dense transformer oracle for
  both Qwen2 and Llama, plus Llama-3 RoPE scaling and tied-output-head smoke tests.
- KV-cache reset, memory-admission rejection, context-budget validation.
- OpenAI-style JSON/SSE equivalence, usage, stop strings split across tokens,
  model errors, validation errors, body size limits, and busy-request rejection.

## Synthetic matrix benchmark

Deterministic Gaussian **2048 × 4096** matrix, 8,388,608 weights; default
2:4 pruning and per-matrix FP16 scale. Ten timed native GEMVs after warm-up,
with a separate worker process from checkpoint creation/reference calculation.
These measurements depend on CPU load, filesystem cache, compiler and hardware.

| Metric | Observed |
|---|---:|
| Original FP16 weight bytes | 16,777,216 |
| Packed file bytes, including header | 2,796,331 |
| Zero fraction | 0.500000 |
| Checkpoint conversion time | 1.359 s |
| Average native matvec time | 96.795 ms |
| Logical weights processed per second | 86.664 million |
| Worker peak RSS (`ru_maxrss`) | 119,119,872 bytes |
| Kernel relative L2 error vs dequantized floating-point matvec | 0.000031402 (0.00314%) |
| Quantized weight relative RMSE vs original FP16 weights | 0.436011 (43.60%) |

The small **kernel** error does not mean quantization preserves model quality.
The much larger **weight reconstruction** error measures pruning/quantization
loss on this random matrix. Neither metric is a perplexity measurement.

A two-layer, hidden-size-32, random-weight Qwen2 fixture generated 16 tokens
following a 7-token prompt in **0.02565 s** including prefill in the direct runtime.
Its allocated KV cache was **8,192 bytes** for context 64. Do not extrapolate this
tiny fixture's speed to a 3B model.

## Live HTTP smoke test

A real server was started on `0.0.0.0:8000` using that tiny random fixture.
JSON and SSE requests returned identical generated text and usage counts.
SSE included role/content/finish/usage chunks and a final `[DONE]` event.
An early-disconnected stream released the runtime; a subsequent request
succeeded. A brief 429 while an in-flight native step finishes is expected.

The final health check reported **70,914,048 bytes current RSS**. This is the
tiny synthetic API process, **not** a 3B model memory measurement or a peak-RSS
bound. `artifacts/benchmark/results.json` and `http-smoke.json` contain raw results.
Generated checkpoints and raw artifacts are intentionally excluded from Git.

## Strict 450 MB profile verification

See [MEMORY.md](MEMORY.md) for the corrected allocation math and deployment steps.
The new planner reads actual file metadata and applies no fictional zero-sparsity
storage discount. The strict API profile refuses to load a model without verified
cgroup-v2 memory and swap limits. Dry-run tests preserve venv interpreter paths
and pass filenames without shell expansion.

On systemd 252, two fresh isolated system scopes ran as the ordinary user under
`memory.max=449998848`, `memory.swap.max=0`, and `OOMPolicy=kill`. One exercised a
tiny model's full 64-token cache and JSON/SSE API paths with no OOM events. The
other attempted a 600,000,000-byte allocation: it received SIGKILL and systemd
reported `Result=oom-kill`. Only the failed test scope was cleaned up; no shared
parent cgroup limit was changed. Raw reports are in `artifacts/memory450/`.

This environment has no user systemd bus, so the per-user launcher command was
unit/dry-run tested; actual enforcement used administrator-created system scopes.
The live Uvicorn API was also restarted inside an equivalent system scope with
the strict flag and a tiny synthetic QAT fixture. Real HTTP JSON/SSE smoke tests
passed; `artifacts/memory450/live-http.json` records the verified limits and health.

These tests establish cgroup enforcement mechanics, not a byte-exact instantaneous
RSS bound, full-model successful-fit guarantee, or model reasoning quality.

## Remaining validation before real deployment

1. Supply a licensed, local Llama-3.2-3B-Instruct or Qwen2.5-3B-Instruct checkpoint.
2. Convert it and verify every model tensor/config is accepted.
3. Check full-model prefill/decode against an independently implemented decoder
   using the **same quantized weights and activation rounding**.
4. Measure maximum-context process RSS and cgroup memory (including page cache)
   under an external hard memory limit.
5. Measure held-out perplexity and downstream tasks; expect substantial quality
   degradation without calibration or quantization/sparsity-aware training.
6. Benchmark cold-file-cache throughput and real storage I/O, not only warm-cache
   synthetic GEMVs. Add authentication and operational controls before exposure.
