# A defensible 450 MB deployment budget

**450 MB means 450,000,000 bytes, not 450 MiB.** Five-state packing and zero
skipping are already implemented. The proposed claim that 1.3–1.5B packed
weights plus a 100 MB cache fit in ~400 MB is not valid for this format.

## Correct storage arithmetic

Ignoring per-tensor rounding, vectors, scales, and directory overhead:

| Nominal packed matrix weights | Packed payload | Payload + a 100 MB KV cache |
|---:|---:|---:|
| 3.0 billion | 1,000 MB | 1,100 MB |
| 1.5 billion | 500 MB | 600 MB |
| 1.35 billion | 450 MB | 550 MB |
| 1.3 billion | 433.33 MB | 533.33 MB |
| 0.9 billion | 300 MB | 400 MB |
| 0.6 billion | 200 MB | 300 MB |

The last column **still excludes** the tokenizer, Python/native libraries, HTTP
server, activations, attention/sampling scratch space, metadata, and headroom.
It is not a complete engine budget. Actual parameter counts and tied weights
must be read from the exported file, not inferred from a model's marketing name.

A ternary digit is a *trit*. This format packs three **base-5 digits/weight
states** into one byte, not three trits.

### Zero skipping is not a negative RAM allocation

The packed payload remains `sum(ceil(tensor_elements / 3))`. A zero is represented
by digit 2; three zeros encode to byte 62. Skipping its arithmetic does not delete
that byte. Even a completely zero matrix has the same packed length as a nonzero
matrix of the same dimensions. Thus the claimed **“−150 MB saved” is 0 MB** in
this format. Masks/indices and a different loader would be needed for another
sparse storage representation; it would require its own measured size budget.

### What “strictly pentanary” means in this code

Only **matrix weight states** are constrained to `{-2,-1,0,1,2}`, with an FP16
scale per matrix. Norm vectors and biases remain FP16. Linear-weight inner
accumulation uses integer additions, subtractions and safe shifts. Activations
are not limited to five values. Scale application, RMSNorm, attention, RoPE,
SiLU and sampling use floating point. This is **not an entirely floating-point-free
or five-state-arithmetic transformer**. Eliminating all floating operations would
be a new numerical implementation requiring validation and possibly retraining.

QAT can improve adaptation; it cannot promise unchanged reasoning quality.
A smaller checkpoint is not automatically a distilled reasoning model. Evaluate
held-out perplexity and relevant tasks before accepting a deployment candidate.

## The actual loader and KV cache

The native loader already streams weights through a **64 KiB `pread` buffer**.
A 500 MB or 1 GB model file therefore does not imply an equally sized process
resident weight allocation. Downsizing helps I/O and computation but is **not a
mathematical prerequisite for streamed operation** under a smaller process budget.
Cold-disk performance and filesystem cache pressure can be severe.

The current cache is ordinary, preallocated **FP16 K and V**, not a compressed
ring buffer. Its exact allocation is:

```
2 * layers * max_context * num_kv_heads * head_dim * 2 bytes
```

For example, 24 layers, 2 KV heads, head dimension 128 and context 512 require
12,582,912 bytes. This is an illustrative config calculation, not a claim about a
specific checkpoint. Cache memory is **added**, never subtracted, from the budget.
A ring buffer that drops old tokens can change a full-attention model's outputs;
it is not a free, mathematically equivalent optimization. The API instead rejects
requests beyond the configured context limit.

## Inspect a real exported model

```bash
.venv/bin/python memory_plan.py \
  --model artifacts/model_pentanary.bin \
  --config checkpoints/my-model/config.json \
  --max-context 512
```

The report separates:

- Actual packed payload, FP16 vectors, headers/padding and file size.
- **Zero bytes** of storage discount from zero arithmetic skipping.
- Exact KV allocation for the selected context.
- Configurable process/tokenizer/framework allowance (default 80 decimal MB).
- Decoder scratch reserve, native buffer and fixed-point activation array.
- A **hypothetical fully resident** estimate vs the **current streamed** estimate.

Defaults reserve 30 MB below the 450 MB cap. `--engine-overhead-mb` and
`--headroom-mb` are planning inputs, not measurements. The decoder's scratch
reserve alone starts at 64 MiB; do not assign all remaining RAM to weights.
`--require-resident-fit` exits 2 if the *hypothetical* resident layout exceeds
this estimate. There is no resident-loading mode; the existing loader streams.

The reported upper bound on resident packed states is roughly
`3 * (admission_budget - nonweight_memory - format_overhead)`, before tensor-tail
rounding. For example, reserving 150 MB for everything other than weight states
leaves at most ~900M states; reserving 250 MB leaves ~600M. These are arithmetic
examples, **not quality or actual-runtime fit guarantees**. Use the report on the
actual smaller checkpoint instead of automatically choosing a nominal 1.5B model.

## Enforce a kernel memory limit on Arch/Linux

`run_450.py` creates a **systemd user scope** before starting the Python API:

```bash
.venv/bin/python run_450.py -- \
  --model artifacts/model_pentanary.bin \
  --config checkpoints/my-model/config.json \
  --tokenizer checkpoints/my-model/tokenizer.json \
  --max-context 512 --port 8000
```

Requirements: Linux cgroup v2, a running user systemd manager, delegated memory
control, and `systemd-run`. Run from the inference venv, not the PyTorch training
venv. Model paths should point to the three compatible deployment files.

The scope requests:

- `MemoryMax`: **450,000,000 rounded down to whole pages**; on 4096-byte pages,
  this is **449,998,848 bytes**.
- `MemorySwapMax=0`: do not satisfy the workload by swapping anonymous memory.
- `OOMPolicy=kill`: terminate the scope workload on OOM instead of treating a
  partially failed engine as healthy.
- One BLAS/OpenMP thread and bounded glibc allocator arenas via environment settings.

The API's `--strict-450mb` flag is injected automatically. Before model loading,
it reads its **actual** `/proc/self/cgroup` and cgroup-v2 mount, checks effective
memory and swap limits (including visible ancestors), and **refuses startup** if
it cannot verify `memory.max <= 450,000,000` and `memory.swap.max = 0`.
There is no fallback to an unrestricted server. `--dry-run` prints the command
without creating a scope or asserting that limits are active.

Inside the strict profile, process admission defaults to **420,000,000 bytes**,
leaving a planning margin below the kernel cap. This margin is not a separately
reserved physical pool; filesystem cache can consume available space. A smaller explicit soft budget
is accepted; a budget above 420 MB is rejected. The regular server's legacy
`--memory-budget-mb` option alone is only an estimate and periodic RSS check,
**not** a hard limit. The runtime/API constructor alone also does not create a
cgroup; use the launcher or an equivalent administrator-managed scope.

If there is no user systemd manager or memory-controller delegation, startup
fails. An administrator can provision an equivalent system scope. For example,
from a normal login account, with the actual account name substituted:

```bash
sudo systemd-run --scope --quiet \
  --property=MemoryMax=449998848 --property=MemorySwapMax=0 --property=OOMPolicy=kill \
  -- runuser -u YOUR_ACCOUNT -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PWD/.venv/bin/python" "$PWD/server.py" --strict-450mb \
  --model "$PWD/artifacts/model_pentanary.bin" \
  --config "$PWD/checkpoints/my-model/config.json" \
  --tokenizer "$PWD/checkpoints/my-model/tokenizer.json" --max-context 512
```

This alternative keeps the inference process unprivileged. Do not alter a shared
parent cgroup or give the model-serving process general root privileges. The
hard cap covers the scope workload, not the desktop, system manager, or the
entire 4 GB laptop. External supervisors also consume memory outside that scope.

### What the hard limit does and does not promise

Linux enforces **cgroup-accounted memory**, including charged anonymous memory,
page cache and other accounted allocations. This is not identical to process
RSS: shared pages/page-cache attribution can differ, and kernel reclaim can
produce transient accounting overshoots. No claim of byte-exact instantaneous
physical-memory or RSS bounds is made. Smaller ancestor limits may also apply.

The limit can reclaim cache, stall allocations, or **kill the workload**. It
cannot guarantee that a given model will run successfully, preserve intelligence,
or avoid all OOM failures. In particular, “hard cap” and “cannot run out of memory”
are different promises. Filesystem cache is not budgeted by the streamed-process
planner; it is managed under the actual cgroup accounting rules at execution.

`/health` now reports the selected profile, soft admission estimate, RSS, cgroup
current/peak usage, effective limits, swap limits and OOM counters. Verify those
fields and measure full-context, cold-cache workloads on the target machine.

## Verification in this workspace

- Normal suite: **31 passed, 4 skipped** in the inference-only environment.
- Optional QAT suite: **20 passed** in the separate training environment.
- Memory suite with real kernel tests enabled: **17 passed**.
- Native address/undefined-behavior sanitizer checks: passed.

The real kernel tests used **new isolated system scopes**, since this sandbox has
no user systemd bus. They ran the workload as the ordinary `user` account; they
did not change the session's parent cgroup. The tested systemd version was 252.
The per-user launcher command was dry-run/unit-tested, not exercised against a
running user manager in this sandbox.

The tiny synthetic decoder filled its 64-token cache and completed JSON/SSE API
checks under `memory.max=449998848` and `memory.swap.max=0`, with no OOM events.
A separate test intentionally attempted a **600,000,000-byte allocation**; it
was killed with SIGKILL, and systemd reported **`Result=oom-kill`**. Its failed
scope was then cleaned up. These results verify enforcement mechanics, not a
pretrained-model fit or quality result.

To reproduce the optional isolated kernel tests in a suitable Linux workspace:

```bash
PENTA_TEST_SYSTEMD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_memory_budget.py
```

These opt-in tests need noninteractive administrative scope-creation rights and
include a deliberate OOM **inside the new test scope**. Do not run them against
an existing shared scope. Without the opt-in flag, both live kernel tests skip.
The live Uvicorn preview was also restarted in an equivalent verified system
scope, serving the tiny synthetic QAT fixture. Real HTTP JSON and SSE requests
passed. Its health snapshot reported **70,270,976 bytes RSS** and **45,428,736 bytes
peak cgroup charge**, with no OOM events. These are different accounting measures,
including different shared-page attribution—not a sparsity saving. Neither number
is a pretrained model measurement.

Raw results are in `artifacts/memory450/`. No pretrained 0.5B/1.3B/1.5B/3B model,
cold-cache full-model workload, or actual Arch laptop was tested.
