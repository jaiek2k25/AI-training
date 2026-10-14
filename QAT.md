# Pentanary quantization-aware fine-tuning

This is an optional **cloud-side training extension**, not a replacement for the
lightweight inference environment. It implements STE-based QAT of existing local
Llama/Qwen2 checkpoints, and exports directly into the existing PENTA5 v1 runtime.

## Important corrections to the proposed workflow

- Five-state quantization combined with naive 50% pruning can substantially harm
  a pretrained model. The exact damage must be measured; it is not inevitable
  that every PTQ model becomes gibberish. Checkpoints may use FP16, BF16, or other
  training/storage precisions.
- QAT and/or distillation **may** recover quality. Passing STE gradients does not
  prove preservation of intelligence. An STE is a surrogate gradient, not the
  true derivative of rounding. A short run can help, do nothing, or regress.
- Training duration depends on hardware, sequence length, data volume, model
  size and optimizer. **No 1–2 hour recovery promise is justified.**
- Three weights per byte means **3 billion weights ≈ 1 billion bytes**, not
  420 MB. QAT changes values, not the packing ratio. 50% zero states do not
  shrink the current fixed byte format. Norms/biases and metadata add overhead.
- A 420 MB budget holds about **1.26 billion packed weights**, before overhead.
  Storing only nonzeros would require a new format, locations/masks, a new loader,
  and a fresh size/performance analysis. It is not implemented here.
- Transfer **`model_pentanary.bin`, `config.json`, and `tokenizer.json`**, not just
  one binary. Rebuild the native library on the laptop. The existing streaming
  inference memory caveats still apply.

## What is implemented

### Forward/backward contract (`qat.py`)

1. Keep **FP32 master parameters** for optimization.
2. Fit a positive scale per matrix using the existing least-squares PTQ initializer.
   Round it to FP16 once and **freeze it** throughout training. Scales receive no
   gradient, matching the simple STE formulation rather than a learned-scale method.
3. In every forward pass, construct exactly the five-state matrix used at export.
   By default, choose the two largest magnitudes in each four-column group, with
   stable tie breaking. Pruned entries are zero; surviving nonzeros choose ±1/±2
   at `1.5 * alpha`. Masks are dynamic and recomputed from the current master.
4. In backward, pass the weight gradient unchanged, including through pruned or
   saturated entries. This allows previously pruned weights to regrow. It is a
   chosen surrogate estimator, not an unbiased gradient of the discrete operation.
5. FP16-round normalization vectors and biases with an STE as well. Linear input
   activations fake-quantize per token to the native kernel's symmetric int16 range
   by default. `--no-activation-qat` disables this approximation in training only.
6. Shared embedding/output parameters retain **one shared master and scale**.
   They are not silently untied or optimized twice. The CLI omits the duplicate
   output tensor for a genuinely tied output head, preserving compact storage.
7. Export in bounded chunks using the **existing trained scales and exactly the
   same state-selection rules**. No new PTQ fit or post-training sparsity pass
   is applied. Norm/bias vectors receive the same FP16 rounding as in training.

`--dense` disables 2:4 masks consistently in training and export; it does not
change the fixed bytes-per-weight ratio. Dense ties use half-away-from-zero,
matching the existing quantizer rather than `torch.round`'s ties-to-even.

**Do not run `quantize.py` on the trained parameters afterwards.** Re-fitting
scales or introducing pruning only at export changes what the model was trained
against. `train_qat.py` calls `export_qat` directly.

### Scope and numerical limits

The driver performs full-parameter causal-LM fine-tuning with AdamW, gradient
accumulation, optional gradient checkpointing, gradient clipping, and held-out
evaluation. It is single-device and FP32; there is **no AMP, DDP/FSDP/ZeRO, LoRA,
optimizer offload, learned scales, distillation teacher, training checkpoint
resume, or assistant-only label masking**. Avoid claiming these capabilities.

HF teacher-forced attention uses floating point; the deployed runtime uses a
FP16 KV cache and a different execution order. Its int64 weight accumulators
also differ from floating-point training GEMMs. Activation fake quantization
reduces one mismatch but does **not** make the complete graphs bit-identical.
Tests compare exported tiny-model logits within a tolerance, not bit-for-bit.

## Training memory: not a 4 GB task

At 3 billion parameters, FP32 masters require ~12 GB, gradients another ~12 GB,
and two AdamW moment buffers another ~24 GB: **~48 GB before activations,
fake-quantized matrices, temporary tensors, and CUDA/framework overhead**.
The fake-quantized forward weights also occupy memory. Activation gradient
checkpointing does not eliminate optimizer state or master parameters.

Plan capacity from an actual profiling run. A free Colab GPU or a 24 GB GPU
should not be assumed sufficient for this implementation's full 3B QAT.
An appropriate high-memory accelerator or future sharded/offloaded trainer is
needed. Train a much smaller model to validate the pipeline first. The tiny
CPU fixtures below are not indicative of 3B memory or training time.

## Install the optional environment

Use the cloud machine's CUDA-compatible PyTorch installation. PyPI may install
large CUDA dependency wheels even when tests will run on CPU. Keep training
dependencies out of the laptop inference environment.

```bash
# Nested under .venv to keep all training packages out of repository snapshots.
python3 -m venv .venv/qat
.venv/qat/bin/pip install -r requirements-qat.txt
make test-qat
```

The tested stack is PyTorch 2.6.0 and Transformers 4.48.3. Exact package versions
are in `requirements-qat-lock.txt`; CUDA/CPU wheel selection is deployment-specific.
`requirements.txt` remains sufficient for inference, with no torch dependency.
On Arch Linux, install `python`, `python-pip`, and `base-devel` as needed, create
a venv, and compile on that laptop. This workspace's verification was Linux/GCC,
not an actual Arch installation.

## Data format and loss

Provide two distinct local UTF-8 JSONL files with string `text` fields. The driver
streams records rather than holding an entire dataset in RAM. Each record is
limited to 1 MiB. Data order is deterministic and repeats across epochs; there
is no automatic shuffle or deduplication, so preprocess/shuffle/split deliberately.

For instruction data, **preformat each complete conversation** with the model's
canonical template. For example, a Qwen conversation record can contain:

```json
{"text":"<|im_start|>user\nExplain integer shifts.<|im_end|>\n<|im_start|>assistant\nA left shift moves the bit pattern toward higher powers of two.<|im_end|>"}
```

JSON encodes the newlines as `\n`. Ordinary text is also accepted for language
model adaptation. No extra BOS/EOS or chat template is inserted automatically.
The trainer uses causal loss on **all nonpadding next-token targets**, including
prompt/system/user tokens. If you need assistant-only supervised fine-tuning,
add explicit loss masks before treating this as that training objective.

Long records are split into nonoverlapping context-length chunks; context and
next-token targets do not cross chunk boundaries. Trailing one-token chunks
have no next-token target and are dropped. Held-out metrics are weighted by
predicted token count, not by the number of padded batches. `--eval-batches`
limits evaluation to a deterministic prefix of the evaluation data: it is not
necessarily full-corpus perplexity. Avoid train/eval overlap beyond merely using
different filenames.

## Cloud training command

The source directory must contain a standard local HF model checkpoint,
`config.json`, and `tokenizer.json`. Model licensing/access and dataset preparation
are separate. No downloads or `trust_remote_code` are enabled.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/qat/bin/python train_qat.py \
  --checkpoint checkpoints/Qwen2.5-3B-Instruct \
  --train-data data/train.jsonl \
  --eval-data data/heldout.jsonl \
  --output artifacts/qwen-pentanary-qat \
  --device cuda \
  --steps 1000 --sequence-length 128 --batch-size 1 --grad-accum 8 \
  --lr 1e-5 --gradient-checkpointing
```

These are example hyperparameters, **not a proven recipe**. Use a new output
directory for each experiment; the driver rejects existing output directories.
For the tiny synthetic fixture use `--device cpu`. Do not run a real 3B training
job on a small laptop using that switch.

Outputs:

- `model_pentanary.bin`: deployment weights with unchanged QAT scales/states.
- `model_pentanary.bin.json`: quantization provenance, scales and zero fractions.
- `config.json`, `tokenizer.json`: deployment architecture and tokenization.
- Optional tokenizer/generation metadata copied from the source.
- `qat_metrics.json`: original FP32, pre-QAT quantized, and final QAT held-out
  NLL/perplexity, token counts, training losses, settings and elapsed train time.

`improved_over_ptq` is a comparison, **not an automatic quality certification**.
The script warns if held-out NLL does not improve, but still exports the candidate.
It does not save a resumable optimizer/master checkpoint or automatically select
the best intermediate model. Keep the original pretrained source intact.

## Laptop deployment

Copy the three required deployment files to the laptop, build with `make`, and
run the unchanged runtime in its inference-only environment:

```bash
# Requires cgroup v2 and a delegated user systemd manager; see MEMORY.md.
.venv/bin/python run_450.py -- \
  --model artifacts/qwen-pentanary-qat/model_pentanary.bin \
  --config artifacts/qwen-pentanary-qat/config.json \
  --tokenizer artifacts/qwen-pentanary-qat/tokenizer.json \
  --max-context 512
```

Training RAM and inference RAM are distinct budgets. Storage/page-cache effects,
context size, original tokenizer size and cold-disk streaming still matter.
QAT does not by itself make the runtime faster or guarantee sub-450 MB RSS.
The [strict 450 MB profile](MEMORY.md) enforces a kernel-accounted limit; it may
terminate an oversized workload rather than ensure successful inference.

## Verified here

- **20 optional QAT tests passed** on CPU, including STE gradients, pruning/tie
  rules, activation rounding, optimizer updates, fixed-scale export parity,
  tied parameters, token-weighted evaluation and the CLI.
- Tiny HF Llama and Qwen models completed gradient-checkpointed training, exported
  successfully, and matched the native decoder's logits within test tolerance.
  A tied Qwen output head was also tested without duplicating its packed tensor.
- PyTorch checkpoint input, previously skipped in the inference-only environment,
  was exercised successfully in the optional training environment.
- A separate **20-step synthetic run** on a random two-layer, hidden-size-32
  Qwen fixture reduced held-out toy NLL from **4.9331 to 2.1437** (perplexity
  **138.81 to 8.53**) on 40 predicted tokens. The FP32 random-weight baseline was
  NLL 5.1704. Training took ~1.11 s on this tiny fixture, excluding loading,
  scale fitting and evaluation. This is learning a tiny templated task from
  random weights, **not recovering intelligence in a pretrained model**.

No 3B pretrained adaptation, cloud GPU run, real instruction benchmark,
production latency target, or Arch laptop measurement was performed.
