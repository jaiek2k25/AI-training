#!/usr/bin/env python3
"""Cloud-oriented, single-device full-parameter QAT of a local HF checkpoint.

Data: JSONL records {"text": "..."}, already formatted for your model if using
instruction conversations. No remote downloads, arbitrary remote code, or
promises of accuracy recovery. FP32 masters + AdamW are memory-intensive.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import shutil
import time
import torch
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForCausalLM
from qat import install_qat, export_qat


def token_rows(path, tokenizer, sequence_length):
    # Bounded records; preprocessing must deliberately format instructional data.
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline((1 << 20) + 1)
            if not line:
                break
            if len(line.encode('utf-8')) > 1 << 20:
                raise ValueError('dataset record exceeds 1 MiB; split it first')
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise ValueError('each JSONL record must have a string text field')
            # Match runtime: canonical markers must already be in the text;
            # never silently insert an extra BOS via a tokenizer postprocessor.
            ids = tokenizer.encode(item['text'], add_special_tokens=False).ids
            # Split rather than discard all tokens past the first context window.
            for start in range(0, len(ids), sequence_length):
                chunk = ids[start:start + sequence_length]
                if len(chunk) >= 2:
                    yield chunk


def batches(path, tokenizer, sequence_length, batch_size, pad_id, repeat=False):
    while True:
        pending, seen = [], False
        for row in token_rows(path, tokenizer, sequence_length):
            seen = True
            pending.append(row)
            if len(pending) == batch_size:
                yield collate(pending, pad_id)
                pending = []
        if pending:
            yield collate(pending, pad_id)
        if not seen:
            raise ValueError(f'{path}: no records with at least two tokens')
        if not repeat:
            return


def collate(rows, pad_id):
    width = max(map(len, rows))
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = 1
        labels[i, :len(row)] = ids[i, :len(row)]
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def target_count(batch):
    # HF causal language models predict labels[:, 1:] from logits[:, :-1].
    return int((batch['labels'][:, 1:] != -100).sum())


@torch.no_grad()
def evaluate(model, iterator, max_batches, device):
    was_training = model.training
    model.eval()
    total_nll, tokens, count = 0.0, 0, 0
    try:
        for batch in itertools.islice(iterator, max_batches):
            n = target_count(batch)
            loss = model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite evaluation loss')
            total_nll += float(loss) * n
            tokens += n
            count += 1
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError('evaluation dataset is empty')
    nll = total_nll / tokens
    return {'nll': nll, 'perplexity': math.exp(nll) if nll < 700 else None,
            'target_tokens': tokens, 'batches': count}


def train_steps(model, optimizer, iterator, steps, device, grad_accum=1, clip=1.0):
    model.train()
    losses = []
    for step in range(steps):
        # Weight gradients by target tokens, not by differently padded batch sizes.
        microbatches = [next(iterator) for _ in range(grad_accum)]
        total_tokens = sum(target_count(batch) for batch in microbatches)
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for batch in microbatches:
            n = target_count(batch)
            result = model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
            loss = result.loss
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training loss; reduce learning rate/check data')
            weighted = loss * (n / total_tokens)
            weighted.backward()
            step_loss += float(weighted.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        losses.append(step_loss)
        print(json.dumps({'step': step + 1, 'train_nll': step_loss, 'target_tokens': total_tokens}), flush=True)
    return losses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--eval-data', type=Path, required=True, help='separate held-out JSONL')
    parser.add_argument('--output', type=Path, required=True, help='new deployment directory')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--eval-batches', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dense', action='store_true', help='disable 2:4 masks in training and export')
    parser.add_argument('--no-activation-qat', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.grad_accum, args.eval_batches) < 1 or args.sequence_length < 2:
        parser.error('steps/batch/accumulation/evaluation must be positive; context at least 2')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('learning rate must be finite and positive')
    if args.train_data.resolve() == args.eval_data.resolve():
        parser.error('training and held-out evaluation must use different files')
    if not args.checkpoint.is_dir():
        parser.error('--checkpoint must be a local HF checkpoint directory')
    if args.output.resolve() == args.checkpoint.resolve() or args.output.exists():
        parser.error('--output must be a new directory, not the source checkpoint')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu for tiny tests only')
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    if config.model_type not in ('llama', 'qwen2'):
        parser.error('only Llama and Qwen2 dense decoder checkpoints are supported')
    scaling = config.rope_scaling or {}
    if (getattr(config, 'hidden_act', 'silu') != 'silu'
            or getattr(config, 'use_sliding_window', False)
            or scaling.get('rope_type', scaling.get('type', 'default')) not in ('default', 'llama3')):
        parser.error('checkpoint uses architecture options not supported by the runtime')
    if args.sequence_length > config.max_position_embeddings:
        parser.error('sequence-length exceeds checkpoint position limit')
    tokenizer = Tokenizer.from_file(str(args.checkpoint / 'tokenizer.json'))
    if tokenizer.get_vocab_size(with_added_tokens=True) > config.vocab_size:
        parser.error('tokenizer vocabulary exceeds model vocabulary')
    markers = (['<|im_start|>', '<|im_end|>'] if config.model_type == 'qwen2' else
               ['<|begin_of_text|>', '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>'])
    if any(tokenizer.token_to_id(marker) is None for marker in markers):
        parser.error('tokenizer lacks the supported Qwen2/Llama3 chat markers')
    pad = config.pad_token_id if config.pad_token_id is not None else 0
    if not 0 <= pad < config.vocab_size:
        parser.error('invalid pad token id')
    # No device_map/offload/LoRA: optimize full FP32 masters on one device.
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, config=config, torch_dtype=torch.float32,
                                               local_files_only=True, trust_remote_code=False,
                                               attn_implementation='eager').to(args.device)
    model.config.use_cache = False
    def eval_batches():
        return batches(args.eval_data, tokenizer, args.sequence_length, args.batch_size, pad)
    baseline = evaluate(model, eval_batches(), args.eval_batches, args.device)
    state = install_qat(model, sparse=not args.dense, activation_quantization=not args.no_activation_qat)
    ptq = evaluate(model, eval_batches(), args.eval_batches, args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    iterator = batches(args.train_data, tokenizer, args.sequence_length, args.batch_size, pad, repeat=True)
    start = time.perf_counter()
    losses = train_steps(model, optimizer, iterator, args.steps, args.device, args.grad_accum)
    elapsed = time.perf_counter() - start
    final = evaluate(model, eval_batches(), args.eval_batches, args.device)
    args.output.mkdir(parents=True)
    export_qat(state, args.output / 'model_pentanary.bin',
               omit_tied_output=bool(getattr(config, 'tie_word_embeddings', False)))
    # Native runtime needs all three files, not just packed weights.
    model.config.to_json_file(args.output / 'config.json')
    shutil.copy2(args.checkpoint / 'tokenizer.json', args.output / 'tokenizer.json')
    for extra in ['tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json']:
        if (args.checkpoint / extra).exists():
            shutil.copy2(args.checkpoint / extra, args.output / extra)
    metrics = {'baseline_fp32': baseline, 'before_qat': ptq, 'after_qat': final,
               'improved_over_ptq': final['nll'] < ptq['nll'],
               'train_seconds': elapsed, 'train_losses': losses,
               'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'notes': ['HF teacher-forced evaluation, not native-runtime perplexity.',
                         'No distillation teacher; no quality-recovery guarantee.',
                         'FP16 KV cache and finite-precision arithmetic can differ at deployment.']}
    (args.output / 'qat_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    if not metrics['improved_over_ptq']:
        print('WARNING: held-out QAT NLL did not improve. Export is a candidate, not an approved model.')


if __name__ == '__main__':
    main()
