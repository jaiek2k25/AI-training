#!/usr/bin/env python3
"""Reproducible synthetic verification. No pretrained checkpoint is downloaded."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np
from native import NativeModel
from quantize import quantize
from penta_format import read_directory, unpack_pentanary
from synthetic import write_safetensors, create_checkpoint


def worker(path, repeats):
    with NativeModel(path) as model:
        rows, columns = model.tensors['benchmark.weight'].shape
        x = np.random.default_rng(123).normal(size=columns).astype(np.float32)
        model.matvec('benchmark.weight', x)
        begin = time.perf_counter()
        for _ in range(repeats):
            result = model.matvec('benchmark.weight', x)
        seconds = (time.perf_counter() - begin) / repeats
        print(json.dumps({'shape': [rows, columns], 'seconds_per_matvec': seconds,
                          'million_logical_weights_per_second': rows * columns / seconds / 1e6,
                          'worker_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          'checksum': float(result.astype(np.float64).sum()), 'repetitions': repeats}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/benchmark')
    parser.add_argument('--worker')
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.repeats)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'matrix.safetensors'
    target = directory / 'matrix.bin'
    rng = np.random.default_rng(42)
    matrix = rng.normal(0, .1, (2048, 4096)).astype(np.float16)
    write_safetensors(source, {'benchmark.weight': matrix})
    del matrix
    start = time.perf_counter()
    report = quantize(source, target)
    quant_seconds = time.perf_counter() - start
    # Separate process excludes checkpoint creation and dense reference from
    # the runtime's high-water RSS. Page cache is NOT included in process RSS.
    env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    native = json.loads(subprocess.check_output([sys.executable, __file__, '--worker', str(target),
                                                '--repeats', str(args.repeats)], env=env, text=True))
    start, entries = read_directory(target)
    t = entries['benchmark.weight']
    with target.open('rb') as f:
        f.seek(start + t.offset)
        q = unpack_pentanary(f.read(t.nbytes), t.size).reshape(t.shape)
    x = np.random.default_rng(123).normal(size=t.shape[1]).astype(np.float32)
    dense = q.astype(np.float64) @ x.astype(np.float64) * t.alpha
    with NativeModel(target) as model:
        actual = model.matvec('benchmark.weight', x)
    relative_error = float(np.linalg.norm(actual - dense) / np.linalg.norm(dense))
    assert relative_error < 1e-3
    assert native['worker_peak_rss_bytes'] < 500_000_000
    demo = directory / 'tiny_model'
    create_checkpoint(demo)
    quantize(demo, demo / 'model_pentanary.bin')
    from runtime import Runtime
    rt = Runtime(demo / 'model_pentanary.bin', demo / 'config.json', demo / 'tokenizer.json', max_context=64)
    prompt = rt.encode_messages([{'role': 'user', 'content': 'hello world'}])
    begin = time.perf_counter()
    generated = list(rt.generate(prompt, max_tokens=16))
    elapsed = time.perf_counter() - begin
    result = {'notice': 'Synthetic only: not 3B memory, speed, or quality validation.',
              'matrix': {**native, 'packed_file_bytes': target.stat().st_size,
                         'fp16_weight_bytes': t.size * 2, 'quantization_seconds': quant_seconds,
                         'zero_fraction': report['tensors'][0]['zero_fraction'],
                         'weight_relative_rmse': report['tensors'][0]['relative_rmse'],
                         'kernel_relative_l2_error_vs_dequantized_float_matvec': relative_error},
              'tiny_random_transformer': {'layers': 2, 'hidden_size': 32, 'prompt_tokens': len(prompt),
                                         'completion_tokens': generated[-1][1],
                                         'prefill_and_decode_seconds': elapsed,
                                         'kv_cache_bytes': rt.kv_bytes}}
    (directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
