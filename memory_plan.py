#!/usr/bin/env python3
"""Report actual PENTA5 file storage separately from estimated inference memory."""
import argparse
import json
from pathlib import Path
from penta_format import read_directory, PACKED, FP16
from memory_limits import (STRICT_LIMIT_BYTES, NATIVE_BUFFER_BYTES,
                           kv_cache_bytes, scratch_reserve_bytes)


def memory_plan(model_path, config, context=512, engine_overhead_bytes=80_000_000,
                limit_bytes=STRICT_LIMIT_BYTES, headroom_bytes=30_000_000):
    if (limit_bytes <= 0 or headroom_bytes < 0 or headroom_bytes >= limit_bytes
            or engine_overhead_bytes <= 0):
        raise ValueError('invalid memory budget or overhead estimate')
    start, tensors = read_directory(model_path)
    packed = [t for t in tensors.values() if t.kind == PACKED]
    vectors = [t for t in tensors.values() if t.kind == FP16]
    packed_bytes = sum(t.nbytes for t in packed)
    file_bytes = Path(model_path).stat().st_size
    kv = kv_cache_bytes(config, context)
    head_dim = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
    scratch = scratch_reserve_bytes(head_dim, config['vocab_size'], context)
    activation_bytes = 4 * max((t.shape[-1] for t in packed), default=0)
    # Conservative estimate, not measured RSS: engine overhead includes Python,
    # native libraries, tokenizer, directory metadata and HTTP framework.
    nonweight = engine_overhead_bytes + kv + scratch + NATIVE_BUFFER_BYTES + activation_bytes
    admitted = limit_bytes - headroom_bytes
    resident_total = file_bytes + nonweight
    # File layout overhead includes all FP16 tensors, alpha/directory, padding.
    format_overhead = file_bytes - packed_bytes
    max_packed_states = 3 * max(0, admitted - nonweight - format_overhead)
    return {
        'units': 'bytes (decimal MB = 1,000,000 bytes)',
        'hard_limit_bytes': limit_bytes,
        'headroom_bytes': headroom_bytes,
        'admission_target_bytes': admitted,
        'storage': {'model_file_bytes': file_bytes,
                    'packed_weight_states': sum(t.size for t in packed),
                    'packed_weight_bytes': packed_bytes,
                    'fp16_vector_parameters': sum(t.size for t in vectors),
                    'fp16_vector_bytes': sum(t.nbytes for t in vectors),
                    'directory_and_initial_padding_bytes': start,
                    'other_gaps_or_trailing_bytes': file_bytes - start - sum(t.nbytes for t in tensors.values()),
                    'zero_skipping_storage_discount_bytes': 0},
        'estimated_nonweight_memory': {'engine_tokenizer_framework_bytes': engine_overhead_bytes,
                                      'fp16_kv_cache_bytes': kv,
                                      'decoder_scratch_reserve_bytes': scratch,
                                      'native_io_buffer_bytes': NATIVE_BUFFER_BYTES,
                                      'native_fixed_activation_bytes': activation_bytes},
        'hypothetical_fully_resident': {'estimated_bytes': resident_total,
                                       'fits_admission_estimate': resident_total < admitted,
                                       'upper_bound_packed_states_at_this_overhead': max_packed_states,
                                       'implemented_by_current_loader': False},
        'current_streamed_loader': {'estimated_process_bytes': nonweight,
                                    'fits_process_admission_estimate': nonweight < admitted,
                                    'filesystem_page_cache_budgeted': False,
                                    'hard_limit_or_success_guaranteed_by_this_report': False},
        'warnings': ['This is a planning estimate, not a measurement or an OS limit.',
                     'Zero arithmetic skipping does not shrink the packed payload.',
                     'Streamed file size is not resident process memory; cgroup page cache can add pressure.',
                     'KV cache is ordinary FP16, not a compressed ring buffer.',
                     'Quantized weights do not make attention/norms/scaling floating-point-free.',
                     'Measure actual full-context workloads under a cgroup limit; OOM remains possible.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--max-context', type=int, default=512)
    p.add_argument('--engine-overhead-mb', type=int, default=80, help='estimate, not a measured value')
    p.add_argument('--headroom-mb', type=int, default=30)
    p.add_argument('--require-resident-fit', action='store_true',
                   help='exit 2 if the hypothetical fully resident layout does not fit the estimate')
    args = p.parse_args()
    try:
        report = memory_plan(args.model, json.loads(Path(args.config).read_text()), args.max_context,
                             args.engine_overhead_mb * 1_000_000, headroom_bytes=args.headroom_mb * 1_000_000)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps(report, indent=2))
    if args.require_resident_fit and not report['hypothetical_fully_resident']['fits_admission_estimate']:
        p.exit(2, 'Fully resident estimate exceeds the budget; choose a smaller checkpoint or stream weights.\n')


if __name__ == '__main__':
    main()
