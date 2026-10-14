#!/usr/bin/env python3
"""Create a deterministic TINY RANDOM Qwen2 checkpoint; NOT a pretrained model."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np


def write_safetensors(path, tensors, bf16=False):
    header, offset, arrays = {}, 0, []
    for name, x in tensors.items():
        x = np.asarray(x)
        if bf16:
            raw = (x.astype('<f4').view('<u4') >> 16).astype('<u2').tobytes()
            dtype = 'BF16'
        else:
            raw = x.astype('<f2').tobytes()
            dtype = 'F16'
        header[name] = {'dtype': dtype, 'shape': list(x.shape), 'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
        arrays.append(raw)
    encoded = json.dumps(header).encode()
    encoded += b' ' * (-len(encoded) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
        for raw in arrays:
            f.write(raw)


def create_checkpoint(directory, hidden=32, layers=2, seed=42, family='qwen2'):
    from tokenizers import Tokenizer, models, pre_tokenizers, AddedToken
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    heads, kv_heads, intermediate, vocab = 4, 2, hidden * 2, 128
    config = {'model_type': family, 'hidden_size': hidden, 'intermediate_size': intermediate,
              'num_hidden_layers': layers, 'num_attention_heads': heads, 'num_key_value_heads': kv_heads,
              'vocab_size': vocab, 'rms_norm_eps': 1e-6, 'rope_theta': 10000,
              'max_position_embeddings': 512, 'eos_token_id': 127, 'tie_word_embeddings': False,
              'synthetic_random_weights': True}
    weights = {}
    def matrix(name, shape):
        weights[name] = rng.normal(0, 0.1, shape).astype(np.float16)
    matrix('model.embed_tokens.weight', (vocab, hidden))
    weights['model.norm.weight'] = np.ones(hidden, np.float16)
    matrix('lm_head.weight', (vocab, hidden))
    for i in range(layers):
        p = f'model.layers.{i}.'
        weights[p + 'input_layernorm.weight'] = np.ones(hidden, np.float16)
        weights[p + 'post_attention_layernorm.weight'] = np.ones(hidden, np.float16)
        for name, shape in [('self_attn.q_proj', (hidden, hidden)),
                            ('self_attn.k_proj', (hidden // 2, hidden)),
                            ('self_attn.v_proj', (hidden // 2, hidden)),
                            ('self_attn.o_proj', (hidden, hidden)),
                            ('mlp.gate_proj', (intermediate, hidden)),
                            ('mlp.up_proj', (intermediate, hidden)),
                            ('mlp.down_proj', (hidden, intermediate))]:
            matrix(p + name + '.weight', shape)
            if family == 'qwen2' and name in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'):
                weights[p + name + '.bias'] = rng.normal(0, .01, shape[0]).astype(np.float16)
    write_safetensors(directory / 'model.safetensors', weights)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    specials = ['[UNK]', '<|im_start|>', '<|im_end|>', '<|begin_of_text|>',
                '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
    words = specials + ['system', 'user', 'assistant', 'hello', 'world', 'test']
    words += [f'word{i}' for i in range(vocab - len(words))]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens([AddedToken(t, special=True) for t in specials])
    tokenizer.save(str(directory / 'tokenizer.json'))
    return config, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/demo')
    args = parser.parse_args()
    create_checkpoint(args.output)
    from quantize import quantize
    quantize(args.output, Path(args.output) / 'model_pentanary.bin')
    print('Synthetic random-weight fixture only; output is not meaningful language.')
