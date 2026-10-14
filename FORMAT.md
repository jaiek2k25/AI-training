# PENTA5 v1 binary format

All integers and FP16 values use **little-endian** encoding. There are no native
C++ structure dumps or alignment-dependent fields. Offsets are byte offsets.

## Header (24 bytes)

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 8 bytes | Magic `PENTA5\0\0` |
| 8 | uint32 | Version, currently 1 |
| 12 | uint32 | Number of tensors, 1–100000 |
| 16 | uint64 | Absolute start of payload area |

The directory immediately follows the header. Writer pads its end with zeros
until the payload starts at a multiple of 64 bytes. Reader allows bounded
padding and does not require this alignment. Header + directory + padding must
fit in 64 MiB.

## Each directory record

| Relative offset | Type | Meaning |
|---:|---|---|
| 0 | uint16 | UTF-8 name byte length (1–4096) |
| 2 | uint8 | Kind: 1 = pentanary; 2 = FP16 |
| 3 | uint8 | Rank, 1–8 |
| 4 | IEEE FP16 | Positive finite `alpha` |
| 6 | uint16 | Flags: bit 0 = 2:4 structural pruning applied; other bits zero |
| 8 | uint64 | Payload offset **relative to the payload area start** |
| 16 | uint64 | Payload length in bytes |
| 24 | uint64[rank] | Dimensions, each positive |
| 24 + 8*rank | bytes[name_length] | UTF-8 name, no NUL or terminator |

Names are unique. Record payload intervals are monotonically ordered and
nonoverlapping. The shape product cannot exceed 2^40 elements. Padding between
payloads/trailing data is permitted by readers but not produced by the writer.
No checksum is stored: structurally valid bit corruption is not detectable.
Do not treat this format as an integrity or authenticity boundary; distribute a
cryptographic checksum separately when needed.

## Kind 1: pentanary matrix

Let `N` be the shape product. Payload length is `ceil(N / 3)` bytes.
Flatten in C (row-major) order. For every three weights:

```
byte = (w0+2) + 5*(w1+2) + 25*(w2+2)
```

Each weight is an integer in `[-2, 2]`. A stored weight represents
`float(alpha) * weight`. Unused positions in the final triplet have weight zero
(digit 2). Readers ignore those positions. Rows are **not separately padded**.
Bytes must be <125. All-zero triplets encode to 62, not 0.

Bit 0 in flags describes converter behavior: two smallest absolute weights per
four-column group are pruned. It does not select another storage layout and does
not make the loader trust a particular count of zeros. Naturally zero surviving
weights can make sparsity greater than 50%.

## Kind 2: FP16 vectors

Payload length is `2*N`, containing ordinary IEEE binary16 values in row-major
order. The converter stores normalization vectors and biases this way, with
`alpha=1` and flags=0. Kind-2 values are already scaled; the loader does not
multiply them by `alpha`. Nonfinite FP16 values are rejected when read.

## Decoder naming

The container itself can name arbitrary tensors. The decoder expects standard
HF Llama/Qwen2 state-dictionary names such as `model.embed_tokens.weight`,
`model.layers.0.self_attn.q_proj.weight`, `model.norm.weight`, and `lm_head.weight`.
Model architecture and tokenizer data are supplied as separate original HF
`config.json` and `tokenizer.json` files, not embedded in the packed file.
