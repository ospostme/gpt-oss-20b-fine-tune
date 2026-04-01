---
base_model: unsloth/gpt-oss-20b
tags:
  - gguf
  - quantized
  - mxfp4
  - iq4_xs
---

# GPT-OSS-20B SFT — GGUF

Fine-tuned GPT-OSS-20B (SFT) exported as GGUF for llama.cpp inference.

## Files

| File | Size | Description |
|---|---|---|
| `gpt-oss-20b.MXFP4.gguf` | ~13.8 GB | Native MXFP4 format — highest quality |
| `gpt-oss-20b-sft-IQ4_XS-hybrid.gguf` | ~11.6 GB | Hybrid: MXFP4 experts + IQ4_XS attention |

## Quantization Strategy

GPT-OSS-20B uses MXFP4 MoE expert weights that **cannot be requantized without quality loss**.
The hybrid GGUF keeps expert weights in their native MXFP4 format:

- `ffn_*_exps` (72 tensors) → `IQ4_NL` — MXFP4 preserved
- `attn_q/k` and other 2880-column tensors → `IQ4_NL` — avoids 256-block fallback
- All other tensors → `IQ4_XS` + imatrix calibration

## Usage

```bash
llama-cli --model gpt-oss-20b-sft-IQ4_XS-hybrid.gguf --n-gpu-layers 99 -p "Your prompt here"
```
