# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo fine-tunes **OpenAI GPT-OSS-20B** (a post-trained open-weights MoE model) to be more GPT-4-like in reasoning and conversational tone, using SFT and DPO. The primary workflow is Jupyter notebooks executed on Lightning AI studios with H200/T4 GPUs.

## Environment Setup

The environment is managed via **micromamba** (Conda-compatible). The pinned env is in `lightning.bak/environment.yml` and full pip deps are in `lightning.bak/requirements.txt`.

Key package versions: `unsloth==2025.11.2`, `trl==0.22.2`, `transformers==4.56.2`, `torch==2.8.0`, `peft==0.17.1`.

```bash
# Restore environment from backup
micromamba env create -f lightning.bak/environment.yml
micromamba activate cloudspace
pip install -r lightning.bak/requirements.txt
```

## Running the Inference Server

```bash
python server.py
# Listens on :8080, POST /generate with {"prompt": "...", "max_new_tokens": 128}
```

## Notebook Workflow

The main work is in notebooks — run them sequentially in JupyterLab:

1. **`SFT_datasets.ipynb`** — downloads and formats SFT datasets (open-thoughts, gsm8k, dolly-15k, ultrachat_200k, etc.)
2. **`SFT_fine_tune.ipynb`** / **`gpt_oss_20B_Fine_tuning.ipynb`** — QLoRA SFT training via Unsloth
3. **`merge_and_export_gpt_oss_20b.ipynb`** — merge LoRA adapters + export BF16 HF model
4. **`merge_and_export_gpt_oss_20b_MXFP4_Q4-K-M.ipynb`** — MXFP4 re-quantization + GGUF conversion

## Model Format Pipeline

The MXFP4 on-disk format cannot be used for QLoRA directly. Unsloth unpacks it internally:

```
unsloth/gpt-oss-20b (MXFP4 MoE)
  → BNB NF4 + LoRA adapters  (training)
  → merged BF16 HF            (save_pretrained_merged, default)
  → FP16 GGUF                 (convert_hf_to_gguf.py --outtype f16)
  → Q4_K_M GGUF               (llama.cpp quantize)
```

HF-to-GGUF conversion only supports: `f32, f16, bf16, q8_0, tq1_0, tq2_0, auto` — you cannot convert directly to Q4_K_M.

## llama.cpp Compilation

Compile inside the micromamba ML env (no system CUDA needed). Specify a single target arch to avoid compiling for every known GPU:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

Default Conda GCC is conservative and slow; `DCMAKE_CUDA_ARCHITECTURES=80` is critical for performance.

## Training Tips

- Use `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (not the base MXFP4 model) for QLoRA — it exposes standard `nn.Linear` layers.
- **Before a long training run**: run 1–5 steps, observe GPU VRAM with `nvidia-smi`, then tune `per_device_train_batch_size` and `gradient_accumulation_steps` to fill VRAM.
- For T4 OOM: reduce `max_seq_length` to 512–1024, enable `gradient_checkpointing=True`.
- Unsloth recommends ~75% reasoning data in the SFT mix to avoid degrading base model reasoning.
- `SFTTrainer` automatically uses `dataset_text_field="text"` — no manual tokenization needed when using Unsloth's trainer.

## Key Distinction: finetune.py vs Notebooks

`finetune.py` + `dataset.py` are a **CLIP fine-tuning** prototype (image-text contrastive learning via PyTorch Lightning) from an earlier stage of the project — not related to the GPT-OSS-20B SFT pipeline. The active work is in the notebooks.
