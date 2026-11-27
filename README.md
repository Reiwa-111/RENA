# RENA-RUNE

**RENA-RUNE is a small decoder-only language model trained from scratch on Deltarune transcript text.** The project contains the model implementation, byte-level BPE tokenizer workflow, corpus preparation, training loop, text generation and speaker-based dialogue tools. It is an experiment in building and training a compact language model—not a general-purpose assistant.

<p align="center">
  <strong>Model: 8.34M parameters</strong>
  ·
  <a href="#model-architecture">Architecture</a>
  ·
  <a href="#quick-start">Quick start</a>
  ·
  <a href="#training">Training</a>
  ·
  <a href="#inference">Inference</a>
</p>

## Quick start

Install Python dependencies, validate the configuration and run the model smoke test:

```bash
python -m pip install -r requirements.txt
python RENA_RUNE.py validate-config
python RENA_RUNE.py smoke
```

Prepare the local corpus and tokenizer, then train:

```bash
python RENA_RUNE.py prepare
python RENA_RUNE.py train --steps 50000
```

Sample from a trained checkpoint:

```bash
python RENA_RUNE.py sample --checkpoint runs/rena_rune/best.pt --prompt 'Susie: "' --tokens 100
```

The commands above are run from the repository root. See [Installation](#installation) and [Training](#training) for setup details.

---

## Contents

- [Origin](#origin)
- [Project overview](#project-overview)
- [Model scope](#model-scope)
- [Highlights](#highlights)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Model architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Safetensors export](#safetensors-export)
- [Configuration](#configuration)
- [Validation and tests](#validation-and-tests)
- [Limitations](#limitations)
- [Repository structure](#repository-structure)
- [License and attribution](#license-and-attribution)

---

## Origin

RENA-RUNE was created as a side project during the development and pretraining of RENA 1.0 NANO-D.

The project started as an experiment in adapting and scaling down the RENA architecture to a much smaller model, with a specific focus on dialogue generation using Deltarune-related data.

While RENA-RUNE shares architectural foundations with RENA 0.7–1.0, it is a separate implementation and training project with its own model configuration and pipeline.

## Project overview

RENA-RUNE is a compact autoregressive Transformer experiment focused on learning dialogue and narrative patterns from a Deltarune transcript corpus. It uses a custom model implementation in PyTorch and a byte-level BPE tokenizer with an 8,000-token vocabulary.

| Component | Purpose |
|---|---|
| RENA-RUNE model | Decoder-only autoregressive Transformer |
| Tokenizer | Byte-level BPE training, encoding and decoding |
| Data pipeline | Corpus preparation, tokenization and train/validation split |
| Trainer | Optimization, validation, checkpointing and sampling during training |
| Generation | Prompt-based generation and interactive speaker dialogue |
| Safetensors exporter | Creates a compact inference-oriented model package |
| Configuration | Model, data, training, runtime and dialogue parameters in JSON |

The code is designed to be understandable and modifiable for experimentation. The model's small size and narrow training corpus impose substantial quality limits.

## Model scope

The model is trained from scratch. It is not fine-tuned from a pretrained foundation model and does not have general knowledge beyond what its training process can learn from the corpus.

The current configuration defines the following model:

| Property | Value |
|---|---|
| Parameters | 8,344,832 (approximately 8.34M) |
| Transformer layers | 8 |
| Hidden size (`d_model`) | 256 |
| Query / KV heads | 4 / 2 |
| Head dimension | 64 |
| Feed-forward dimension | 768, SwiGLU |
| Context length | 256 tokens |
| Vocabulary | 8,000 tokens |
| Attention | Grouped-query attention (GQA) |
| Normalization | RMSNorm and QK-Norm |
| Position encoding | RoPE |
| Embeddings | Tied input/output weights |
| Tokenizer | Byte-level BPE |

The values are defined in [`config/rena_rune.json`](config/rena_rune.json). Run `python RENA_RUNE.py inspect-model` to build the configured model and report its parameter count.

## Highlights

### Compact Transformer implementation

- Decoder-only causal language modeling.
- Grouped-query attention with 4 query heads and 2 key/value heads.
- SwiGLU feed-forward blocks.
- RMSNorm, QK-Norm and rotary positional embeddings.
- Tied token embeddings and output projection.
- Configurable context length, vocabulary size and model dimensions.

### Training workflow

- Byte-level BPE tokenizer training and data preparation.
- Seeded chunk-based train/validation partition.
- AdamW optimizer and configurable learning-rate scheduling.
- Warmup and cosine/linear/constant schedules.
- Gradient accumulation and clipping.
- Validation, early stopping, checkpoint saving and resume support.
- Periodic prompt samples and JSONL metrics.

### Dialogue-oriented inference

- Prompt-based text sampling.
- Temperature, top-k and top-p sampling controls.
- Repetition penalty and optional no-repeat n-gram filtering.
- Interactive speaker dialogue with configurable speaker labels and a short rolling context.
- Loading local training checkpoints or Safetensors inference weights.

---

## Installation

Python 3.10 or newer is recommended. Install the dependencies in a virtual environment.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install packages:

```bash
python -m pip install -r requirements.txt
```

PyTorch installation may depend on the target hardware and platform. Install a suitable PyTorch build for your CPU or GPU environment before running training. GPU acceleration is not guaranteed by the project itself.

---

## Dataset and preprocessing

### Dataset source and attribution

RENA-RUNE uses the [Deltarune Complete Transcript Cleaned dataset](https://huggingface.co/datasets/Deltarunefan/Deltarune-Complete-Transcript-Cleaned), published on Hugging Face by **Deltarunefan**. This link identifies the dataset source; it does not imply that this project owns the transcript material or that the dataset grants permission to redistribute it.

The source dataset, cleaned transcript files and preprocessed token data are not included in this repository.

Obtain the data separately and place the cleaned chapter files in the local `data/` directory using the filenames defined in the configuration. The default configuration expects:

```text
data/
  chap1_cleaned.txt
  chap2_cleaned.txt
  chap3_cleaned.txt
  chap4_cleaned.txt
```

Prepare the corpus and tokenizer:

```bash
python RENA_RUNE.py prepare
```

The preparation pipeline combines the source files, trains or loads the tokenizer and creates tokenized training and validation streams. The default validation configuration uses a seeded chunk-based split. Generated data, tokenizer files and other local artifacts are ignored by Git.

Check or retrain the tokenizer separately:

```bash
python RENA_RUNE.py tokenizer check
python RENA_RUNE.py tokenizer train
```

Data usage and redistribution must comply with the dataset's terms and the rights of the underlying material.

---

## Model architecture

RENA-RUNE predicts the next token autoregressively. Input tokens are mapped to embeddings, processed by causal Transformer blocks and projected to vocabulary logits through a tied embedding matrix.

```text
Token IDs
    │
    ▼
Token embeddings ── RoPE positions
    │
    ▼
8 × Transformer block
  ├─ RMSNorm / QK-Norm
  ├─ Causal Grouped-Query Attention (4 Q / 2 KV heads)
  ├─ Residual connection
  ├─ RMSNorm
  ├─ SwiGLU feed-forward (768)
  └─ Residual connection
    │
    ▼
Final normalization
    │
    ▼
Tied vocabulary projection
    │
    ▼
Next-token logits
```

This is a small experimental architecture. The context window is fixed by the model configuration and currently defaults to 256 tokens.

---

## Training

Run the configured training schedule:

```bash
python RENA_RUNE.py train --steps 50000
```

Useful overrides:

```bash
python RENA_RUNE.py train --steps 10000 --lr 0.00015 --batch-size 16
python RENA_RUNE.py train --device cpu --threads 8
python RENA_RUNE.py train --resume auto
```

Training prepares the data when necessary, loads the token streams, constructs the model and optimizer, then records training and validation metrics. Checkpoints are written to the run directory, which defaults to `runs/rena_rune/`.

Common training options:

| Option | Description |
|---|---|
| `--steps` | Override the number of training steps and scheduler duration |
| `--lr` | Set the learning rate |
| `--batch-size` | Set the micro-batch size |
| `--device` | Select a device, such as `cpu` or a supported GPU device |
| `--threads` | Set CPU thread count |
| `--run-dir` | Change the output directory |
| `--resume` | Resume from a checkpoint path or use `auto` |
| `--force-prepare` | Force data preparation during training |

The training run produces local checkpoints (`best.pt`, `last.pt`), metrics and configuration snapshots. These are training artifacts and are not intended for source control.

---

## Inference

### Text generation

Generate a continuation from a prompt:

```bash
python RENA_RUNE.py sample \
  --checkpoint runs/rena_rune/best.pt \
  --prompt 'Susie: "' \
  --tokens 100 \
  --temperature 0.85 \
  --top-k 40 \
  --top-p 0.95
```

Sampling options include token count, temperature, top-k, top-p, repetition penalty, no-repeat n-gram size and an optional random seed.

### Interactive dialogue

Start a speaker-oriented dialogue session:

```bash
python RENA_RUNE.py talk \
  --checkpoint runs/rena_rune/best.pt \
  --speaker Susie \
  --as-speaker Kris
```

Available session commands:

| Command | Action |
|---|---|
| `/exit` | Exit dialogue |
| `/reset` | Clear the rolling conversation history |
| `/speaker NAME` | Change the model's speaker label |
| `/temp X` | Change sampling temperature |

Dialogue uses a small rolling context and should be treated as an experimental text-generation interface, not a faithful simulation of a character.

---

## Safetensors export

For inference distribution, export a local training checkpoint into a separate package:

```bash
python tools/export_safetensors.py runs/rena_rune/best.pt --output-dir release
```

The exporter writes:

| File | Purpose |
|---|---|
| `model.safetensors` | Model weights and embedded inference metadata/configuration |
| `config.json` | Sanitized inference configuration |
| `tokenizer.json` | Matching tokenizer |
| `metadata.json` | Export metadata |

Weights default to FP16. Use `--dtype fp32` for full precision export. The inference loader supports `.safetensors` packages as well as local `.pt` checkpoints. Training checkpoints contain additional state and should remain separate from inference releases.

Before distributing model weights or tokenizer artifacts, review the dataset's terms and the rights applicable to transcript-derived materials. A code license does not automatically grant permission to redistribute dataset-derived artifacts.

See [`PUBLISHING.md`](PUBLISHING.md) for the release checklist.

---

## Configuration

The default configuration is [`config/rena_rune.json`](config/rena_rune.json). It controls:

- Model dimensions and attention settings.
- Tokenizer vocabulary and special tokens.
- Corpus file locations and validation split.
- Training steps, batch size, optimizer, scheduler and checkpoint intervals.
- Runtime device, threads and mixed precision.
- Generation sampling and dialogue defaults.

Pass a different configuration file using `--config`:

```bash
python RENA_RUNE.py --config config/rena_rune.json validate-config
```

Inspect the configured model and parameter count:

```bash
python RENA_RUNE.py inspect-model
```

---

## Validation and tests

Run the basic checks before training or making changes:

```bash
python RENA_RUNE.py validate-config
python RENA_RUNE.py smoke
python -m py_compile RENA_RUNE.py rena_rune/*.py tools/export_safetensors.py
```

`validate-config` checks and prints the configuration summary. `smoke` runs a small forward pass, computes loss, performs backpropagation and verifies generation. The compile check catches Python syntax errors in the project modules and exporter.

---

## Limitations

- The model has only about 8.34M parameters and a 256-token context window.
- Training is limited to a narrow transcript-derived corpus; it does not provide broad factual or general-purpose language capabilities.
- Generated output can be incoherent, repetitive, structurally incorrect or unrelated to the prompt.
- The model may reproduce memorized portions of its training material.
- Dialogue mode is a prompt-format convenience, not persistent character understanding or a reliable conversational agent.
- GPU compatibility depends on the installed PyTorch build and host environment.
- Training checkpoints and tokenizer/data artifacts are generated locally and are not included in the repository.
- Dataset-derived model weights and tokenizer redistribution rights have not been established by the code license.

---

## Repository structure

```text
RENA_RUNE.py                 # Project entry point
config/
  rena_rune.json              # Default model/training configuration
rena_rune/
  cli.py                     # Command-line interface
  config.py                   # Configuration and device helpers
  data.py                     # Token dataset and batch loading
  engine.py                   # Training loop and checkpoint handling
  generate.py                  # Sampling and inference checkpoint loading
  model.py                    # Transformer model implementation
  tokenizer.py                 # Tokenizer and corpus preparation
tools/
  export_safetensors.py         # Inference-package exporter
run_wsl.sh                     # WSL command wrapper
requirements.txt                # Python dependencies
PUBLISHING.md                   # Public release checklist
```

Local `data/`, `runs/`, checkpoints, tokenizer outputs, caches and temporary exports are excluded from the repository.

---

## License and attribution

The source code in this repository is licensed under the Apache License 2.0; see [`LICENSE`](LICENSE). This license applies to the project code, not automatically to third-party material used to train the model.

The transcript dataset is separately attributed to [Deltarunefan on Hugging Face](https://huggingface.co/datasets/Deltarunefan/Deltarune-Complete-Transcript-Cleaned). Deltarune is a work by its respective rights holders. RENA-RUNE does not claim ownership of Deltarune, its characters, dialogue or transcript material.

No transcript files, tokenized corpus, tokenizer artifacts, checkpoints or model weights are included in this source repository. The right to use or redistribute the underlying transcript material, and any dataset-derived tokenizer or model artifacts, is not established by the Apache-2.0 code license. Review the dataset card and applicable rights before using or distributing such materials.
