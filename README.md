<div align="center">

# ConvGamer

[![Python](https://img.shields.io/badge/Python-3.13-3c873a?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.3%2B-de2751?style=flat-square)](https://lightning.ai)
[![Hydra](https://img.shields.io/badge/config-Hydra-00b4d8?style=flat-square)](https://hydra.cc)
[![arXiv](https://img.shields.io/badge/arXiv-2303.16900-b31b1b?style=flat-square)](https://arxiv.org/abs/2303.16900)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)

*A PyTorch research framework implementing **InceptionNeXt** with a causal video encoder for V-JEPA-style self-supervised learning on gaming video.*

[Overview](#overview) • [Getting started](#getting-started) • [Usage](#usage) • [Configuration](#configuration) • [Datasets](#datasets) • [Development](#development) • [References](#references)

</div>

ConvGamer combines an image classification backbone with a fully causal video encoder and a V-JEPA 2.1 pretraining stack, wired together with PyTorch Lightning and Hydra. Use it to train InceptionNeXt on images, pretrain the video encoder with self-supervised feature prediction, evaluate checkpoints, and export models to Hugging Face format.

> [!TIP]
> New here? Run the default image experiment first — it needs no downloads and validates the full train loop in minutes.

## Overview

| Model | Input | Purpose | Key paper |
|-------|-------|---------|-----------|
| **InceptionNeXtEncoder** | `(B, C, H, W)` | 2D image classification backbone | [InceptionNeXt](https://arxiv.org/abs/2303.16900) |
| **ConvGamerEncoder** | `(B, C, T, H, W)` | Causal video encoder for V-JEPA pretraining | V-JEPA 2 + MinConvLSTM ([arXiv:2508.03614](https://arxiv.org/abs/2508.03614)) |

The video encoder builds on InceptionNeXt with:

- **Learned spatial-temporal downsampling** with residual correction
- **Multi-scale 3D causal stem** (1×1×1, 3×3×3, 7×7×7 parallel branches)
- **Causal temporal mixing** via dilated TCN and MinConvLSTM
- **Streaming inference** — `init_state()` / `step()` process one frame at a time with explicit state, no future-frame leakage
- **Foundation-model contract** — no classification head by default; attach one later with `add_classification_head()`

## Features

- **InceptionDWConv2d** — large-kernel depthwise convolutions decomposed into 3×3 square, 1×k horizontal, k×1 vertical, and identity branches (Algorithm 1, arXiv:2303.16900)
- **LayerNorm over BatchNorm** — numerically stable at any batch size
- **V-JEPA 2.1 dense loss** — predict + context terms with distance-weighted λ (Eq. 3, Appendix A), multi-level predictor, and EMA target encoder
- **Tiered gaming datasets** — HF video datasets, static images, and Kinetics regularization behind one DataModule, with a synthetic mode for offline smoke tests
- **Oklab color space** — perceptually uniform sRGB → Oklab conversion for video samples
- **Hugging Face export** — `export-hf` writes `config.json` + safetensors loadable via `ConvGamerConfig` / `ConvGamerModel`

## Getting started

### Prerequisites

- [Python](https://www.python.org/downloads/) >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Gabz4200/ConvGamer.git
cd ConvGamer

# CPU-only
uv sync --extra cpu --extra dev

# CUDA 12.4
uv sync --extra cu124 --extra dev
```

Or with pip:

```bash
pip install -e '.[cpu]'
```

> [!NOTE]
> The `cpu` and `cu124` extras are mutually exclusive. Switching between them with `uv sync --extra <extra>` updates dependencies atomically.

## Usage

### Python API

```python
import torch
from convgamer.models.registry import get_model

# InceptionNeXt (image classification)
model = get_model(
    "InceptionNeXtEncoder", input_dim=3, hidden_dim=96, num_layers=3, num_classes=1000
)
logits = model(torch.randn(2, 3, 224, 224))  # (2, 1000)

# ConvGamer (video foundation encoder — no head)
model = get_model("ConvGamerEncoder", input_dim=3, hidden_dim=96, num_layers=(3, 3, 9, 3))
video = torch.randn(2, 3, 16, 224, 224)  # (B, C, T, H, W)
features = model(video)  # (2, F) causally pooled features

# Streaming inference: one frame at a time, explicit state
state = model.init_state(batch_size=2, height=224, width=224)
for t in range(16):
    out = model.step(video[:, :, t], state)  # StepOutput
    state = out.new_state
    # out.features: (2, F, 1), out.logits: pooled feature (no head)
```

### Training

```bash
# InceptionNeXt image classification (default config)
uv run train model=inception_next data=default trainer=default

# V-JEPA ConvGamer pretraining — all four groups must be selected together
uv run train model=jepa data=jepa trainer=jepa optimizer=jepa
```

> [!IMPORTANT]
> V-JEPA runs only when `model`, `data`, `trainer`, and `optimizer` are all set to `jepa`. Mixing e.g. `model=convgamer` with `data=jepa` fails: the classification modules expect `(x, y)` batches, but `data=jepa` yields `(context, target, mask)` triplets.

### Evaluation and export

```bash
# Evaluate a checkpoint (Hydra)
uv run eval +eval.checkpoint=/path/to/checkpoint.ckpt

# Export to Hugging Face format (config.json + safetensors)
uv run export-hf --checkpoint /path/to/checkpoint.ckpt --output-dir ./exported_model --ckpt-kind video
# --ckpt-kind video|image selects the checkpoint format
```

## Configuration

Experiments are composed from Hydra config groups in `configs/`. The default config (`configs/config.yaml`):

```yaml
defaults:
  - model: inception_next
  - optimizer: adamw
  - data: default
  - trainer: default
```

Override any group or value from the CLI:

```bash
uv run train model=inception_small trainer.max_epochs=50 optimizer.lr=1e-4
```

**Config groups:** `model`, `optimizer`, `data`, `trainer`

| Group | Options |
|-------|---------|
| `model` | `inception_next`, `inception_small`, `convgamer`, `convgamer_small`, `jepa` |
| `data` | `default` (synthetic images), `jepa` (gaming video tiers) |
| `trainer` | `default`, `jepa` |
| `optimizer` | `adamw`, `jepa` |

## Datasets

V-JEPA training uses a tiered dataset strategy configured in `configs/data/jepa.yaml`:

| Tier | Source | Purpose |
|------|--------|---------|
| 1 | Direct MP4 (Hugging Face) | Primary gaming video — Terraria, Hollow Knight, Cuphead, SMB, and more |
| 2 | Extracted archives | Large datasets requiring download + extract |
| 3 | Static images | T=1 spatial regularizer (VideoGameBunny) |
| 4 | Regularization video | Kinetics-400 / UCF101 |

> [!NOTE]
> The default mode is `synthetic` — random tensors, no downloads. Set `data.mode=video` or `data.mode=mixed` once you have network access and disk space for the HF datasets.

## Development

```bash
# Project layout
src/convgamer/
  models/          # InceptionNeXt, ConvGamer encoder, V-JEPA predictor/loss
  modules/         # LightningModule wrappers, EMA
  data/            # DataModules, datasets, Oklab conversion
  scripts/         # train / eval / export-hf entrypoints
  integrations/    # Hugging Face Transformers classes
configs/           # Hydra config groups
tests/             # pytest suite (models, data, wiring seams)
```

```bash
uv run pytest                     # full test suite
uv run pytest tests/models -v     # focused run
uv run ruff check .               # lint
uv run ruff format --check .      # format check
uv run pyrefly check              # type check
```

## References

If you use InceptionNeXt, please cite:

```bibtex
@article{yu2023inceptionnext,
  title={InceptionNeXt: When Inception Meets ConvNeXt},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2303.16900},
  year={2023}
}
```

For V-JEPA 2:

```bibtex
@article{bardes2026vjepa2,
  title={V-JEPA 2: Self-Supervised Video Representation Learning with Feature Prediction},
  author={Bardes, Adrien and Ponce, Jean and LeCun, Yann and Assran, Mahmoud},
  journal={arXiv preprint arXiv:2603.14482},
  year={2026}
}
```

For MinConvLSTM:

```bibtex
@article{tsai2025minconvlstm,
  title={MinConvLSTM: Minimal Convolutional LSTM with Parallel Prefix Scan},
  author={Tsai, Yao-Hung Hubert and Li, Yingzhen and Torr, Philip H. S. and Rubinstein, Ilya and Vedaldi, Andrea},
  journal={arXiv preprint arXiv:2508.03614},
  year={2025}
}
```
