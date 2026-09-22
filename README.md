# ConvGamer

A PyTorch and PyTorch Lightning research framework implementing **InceptionNeXt** (arXiv:2303.16900) with a causal video encoder extension for V-JEPA-style self-supervised learning on gaming video data.

---

## Overview

ConvGamer provides two main model families:

| Model | Input | Purpose | Key Paper |
|-------|-------|---------|-----------|
| **InceptionNeXtEncoder** | `(B, C, H, W)` | 2D image classification backbone | [InceptionNeXt](https://arxiv.org/abs/2303.16900) |
| **ConvGamerEncoder** | `(B, C, T, H, W)` | Causal video encoder for V-JEPA pretraining | V-JEPA 2.1 + MinConvLSTM (arXiv:2508.03614) |

The video encoder builds on InceptionNeXt by adding:
- **Learned spatial-temporal downsampling** with residual correction
- **Multi-scale 3D causal stem** (1×1×1, 3×3×3, 7×7×7 parallel branches)
- **Causal temporal mixing** via dilated TCN and MinConvLSTM
- **Streaming inference** with `init_state()` / `step()` for frame-by-frame processing

---

## Installation

```bash
# CPU-only
uv sync --extra cpu

# CUDA 12.4
uv sync --extra cu124
```

> [!NOTE]
> `cpu` and `cu124` are mutually exclusive extras. Switching between them with `uv sync --extra <extra>` updates dependencies atomically.

---

## Quick Start

### Python API

```python
import torch
from convgamer.models.registry import get_model

# InceptionNeXt (image)
model = get_model(
    "InceptionNeXtEncoder", input_dim=3, hidden_dim=96, num_layers=3, num_classes=1000
)
x = torch.randn(2, 3, 224, 224)
logits = model(x)  # (2, 1000)

# ConvGamer (video)
model = get_model(
    "ConvGamerEncoder", input_dim=3, hidden_dim=96, num_layers=[3, 3, 9, 3], num_classes=0
)
video = torch.randn(2, 3, 16, 224, 224)  # (B, C, T, H, W)
logits = model(video)  # (2, num_classes)

# Streaming inference
state = model.init_state(batch_size=2, height=224, width=224)
for t in range(16):
    frame = video[:, :, t]  # (2, 3, 224, 224)
    features, logits_t = model.step(frame, state)
```

### Command-Line Training

```bash
# InceptionNeXt image classification
uv run train model=inception_next data=default trainer=default

# V-JEPA ConvGamer pretraining (all five groups required together)
uv run train experiment=jepa model=jepa data=jepa trainer=jepa optimizer=jepa
```

### Evaluation & Export

```bash
# Evaluate a checkpoint (Hydra config)
uv run eval +eval.checkpoint=/path/to/checkpoint.ckpt

# Export to Hugging Face format (safetensors + config.json)
uv run export-hf --checkpoint /path/to/checkpoint.ckpt --output-dir ./exported_model [--ckpt-kind video|image]
```

> [!NOTE]
> The command writes a local Hugging Face-format model directory (config.json + safetensors).

---

## Configuration

Experiments use Hydra configs in `configs/`. The default config (`configs/config.yaml`) composes:

```yaml
defaults:
  - model: inception_next
  - optimizer: adamw
  - data: default
  - trainer: default
  - experiment: default
```

Override any group from the CLI:

```bash
uv run train model=inception_small trainer.max_epochs=50 optimizer.lr=1e-4
```

**Available groups:** `model`, `optimizer`, `data`, `trainer`, `experiment`, `debug`

---

## Key Features

- **InceptionDWConv2d** — Decomposes large-kernel depthwise convolutions into 3×3 square, 1×k horizontal, k×1 vertical, and identity passthrough (Algorithm 1, arXiv:2303.16900)
- **LayerNorm over BatchNorm** — Numerically stable, works at any batch size
- **Causal by design** — No future frame influences past output; streaming API for real-time inference
- **V-JEPA 2.1 dense loss** — Predict + context terms with distance-weighted λ (Eq. 3, Appendix A)
- **Hugging Face Transformers integration** — Custom `ConvGamerConfig`/`ConvGamerModel` exportable via `export-hf`
- **Oklab color space** — Perceptually uniform sRGB → Oklab conversion for video datasets

---

## Datasets

V-JEPA training uses a tiered dataset strategy configured in `configs/data/jepa.yaml`:

| Tier | Source | Purpose |
|------|--------|---------|
| 1 | Direct MP4 (Hugging Face) | Primary gaming video — Terraria, Hollow Knight, Cuphead, SMB, etc. |
| 2 | Extracted archives | Large datasets requiring download + extract (ignored during JEPA phase) |
| 3 | Static images | T=1 spatial regularizer (VideoGameBunny) |
| 4 | Regularization video | Kinetics-400 / UCF101 |

Synthetic mode (`mode: synthetic`) runs without downloads for smoke tests.

---

## References

If you use InceptionNeXt in your research, please cite the original paper:

```bibtex
@article{yu2023inceptionnext,
  title={InceptionNeXt: When Inception Meets ConvNeXt},
  author={Yu, Weihao and Luo, Mi and Zhou, Pan and Si, Chenyang and Zhou, Yichen and Wang, Xinchao and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2303.16900},
  year={2023}
}
```

For V-JEPA 2.1:

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

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.