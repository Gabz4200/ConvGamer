# ConvGamer

InceptionNeXt: When Inception Meets ConvNeXt — a CNN research project implementing the model from [arXiv:2303.16900](https://arxiv.org/abs/2303.16900) by Weihao Yu, Pan Zhou, and Shuicheng Yan.

## What is InceptionNeXt?

> *Although ConvNeXt's depthwise convolution has few FLOPs, high memory access costs make it a bottleneck on GPUs. InceptionNeXt decomposes large-kernel depthwise convolution into four parallel branches — small square kernel, two orthogonal band kernels, and an identity mapping — achieving ConvNeXt-level accuracy with ResNet-level speed.* ([Paper](https://arxiv.org/abs/2303.16900))

The core operation (`InceptionDWConv2d`) splits input channels into:
- 1/8 → 3×3 square kernel
- 1/8 → 1×k horizontal band
- 1/8 → k×1 vertical band
- 5/8 → identity passthrough

## Installation

### CPU (default for tests/CI)

```bash
uv sync --extra cpu
# or
pip install -e ".[cpu]"
```

### GPU NVIDIA (CUDA 12.4)

```bash
uv sync --extra cu124
# or
pip install -e ".[cu124]"
```

### Base only (no PyTorch — lint/docs)

```bash
uv sync
```

### Verify

```python
import torch

print(torch.__version__)
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Quick start

```bash
# Install (CPU)
uv sync --extra cpu --extra dev

# Run a fast dev smoke test (1 batch train + 1 batch val)
uv run python scripts/train.py --fast-dev-run true

# Full training
uv run python scripts/train.py
```

## Project structure

```
convgamer/
├── configs/          # Hydra/OmegaConf YAML configs
├── src/convgamer/
│   ├── models/
│   │   ├── inception_next/  # InceptionNeXt backbone (Encoder, blocks)
│   │   └── registry.py      # Generic model registry
│   ├── ops/          # Geometry op abstraction (DIP)
│   ├── kernels/taichi/   # Taichi kernels + runtime isolation
│   ├── integrations/     # PyTorch custom-op + Transformers wrappers
│   ├── data/             # Lightning DataModule + Dataset
│   ├── modules/          # LightningModules (InceptionNeXtModule + ConvGamerModel)
│   ├── callbacks/        # Lightning callbacks
│   ├── training/         # Trainer factory
│   └── scripts/          # Package-level entry points
├── scripts/          # User-facing CLI entry points
├── tests/            # Reference vs impl parity tests
├── notebooks/        # Jupytext-paired quickstart
└── benchmarks/
```

## Switching backends later

```bash
uv sync --extra cu124   # removes cpu extra, adds cu124
```

`uv` handles the transition atomically via the `conflicts` block in `pyproject.toml`.

## License

MIT
