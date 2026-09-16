# ConvGamer

ConvGamer is a PyTorch and PyTorch Lightning research framework implementing [InceptionNeXt](https://arxiv.org/abs/2303.16900) (_When Inception Meets ConvNeXt_), featuring decomposed large-kernel depthwise convolutions, composable Hydra configuration, and accelerated geometry kernels.

## Key Features

- **InceptionNeXt Backbone**: Implements the `InceptionDWConv2d` operator—decomposing large-kernel depthwise convolutions into 3×3 square kernels, 1×k horizontal band kernels, k×1 vertical band kernels, and identity passthrough to achieve ConvNeXt-level accuracy with ResNet-level speed.
- **Dual Execution Backends**: Provides pure PyTorch reference operations alongside accelerated Taichi kernels.
- **Lightning Workflows**: Complete training and evaluation loops powered by PyTorch Lightning with checkpointing, metric logging, and deterministic seeding.
- **Hugging Face Integration**: Custom Hugging Face Transformers model configuration, architecture wrappers, and export tooling (`export-hf`).
- **Hydra Configuration**: Composable YAML configuration groups for models, optimizers, datasets, trainers, and execution backends.

## Installation

### With `uv` (Recommended)

```bash
# CPU setup (default for testing and development)
uv sync --extra cpu

# GPU setup (CUDA 12.4)
uv sync --extra cu124

# Development setup (includes testing and linting tools)
uv sync --extra cpu --extra dev
```

### With `pip`

```bash
# CPU
pip install -e ".[cpu]"

# GPU (CUDA 12.4)
pip install -e ".[cu124]"
```

> [!NOTE]
> ConvGamer configures `cpu` and `cu124` as mutually exclusive environment extras via `uv` conflict resolution. Switching between them with `uv sync --extra <extra>` updates dependencies atomically.

## Quick Start

### Python API

Instantiate and execute the InceptionNeXt encoder directly:

```python
import torch
from convgamer.models.inception_next import InceptionNeXtEncoder

# Initialize Tiny/Small stage layout (96 hidden dims, 4 stages)
model = InceptionNeXtEncoder(
    input_dim=3,
    hidden_dim=96,
    num_layers=[3, 3, 9, 3],
    num_classes=10,
)

x = torch.randn(2, 3, 224, 224)
logits = model(x)
print(f"Output shape: {logits.shape}")  # [2, 10]
```

Models can also be instantiated via the registry:

```python
from convgamer.models.registry import get_model

model = get_model(
    "InceptionNeXtEncoder",
    input_dim=3,
    hidden_dim=96,
    num_layers=[3, 3, 9, 3],
    num_classes=10,
)
```

### Command-Line Training

Train with PyTorch Lightning using Hydra configuration:

```bash
# Full training run with default configuration
uv run train

# Quick single-batch smoke run
uv run train fast_dev_run=true
```

### Evaluation & Model Export

Evaluate checkpoints or export trained weights to Hugging Face Transformers format:

```bash
# Evaluate a trained checkpoint
uv run eval +eval.checkpoint=path/to/checkpoint.ckpt

# Export to Hugging Face format
uv run export-hf --checkpoint path/to/checkpoint.ckpt --output-dir exported_model
```

## Configuration

Experiments are configured using Hydra files in `configs/`. Default parameters can be overridden from the CLI:

```bash
# Select execution backend (reference or taichi)
uv run train ops=taichi

# Override model and training hyperparameters
uv run train model.hidden_dim=48 trainer.max_epochs=50

# Run multi-run sweeps
uv run train --multirun model.hidden_dim=48,96 trainer.max_epochs=10,20
```

Available configuration groups: `model`, `optimizer`, `data`, `trainer`, `ops`, `experiment`, `debug`.

Layout: `models/` holds raw `nn.Module` backbones, `modules/` holds the
Lightning training wrappers around them.

## Development

Run tests, formatting, and type checks:

```bash
# Run test suite
uv run pytest

# Format and lint code
uv run ruff check
uv run ruff format --check

# Type check
uv run pyrefly check
```

## References

If you use InceptionNeXt in your research, please cite the original paper:

```bibtex
@article{yu2023inceptionnext,
  title={InceptionNeXt: When Inception Meets ConvNeXt},
  author={Yu, Weihao and Zhou, Pan and Shuicheng, Yan},
  journal={arXiv preprint arXiv:2303.16900},
  year={2023}
}
```
