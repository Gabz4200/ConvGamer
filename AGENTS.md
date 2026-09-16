# AGENTS.md

> **Source-of-truth rule:** Verify behavior from implementation (`pyproject.toml`, `configs/`, source files) first; treat `README.md` as descriptive and potentially stale.

## Project Overview

ConvGamer is a PyTorch research framework implementing **InceptionNeXt** (arXiv:2303.16900) with a video encoder extension (ConvGamerEncoder) for JEPA-style self-supervised learning. It uses PyTorch Lightning for training, Hydra for configuration, and includes Hugging Face Transformers integration.

**Key components:**
- `src/convgamer/models/inception_next/` — InceptionNeXtEncoder (2D image backbone)
- `src/convgamer/models/convgamer/` — ConvGamerEncoder (causal video encoder, builds on InceptionNeXt)
- `src/convgamer/models/jepa/` — V-JEPA loss, predictor, EMA wrapper
- `src/convgamer/modules/` — LightningModule wrappers (`InceptionNeXtModule`, `ConvGamerModel`, `ConvGamerVJEPAModel`)
- `src/convgamer/data/` — DataModules: `ConvGamerDataModule` (smoke), `VJEPAGamingDataModule` (V-JEPA); datasets: `HFVideoDataset`, `GameVideoDataset`, `GameImageDataset`, `RandomImageDataset`, `RandomVideoDataset`
- `src/convgamer/integrations/transformers/` — HF Transformers model wrapper
- Model registry in `src/convgamer/models/registry.py` with `@register_model` decorator

## Setup Commands

```bash
# Requirements: Python >=3.13
# Install with uv (recommended) — extras must include cpu/cu124 + dev/notebooks
uv sync --extra cpu --extra dev
uv sync --extra cpu --extra notebooks
uv sync --extra cu124 --extra dev
uv sync --extra cu124 --extra notebooks

# Or with pip (supported)
pip install -e '.[cpu]'
pip install -e '.[cu124]'
```

**Note:** `cpu` and `cu124` extras are mutually exclusive — switching between them with `uv sync --extra <extra>` updates dependencies atomically.

## Development Workflow

```bash
# Train InceptionNeXt image classification
uv run train model=inception_next data=default trainer=default

# Train V-JEPA ConvGamer (correct config group selection)
uv run train experiment=jepa model=jepa data=jepa trainer=jepa optimizer=jepa

# Run evaluation (Hydra, expects cfg.eval.checkpoint)
uv run eval +eval.checkpoint=/path/to/checkpoint.ckpt

# Export to Hugging Face format (argparse CLI)
uv run export-hf --checkpoint /path/to/checkpoint.ckpt --output-dir /path/to/output [--ckpt-kind video|image]
```

**Configuration:** Experiments use Hydra configs in `configs/`. Override groups: `model`, `optimizer`, `data`, `trainer`, `experiment`, `debug`. Default config is `configs/config.yaml`. There is no `ops` group.

## Testing Instructions

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/models/test_encoder.py -v

# Run with coverage
uv run pytest --cov=convgamer --cov-report=term-missing

# Run tests matching pattern
uv run pytest -k "test_encoder" -v
```

**Test structure:**
- `tests/models/` — Model behavior tests (architecture, shapes, forward pass)
- `tests/data/` — Dataset and Oklab behavior tests
- `tests/test_wiring_seams.py` — DataModule and script wiring tests
- `tests/ops/` — Operator tests (currently empty)

**Key test patterns:** Tests use `get_model()` from registry, check shapes against paper specifications (stem 4× downsample, channel doubling per stage, MLP ratios 4/4/4/3, LayerNorm over BatchNorm).

## Code Style

```bash
# Lint (non-mutating)
uv run ruff check .

# Format check (non-mutating)
uv run ruff format --check .

# Type check
uv run pyrefly check

# Recommended verification steps
uv run ruff check . && uv run ruff format --check . && uv run pyrefly check && uv run pytest
```

**Ruff config** (pyproject.toml): line-length=100, target-version=py313, selects E/F/W/I/N/UP/B/C4/SIM, ignores N812/N816/N806.

**Pyrefly config:** ignore-missing-imports = ["torch.*"], module-path = ["src"].

**Naming conventions:** Models use PascalCase (`InceptionNeXtEncoder`, `ConvGamerEncoder`). Blocks use PascalCase (`InceptionNeXtBlock`, `LearnedSpatialTemporalDownsampler`). Config keys use snake_case.

**Import/registration:** Importing `convgamer.models` (or any `convgamer.models.*` submodule such as `registry`) initializes `models/__init__.py` and triggers encoder registration (`register_model()` decorators in submodules are executed at import time).

## Build

```bash
# Build package
uv build
```

**Build system:** `uv_build` backend (pyproject.toml). Package installs as `convgamer` with entry points: `train`, `eval`, `export-hf`.

**HF Transformers integration:** Importing `convgamer.integrations.transformers.auto_mapping` calls `register_auto_classes()` at module load. Callers can either import that module or call the function explicitly (`register_auto_classes()`). After registration, `AutoModel.from_pretrained("convgamer/...")` works.

## Pull Request Guidelines

```bash
# Recommended checks before PR
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest
```

- Recommended checks before merging (not enforced by project CI/pre-commit): `uv run ruff check .`, `uv run pytest`.
- Keep conventional commit messages (`feat:`, `fix:`, etc.). Avoid `Co-Authored-By` trailers.

## Architecture Notes

**Model registry:** All models inherit from `BaseModel` (abstract, requires `forward`). Register with `@register_model("Name")` — duplicate names raise `ValueError`. Retrieve via `get_model("Name", **kwargs)`.

**One-way dependency:** ConvGamer → InceptionNeXt (ConvGamerEncoder imports InceptionNeXtEncoder). InceptionNeXt does NOT depend on ConvGamer.

**Lightning modules:** 
- `InceptionNeXtModule` — trains 2D backbone on (B, C, H, W), inherits from `ClassificationLightningModule`
- `ConvGamerModel` — trains causal video encoder on (B, C, T, H, W) for classification, inherits from `ClassificationLightningModule`
- `ConvGamerVJEPAModel` — trains V-JEPA self-supervised on `(x, y, mask)` triplets, extends `pl.LightningModule` directly

**Causal video encoder:** Input (B, C, T, H, W). Fully causal — no future frame influences past output. The ConvGamerEncoder supports streaming:
- `init_state()` resets all internal module caches and returns initialization metadata
- `step(x_t, state=None)` takes a single frame (B, C, H, W) and returns (features, logits) for that frame only; a supplied state drives cumulative pooling
- `forward(x, return_sequence=False)` encodes the complete 5D clip; default returns the latest cumulative-pooled logits, while `return_sequence=True` returns per-frame logits. Streaming state is used only by `step()`.

**Feature extraction:**
- `forward_feature_map()` — spatial feature map (B, F, H', W') for single images (InceptionNeXtEncoder only)
- `forward_feature_maps()` — spatial feature maps (B, F, T, H', W') for video (ConvGamerEncoder only)
- `forward_features()` — spatially pooled frame features (B, F) or (B, F, T)
- `forward()` — classification logits when a head is attached; pooled features when head is `nn.Identity` (e.g., `num_classes=0`)

**Data pipeline:** `VJEPAGamingDataModule` yields (context, target, mask) triplets for V-JEPA. `ConvGamerDataModule` for smoke tests (image/video). `HFVideoDataset` (in `data/dataset.py`) provides `IterableDataset` over HF video datasets (Kinetics, UCF101). `oklab.py` provides `srgb_to_oklab()` conversion. `GameVideoDataset` (MP4 video) and `GameImageDataset` (static JPEG/PNG images) for local gaming content.

## Common Gotchas

- **PyTorch install:** Install with `uv` extras (`cpu` or `cu124`) or `pip install -e '.[cpu]'`. Do not install Torch separately when using the `cpu`/`cu124` extras; both already include it.
- **Import errors:** Run `uv sync --extra cpu --extra dev` (or `uv sync --extra cu124 --extra dev`) if pyrefly complains about missing test imports.
- **Config overrides:** CLI overrides use dot notation: `model.hidden_dim=128 trainer.max_epochs=10`.
- **V-JEPA training:** Must select all five Hydra groups together: `experiment=jepa model=jepa data=jepa trainer=jepa optimizer=jepa`. Mixing `model=convgamer_tiny` with `data=jepa` fails — `ConvGamerModel` expects (x, y) but `data=jepa` yields (context, target, mask).
- **Eval uses Hydra:** Pass checkpoint via `+eval.checkpoint=...` (not `checkpoint=...`).
- **Export-hf uses argparse:** Pass `--checkpoint` and `--output-dir` explicitly, plus optional `--ckpt-kind video|image`.
- **Model registry duplicate:** `@register_model` raises `ValueError` on duplicate name, not `KeyError`.