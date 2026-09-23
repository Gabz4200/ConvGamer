# AGENTS.md

> **Source-of-truth rule:** Verify behavior from implementation (`pyproject.toml`, `configs/`, source files) first; treat `README.md` as descriptive.

## Project Overview

ConvGamer is a PyTorch research framework implementing **InceptionNeXt** (arXiv:2303.16900) with a video encoder extension (ConvGamerEncoder) for JEPA-style self-supervised learning. It uses PyTorch Lightning for training, Hydra for configuration, and includes Hugging Face Transformers integration.

**Key components:**
- `src/convgamer/models/inception_next/` — InceptionNeXtEncoder (2D image backbone)
- `src/convgamer/models/convgamer/` — ConvGamerEncoder (causal video encoder, builds on InceptionNeXt)
- `src/convgamer/models/jepa/` — V-JEPA loss (`JEPALoss`), predictor (`VJEPAPredictor`), helpers
- `src/convgamer/modules/` — LightningModule wrappers (`InceptionNeXtModule`, `ConvGamerModel`, `ConvGamerVJEPAModel`), EMA bookkeeping (`EMAEncoder`)
- `src/convgamer/callbacks/` — EMA lifecycle hook (`EMAUpdateCallback`)
- `src/convgamer/training/` — trainer factory (`create_trainer`, `_build_callback`)
- `src/convgamer/data/` — DataModules: `ConvGamerDataModule` (smoke), `VJEPAGamingDataModule` (V-JEPA); datasets: `RandomImageDataset`, `RandomVideoDataset`, `JEPADataset`, `HFVideoDataset`, `GameVideoDataset`, `GameImageDataset`; `oklab.py` for color conversion
- `src/convgamer/integrations/transformers/` — HF `ConvGamerConfig` / `ConvGamerModel` (`PretrainedConfig` / `PreTrainedModel`)
- Model registry in `src/convgamer/models/registry.py` with `@register_model` decorator
- Streaming I/O contracts in `src/convgamer/models/io.py` (`StreamingState`, `StepOutput`)

## Setup Commands

```bash
# Requirements: Python >=3.13
# Install with uv (recommended) — pick exactly one torch backend + optional dev
uv sync --extra cpu --extra dev
uv sync --extra cu124 --extra dev

# Or with pip (supported)
pip install -e '.[cpu]'
pip install -e '.[cu124]'
```

**Note:** `cpu` and `cu124` extras are mutually exclusive (`[tool.uv] conflicts`) — switching with `uv sync --extra <extra>` updates dependencies atomically. Both extras already include torch; do not install torch separately.

## Development Workflow

```bash
# Train InceptionNeXt image classification (default config)
uv run train model=inception_next data=default trainer=default

# Train V-JEPA ConvGamer — all four groups must be selected together
uv run train model=jepa data=jepa trainer=jepa optimizer=jepa

# Smoke run (forces max_epochs=1 via cfg.fast_dev_run)
uv run train fast_dev_run=true

# Run evaluation (Hydra; needs cfg.eval.checkpoint)
uv run eval +eval.checkpoint=/path/to/checkpoint.ckpt

# Export to Hugging Face format (argparse CLI)
uv run export-hf --checkpoint /path/to/checkpoint.ckpt --output-dir /path/to/output --ckpt-kind video
# --ckpt-kind is video|image
```

**Configuration:** Hydra configs live in `configs/`. Groups: `model`, `optimizer`, `data`, `trainer`. Default composition is `configs/config.yaml` (`model=inception_next`, `optimizer=adamw`, `data=default`, `trainer=default`, plus `fast_dev_run: false`, `seed: 42`). There is **no** `experiment`, `debug`, or `ops` group.

**Model configs:** `inception_next`, `inception_small`, `convgamer`, `convgamer_small`, `jepa`. The `jepa` model uses `target: convgamer.models.jepa` as a composition marker (not an importable backbone class) — dispatch lives in `src/convgamer/scripts/common.py` (`_JEPA_TARGET`).

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
- `tests/models/` — model behavior (architecture, shapes, forward pass, JEPA loss/predictor, causal/streaming parity)
- `tests/data/` — dataset and Oklab behavior
- `tests/test_wiring_seams.py` — script dispatch, trainer factory, datamodule wiring
- `tests/test_shared_seams.py` — registry and callback seams

**Key test patterns:** Tests use `get_model()` from the registry or Hydra `compose` against real YAMLs, check shapes against paper specifications (stem 4× downsample, channel doubling per stage, MLP ratios 4/4/4/3, LayerNorm over BatchNorm). Test names follow `test_when_<scenario>_then_<outcome>`.

## Code Style

```bash
# Lint (non-mutating)
uv run ruff check .

# Format check (non-mutating)
uv run ruff format --check .

# Type check
uv run pyrefly check

# Recommended verification before delivery
uv run ruff check . && uv run ruff format --check . && uv run pyrefly check && uv run pytest
```

**Ruff config** (pyproject.toml): line-length=100, target-version=py313, selects E/F/W/I/N/UP/B/C4/SIM, ignores N812/N816/N806.

**Pyrefly config:** ignore-missing-imports = ["torch.*"], module-path = ["src"].

**pytest config:** testpaths=["tests"], python_files=["test_*.py"], addopts="-v".

**Naming conventions:** Models/blocks use PascalCase (`InceptionNeXtEncoder`, `InceptionNeXtBlock`, `LearnedSpatialTemporalDownsampler`). Config keys use snake_case. Lightning systems live in `modules/`, pure backbones in `models/`.

**Import/registration:** Importing `convgamer.models` (or `convgamer.models.registry`) runs `models/__init__.py`, which imports the encoders and triggers `@register_model` decorators. Duplicate names raise `ValueError`; unknown names in `get_model` raise `KeyError`.

## Build

```bash
# Build package
uv build
```

**Build system:** `uv_build` backend (pyproject.toml). Package installs as `convgamer` with entry points: `train`, `eval`, `export-hf`.

**HF Transformers integration:** Import `ConvGamerConfig` and `ConvGamerModel` from `convgamer.integrations.transformers`. There is no `auto_mapping` module and no `register_auto_classes()` — do not use `AutoModel.from_pretrained` unless you register auto classes yourself. `export-hf` writes `config.json` + weights via `save_pretrained()`.

## Pull Request Guidelines

```bash
# Recommended checks before PR
uv run ruff check .
uv run ruff format --check .
uv run pyrefly check
uv run pytest
```

- Recommended checks before merging (not enforced by project CI — no `.github/workflows`): `uv run ruff check .`, `uv run pytest`.
- Keep conventional commit messages (`feat:`, `fix:`, etc.). Avoid `Co-Authored-By` trailers.

## Architecture Notes

**Model registry:** Encoders inherit `nn.Module` directly (there is **no** `BaseModel` base class). Register with `@register_model("Name")` — duplicate names raise `ValueError`. Retrieve via `get_model("Name", **kwargs)`.

**One-way dependency:** ConvGamer → InceptionNeXt (ConvGamerEncoder uses InceptionNeXtEncoder as its frame encoder). InceptionNeXt does NOT depend on ConvGamer.

**Composition root:** `src/convgamer/scripts/common.py` — configs describe what to build; `build_model` / `build_datamodule` instantiate. Backbones never see config objects. JEPA is selected by the marker target `convgamer.models.jepa`, not by a backbone class.

**Lightning modules:**
- `InceptionNeXtModule` — 2D backbone on (B, C, H, W), inherits `ClassificationLightningModule`
- `ConvGamerModel` (`modules/lightning_module.py`) — causal video encoder on (B, C, T, H, W) for classification, inherits `ClassificationLightningModule`
- `ConvGamerVJEPAModel` — V-JEPA self-supervised on `(context, target, mask)` triplets, extends `pl.LightningModule` directly

Note: `convgamer.integrations.transformers.ConvGamerModel` is a different class (HF `PreTrainedModel`); alias it when importing both.

**Causal video encoder:** Input (B, C, T, H, W). Fully causal — no future frame influences past output. Streaming contract:
- `init_state(batch_size, height, width, ...)` → fresh `StreamingState` (module keeps no history)
- `step(x_t, state)` → `StepOutput` dataclass with `.features` (B, F, 1), `.logits`, `.new_state`; reassign `state = out.new_state` (not tuple unpacking)
- `forward(x, return_sequence=False)` — full clip; default returns latest causal cumulative-mean pooled output (foundation model: pooled feature (B, F)); `return_sequence=True` returns per-frame outputs. Streaming state is used only by `step()`.

**Foundation model contract:** `ConvGamerEncoder` has no classification head by default (`num_classes: null`). Passing `num_classes > 0` emits a deprecation warning; attach later via `add_classification_head(num_classes)`.

**Feature extraction:**
- `forward_feature_map()` — spatial feature map (B, F, H', W') for single images (InceptionNeXtEncoder)
- `forward_feature_maps()` — spatial feature maps (B, F, T, H', W') for video (ConvGamerEncoder)
- `forward_features()` — pooled frame features (B, F) or (B, F, T)
- `forward()` — classification logits when a head is attached; pooled features when head is `nn.Identity`

**Data pipeline:** `VJEPAGamingDataModule` yields (context, target, mask) triplets; modes `synthetic` (default, offline), `video`, `mixed`. `ConvGamerDataModule` for smoke tests. `HFVideoDataset` is an `IterableDataset` over HF video datasets. `oklab.py` provides `srgb_to_oklab()`.

## Common Gotchas

- **PyTorch install:** Use `uv` extras (`cpu` or `cu124`) or `pip install -e '.[cpu]'`. Do not install torch separately when using these extras.
- **Import errors:** Run `uv sync --extra cpu --extra dev` (or `cu124`) if pyrefly complains about missing test imports.
- **Config overrides:** Dot notation: `model.hidden_dim=128 trainer.max_epochs=10`. Use `+eval.checkpoint=...` for eval (key not in defaults).
- **V-JEPA training:** Select all four groups together: `model=jepa data=jepa trainer=jepa optimizer=jepa`. There is no `experiment` group. Mixing `model=convgamer` with `data=jepa` fails — classification modules expect (x, y) but `data=jepa` yields (context, target, mask) triplets. Dispatch is by marker target `convgamer.models.jepa`.
- **Eval uses Hydra:** Pass checkpoint via `+eval.checkpoint=...` (not `checkpoint=...`).
- **Export-hf uses argparse:** Pass `--checkpoint` and `--output-dir` explicitly, plus `--ckpt-kind video|image`.
- **Model registry duplicate:** `@register_model` raises `ValueError` on duplicate name, not `KeyError`. Unknown `get_model` names raise `KeyError`.
- **Streaming state:** `step()` requires a `StreamingState` argument and returns `StepOutput`; always take `.new_state` for the next frame.
- **No notebooks extra:** `pyproject.toml` has only `cpu`, `cu124`, `dev` extras — do not document a `notebooks` extra.
- **No auto classes:** There is no `auto_mapping` / `register_auto_classes()`; use `convgamer.integrations.transformers.ConvGamerConfig` / `ConvGamerModel` directly.
- **No CI workflows:** `.github/` does not exist; verification is local (ruff, pyrefly, pytest).
