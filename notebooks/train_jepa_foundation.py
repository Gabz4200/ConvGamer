# %% [markdown]
# Train ConvGamer V-JEPA foundation model on gameplay videos
#
# ruff: noqa: E402 -- notebook cells import after setup cells by design.
# This notebook installs [ConvGamer](https://github.com/Gabz4200/ConvGamer) from GitHub,
# wires the JEPA stack through the repo composition root
# (`convgamer.scripts.common` + `convgamer.training.engine`), and trains the full
# backbone (`ConvGamerEncoder` + `VJEPAPredictor` + `JEPALoss` in
# `src/convgamer/modules/jepa_module.py`) on real gameplay MP4s.
# The result is a self-supervised foundation encoder with no classification head.
# A later notebook/script attaches an action-prediction head via
# `encoder.add_classification_head(num_actions)`.
#
# Prerequisites: on Kaggle, enable **Internet: ON** (right panel, for the clone
# and the Hugging Face downloads) and pick a **GPU accelerator**
# (P100 or T4 x2; CPU works for the smoke cells only). The full 135k-step run
# exceeds Kaggle's 9h session limit — train in chunks with Save Version outputs,
# or lower MAX_STEPS. All configured video datasets are public, so no HF token
# is needed unless you add a gated one (then store it under Add-ons → Secrets).

# %%
# Clone the repo (skip if already present) and add it to the path.
import os
import sys

if not os.path.isdir("ConvGamer"):
    os.system("git clone --depth 1 https://github.com/Gabz4200/ConvGamer.git")
sys.path.insert(0, os.path.abspath("ConvGamer/src"))
print("repo:", os.path.abspath("ConvGamer"))

# %%
# Install runtime deps (torch/torchvision ship with the Kaggle image, so they
# are intentionally not listed; CPU-only runtimes still run the smoke cells).
# !pip install -q hydra-core omegaconf pytorch-lightning transformers
# !pip install -q huggingface-hub einops decord pillow

# %%
import torch

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Compose the JEPA training config
#
# `model=jepa data=jepa trainer=jepa optimizer=jepa` must be selected together.
# `data.mode=video` trains on `video_dataset_ids` below (`mode=fixture` is the
# offline smoke clip only). Kaggle-sized for 2x T4: the full 5-repo gameplay
# list from `configs/data/jepa.yaml` (~2k clips), batch 4 per GPU
# (effective 4 x 2 GPUs x 2 accum = 16; raise to 8 per GPU for effective 32
# if VRAM allows), `accelerator/devices/strategy` all `auto`
# (→ cuda/2/DDP on the T4 host, cpu/1/single elsewhere).
# For the full run, raise `MAX_STEPS` (paper Appendix A uses 135000).

# %%
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from convgamer.scripts.common import CONFIG_DIR

USE_CUDA = torch.cuda.is_available()
MAX_STEPS = 1000  # demo budget; raise toward 135000 for the real run
# Full gameplay list from configs/data/jepa.yaml (~2k clips across 5 repos).
VIDEO_IDS = list(OmegaConf.load(CONFIG_DIR / "data" / "jepa.yaml")["video_dataset_ids"])

overrides = [
    "model=jepa",
    "data=jepa",
    "trainer=jepa",
    "optimizer=jepa",
    "data.mode=video",
    "trainer.accelerator=auto",
    "trainer.devices=auto",
    "trainer.strategy=auto",
    f"trainer.precision={'16-mixed' if USE_CUDA else '32-true'}",
    f"trainer.max_steps={MAX_STEPS}",
    "trainer.limit_val_batches=10",
    "trainer.val_check_interval=100",
    "data.batch_size=4",
    "data.num_workers=4",
    f"data.video_dataset_ids=[{','.join(VIDEO_IDS)}]",
]

with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
    cfg = compose(config_name="config", overrides=overrides)

print(OmegaConf.to_yaml(cfg.data))
print("video ids:", len(cfg.data.video_dataset_ids))

# %% [markdown]
# ## 2. Pre-download in the main process, then smoke-test one batch
#
# Workers each resolve paths on their own, so download once here first —
# otherwise N workers race N concurrent snapshot downloads. `build_model`
# dispatches on the `convgamer.models.jepa` marker to `ConvGamerVJEPAModel`
# (encoder/predictor/loss from `configs/model/jepa.yaml`).
# Batches are `(context, target, mask)` triplets of standardized RGB.

# %%
from convgamer.data.sources import _resolve_video_paths
from convgamer.scripts.common import build_datamodule, build_model

_resolved = _resolve_video_paths(list(cfg.data.video_dataset_ids), data_dir=str(cfg.data.data_dir))
print("resolved video files:", len(_resolved))

model = build_model(cfg)
datamodule = build_datamodule(cfg)
datamodule.setup("fit")

batch = next(iter(datamodule.train_dataloader()))
x, y, mask = batch
print(
    "x:", tuple(x.shape), x.dtype, "| y:", tuple(y.shape), "| mask:", tuple(mask.shape), mask.dtype
)

with torch.no_grad():
    loss = model(x[:1], y[:1], mask[:1])
print("smoke loss:", float(loss))

# %% [markdown]
# ## 3. Smoke-fit (a few optimizer steps, verifies training loop + EMA)

# %%
import pytorch_lightning as pl

from convgamer.training.engine import create_trainer

pl.seed_everything(42, workers=True)
smoke_cfg = OmegaConf.merge(cfg, {"trainer": {"max_steps": 5, "limit_val_batches": 0}})
smoke_trainer = create_trainer(smoke_cfg)
smoke_trainer.fit(model, datamodule=datamodule)
print("smoke fit ok; ema shadow device:", next(model.ema_encoder.shadow_module.parameters()).device)

# %% [markdown]
# ## 4. LR finder, then the full JEPA pretraining run
#
# Lightning sweeps the LR exponentially, restores the pre-sweep weights, and
# writes the suggestion back to `model.lr` (read by `configure_optimizers`).
# Checkpoints land in `checkpoints/jepa/` (see `configs/trainer/jepa.yaml`).
# On Kaggle, /kaggle/working persists only via Save Version → save outputs.
# Resume with `trainer.fit(model, datamodule=datamodule, ckpt_path="...")`.

# %%
pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
trainer = create_trainer(cfg)

from convgamer.training.engine import maybe_run_lr_finder

cfg.trainer.lr_finder = {
    "enabled": True,
    "min_lr": 1e-6,
    "max_lr": 1e-2,
    "num_training": 50,
    "mode": "exponential",
}
found_lr = maybe_run_lr_finder(trainer, model, datamodule, cfg)
print("training with lr:", model.lr)
trainer.fit(model, datamodule=datamodule)

# %%
# List checkpoints produced by this run.
import glob

for ckpt in sorted(glob.glob("checkpoints/jepa/*.ckpt")):
    print(ckpt)

# %% [markdown]
# ## 5. Next step: action-prediction head (for the follow-up task)
#
# The checkpoint holds a headless foundation encoder. Attach a head later:
# `encoder.add_classification_head(num_actions)` then fine-tune supervised on
# labeled gameplay frames. Skeleton (not run here):

# %%
# from convgamer.modules.jepa_module import ConvGamerVJEPAModel
#
# CKPT = "checkpoints/jepa/last.ckpt"  # or a jepa-convgamer-*.ckpt
# NUM_ACTIONS = 8  # set to your action space
# jepa = ConvGamerVJEPAModel.load_from_checkpoint(CKPT, strict=False)
# encoder = jepa.encoder
# encoder.add_classification_head(NUM_ACTIONS)
# for p in encoder.frame_encoder.parameters():
#     p.requires_grad = False  # optional: linear-probe the head first
# print(type(encoder.head), "| frozen frame encoder ready for action fine-tuning")
