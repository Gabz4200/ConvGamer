"""LR range test for the JEPA stack (Phase F procedure).

Runs Lightning's `lr_find` on a tiny JEPA model over the fixture clip and
prints the suggested LR. This validates the finder wiring on CPU; the
production LR for a full run must be re-found on GPU with real data and
recorded in docs/ablations/temporal_and_norm.md (or the training log).

Usage:
    .venv/bin/python scripts/lr_find_jepa.py [--min-lr 1e-6] [--max-lr 1e-2]
"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.tuner import Tuner

from convgamer.data.dataset import FixtureClipDataset
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.jepa_module import ConvGamerVJEPAModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--max-lr", type=float, default=1e-2)
    args = ap.parse_args()

    pl.seed_everything(42, workers=True)
    encoder = ConvGamerEncoder(
        input_dim=3,
        hidden_dim=8,
        num_layers=1,
        patch_size=8,
        expansion=4,
        level_dim=16,
        temporal_dilations=(1, 2),
    )
    model = ConvGamerVJEPAModel(
        encoder=encoder,
        predictor=VJEPAPredictor(feature_dim=16, predictor_dim=16, num_layers=1, num_outputs=4),
        loss=JEPALoss(lambda_base=0.5),
        lr=5.25e-4,
    )
    loader = torch.utils.data.DataLoader(
        FixtureClipDataset(num_frames=8, height=64, width=64, repeats=8),
        batch_size=2,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    result = Tuner(trainer).lr_find(
        model, train_dataloaders=loader, min_lr=args.min_lr, max_lr=args.max_lr
    )
    if result is None:
        raise RuntimeError("lr_find returned no result")
    print(f"suggested_lr: {result.suggestion()}")
    print(f"tested_lrs: {[float(v) for v in result.results['lr']]}")
    print(f"losses: {[float(v) for v in result.results['loss']]}")


if __name__ == "__main__":
    main()
