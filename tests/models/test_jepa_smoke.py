"""Smoke integration: fixture datamodule batches train the JEPA stack.

Nothing else in the suite exercises the datamodule -> module path; the
train scripts do, but outside pytest. This pins it: real fixture batches
through ``training_step`` with finite loss and gradients.
"""

import torch

from convgamer.data.jepa_datamodule import VJEPAGamingDataModule
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor
from convgamer.modules.jepa_module import ConvGamerVJEPAModel


def test_fixture_batches_train_jepa_end_to_end() -> None:
    """Two fixture batches through training_step: finite loss, healthy grads."""
    dm = VJEPAGamingDataModule(
        batch_size=2,
        num_workers=0,
        num_frames=8,
        height=32,
        width=32,
        mode="fixture",
    )
    dm.setup()
    encoder = ConvGamerEncoder(
        input_dim=3, hidden_dim=8, num_layers=1, level_dim=16, temporal_dilations=(1,)
    )
    model = ConvGamerVJEPAModel(
        encoder=encoder,
        predictor=VJEPAPredictor(feature_dim=16, predictor_dim=16, num_layers=1),
        loss=JEPALoss(lambda_base=0.5),
        ema_decay=0.99,
    )
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        x, y, mask = batch
        assert x.shape == y.shape == (2, 3, 8, 32, 32)
        loss = model.training_step((x, y, mask), batch_idx=i)
        assert loss.dim() == 0 and torch.isfinite(loss)
        loss.backward()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters()
        )
        model.zero_grad()
