"""Temporal-norm ablation: prenorm vs postnorm vs groupnorm vs none.

Runs tiny end-to-end JEPA training on a real game clip fixture
(tests/fixtures/terraria16_u8.pt) and reports, per variant: final JEPA val
loss, gradient-norm range (stability), train step time, streaming step
latency, and long-horizon streaming drift. Results are appended as a markdown
table to docs/ablations/temporal_and_norm.md.

The TCN-vs-MinConv temporal comparison ran before the MinConv branch was
deleted; its table is preserved in docs/ablations/temporal_and_norm.md.

Usage:
    .venv/bin/python scripts/bench_temporal.py [--steps N] [--seed S]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from convgamer.data.masking import multiblock_mask_3d
from convgamer.models.convgamer.causal import CausalTemporalMixer
from convgamer.models.convgamer.encoder import ConvGamerEncoder
from convgamer.models.jepa import JEPALoss, VJEPAPredictor

FIXTURE = Path("tests/fixtures/terraria16_u8.pt")
DOC = Path("docs/ablations/temporal_and_norm.md")


def _postnorm_forward(mixer: CausalTemporalMixer, x: torch.Tensor) -> torch.Tensor:
    for conv in mixer.layers:
        x = x + mixer.act(conv(x))
    y = x.permute(0, 2, 3, 4, 1)
    y = mixer._post(y)  # type: ignore[attr-defined]
    return y.permute(0, 4, 1, 2, 3)


def _norm_for(mixer: CausalTemporalMixer, kind: str, channels: int) -> None:
    """Swap the TCN pre-norms for the ablation variant under test."""
    if kind == "prenorm":
        return
    if kind == "postnorm":
        mixer.norms = nn.ModuleList([nn.Identity() for _ in mixer.norms])  # type: ignore[assignment]
        mixer._post = nn.LayerNorm(channels)  # type: ignore[attr-defined]
        mixer.forward = lambda x: _postnorm_forward(mixer, x)  # type: ignore[method-assign]
    elif kind == "groupnorm":
        mixer.norms = nn.ModuleList([nn.GroupNorm(4, channels) for _ in mixer.norms])  # type: ignore[assignment]
    elif kind == "none":
        mixer.norms = nn.ModuleList([nn.Identity() for _ in mixer.norms])  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown norm {kind!r}")


def _build(norm: str, seed: int):
    torch.manual_seed(seed)
    enc = ConvGamerEncoder(
        input_dim=3,
        hidden_dim=8,
        num_layers=1,
        patch_size=8,
        expansion=4,
        level_dim=16,
        temporal_dilations=(1, 2),
    )
    _norm_for(enc.temporal_mix, norm, 16)
    pred = VJEPAPredictor(feature_dim=16, predictor_dim=16, num_layers=1, num_outputs=4)
    loss_fn = JEPALoss(lambda_base=0.5)
    return enc, pred, loss_fn


def _clips() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = torch.load(FIXTURE, weights_only=True).float() / 255.0  # (3,16,144,256)
    x = torch.stack([raw, raw.flip(-1)])  # (2,3,16,144,256)
    mask = torch.stack([multiblock_mask_3d(16, 144, 256, 0.25, seed=s) for s in (7, 8)])
    xc = x.clone()
    xc[mask.unsqueeze(1).expand(-1, 3, -1, -1, -1)] = 0.0
    return xc, x, mask


def _run_variant(norm: str, steps: int, seed: int) -> dict:
    enc, pred, loss_fn = _build(norm, seed)
    enc.train()
    pred.train()
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()), lr=5e-4, weight_decay=0.01
    )
    xc, x, mask = _clips()
    grad_norms: list[float] = []
    step_times: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        t0 = time.perf_counter()
        x_pyr = enc.forward_feature_maps(xc)
        p = pred(enc.fuse_pyramid(x_pyr), mask)
        with torch.no_grad():
            y_pyr = enc.forward_feature_maps(x)
        loss = loss_fn(p, y_pyr, mask)
        loss.backward()
        step_times.append(time.perf_counter() - t0)
        total = sum(
            p.grad.abs().sum().item() ** 2
            for p in list(enc.parameters()) + list(pred.parameters())
            if p.grad is not None
        )
        grad_norms.append(total**0.5)
        opt.step()
    with torch.no_grad():
        final = loss_fn(
            pred(enc.fuse_pyramid(enc.forward_feature_maps(xc)), mask),
            enc.forward_feature_maps(x),
            mask,
        ).item()
    # Streaming: latency per frame + long-horizon drift over 64 frames.
    enc.eval()
    long_clip = x[:1].repeat(1, 1, 4, 1, 1)  # (1,3,64,144,256)
    t0 = time.perf_counter()
    with torch.no_grad():
        state = enc.init_state(1, 144, 256)
        outs = []
        for t in range(long_clip.shape[2]):
            out = enc.step(long_clip[:, :, t], state)
            outs.append(out.features)
            state = out.new_state
        rolled = torch.cat(outs, dim=2)
        ref = enc.forward_features(long_clip)
    step_ms = (time.perf_counter() - t0) / long_clip.shape[2] * 1000
    drift = (rolled - ref).abs().max().item()
    return {
        "norm": norm,
        "final_loss": round(final, 5),
        "grad_min": f"{min(grad_norms):.2e}",
        "grad_max": f"{max(grad_norms):.2e}",
        "step_ms": round(sum(step_times) / len(step_times) * 1000, 1),
        "stream_ms": round(step_ms, 2),
        "drift": f"{drift:.2e}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for norm in ("prenorm", "postnorm", "groupnorm", "none"):
        print(f"running norm={norm} ...", flush=True)
        rows.append(_run_variant(norm, args.steps, args.seed))

    header = "| norm | final_loss | grad_min | grad_max | train_ms | stream_ms | drift |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep] + [
        f"| {r['norm']} | {r['final_loss']} | {r['grad_min']} | "
        f"{r['grad_max']} | {r['step_ms']} | {r['stream_ms']} | {r['drift']} |"
        for r in rows
    ]
    table = (
        f"\n## Norm benchmark run (steps={args.steps}, seed={args.seed}, "
        f"fixture=terraria16, CPU)\n\n" + "\n".join(lines) + "\n"
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    with open(DOC, "a") as f:
        f.write(table)
    print(table)


if __name__ == "__main__":
    main()
