# Color space: oklab vs normalized RGB

Setup: tiny JEPA stack (encoder hidden_dim=8, level_dim=16, 1-layer
predictor), 6 AdamW steps on the Terraria fixture (seed 0, CPU). `oklab` =
sRGB→Oklab conversion in the dataset; `rgb` = ImageNet-standardized sRGB.
Same masks, same init — the only difference is input scaling.

## Color benchmark run (steps=6, seed=0, fixture=terraria16, CPU)

| space | final_loss | train_ms |
|---|---|---|
| oklab | 1.14508 | 125.5 |
| rgb | 1.15061 | 90.8 |

## Decision (2026-10-09)

**Delete the oklab path; train on ImageNet-standardized RGB.**

Why: 6-step loss is tied (Δ=0.006, noise), while oklab adds ~40% data-step
cost and threads a `to_oklab` flag through every dataset, datamodule,
config, and test. The perceptual-uniformity argument for L1-in-Lab is
principled but unmeasured here — and flat-shaded game art (the current
data) is exactly where Lab matters least. Per the evidence-gated rule,
theory alone does not keep 62 lines plus flag plumbing.

Kept: `FixtureClipDataset`/game datasets standardize RGB with ImageNet
mean/std (cheap, standard baseline).

Overturn condition: a GPU run where oklab wins a *quality* metric (linear
probe or JEPA val at convergence, not step-6 loss) restores
`data/oklab.py` from git history with this doc updated.
