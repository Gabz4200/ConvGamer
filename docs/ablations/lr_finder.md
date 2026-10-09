# LR finder smoke (wiring validation, not a production LR)

`scripts/lr_find_jepa.py` runs Lightning's `Tuner(trainer).lr_find` on a tiny
JEPA stack over the fixture clip (CPU). Note: installed `pytorch_lightning`
2.6.5 exposes this via `pytorch_lightning.tuner.Tuner`, not
`trainer.tuner`.

Smoke result (2026-10-09, seed 42, range 1e-6→1e-2): loss rose from the first
step (0.57 → 0.93 over 4 tested LRs) and the finder stopped with suggestion
`None`. The wiring works (curve produced, no crash); the outcome is
inconclusive — a random-init tiny model diverging immediately says nothing
about the production LR. Re-find on GPU with real data before any full run
and record the curve + chosen LR here.
