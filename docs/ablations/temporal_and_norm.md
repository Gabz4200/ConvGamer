# Temporal + norm ablation: TCN vs MinConv, pre/post/group/none

Setup: tiny JEPA stack (encoder hidden_dim=8, level_dim=16, dilations (1,2),
1-layer predictor) trained 8 AdamW steps on a real Terraria clip fixture
(`tests/fixtures/terraria16_u8.pt`, 16 frames 256×144, seed 0, CPU). The
recorded table below used fixed iid 25% masks; the current harness
(`scripts/bench_temporal.py`) uses multi-block masks, so rerun numbers are
not directly comparable to the recorded rows — only within a rerun. This is
a *relative* comparison for a deletion decision, not a quality claim:
8 steps cannot rank converged JEPA quality.

## Benchmark run (steps=8, seed=0, fixture=terraria16, CPU)

| temporal | norm | final_loss | grad_min | grad_max | train_ms | stream_ms | drift |
|---|---|---|---|---|---|---|---|
| both | prenorm | 1.17964 | 1.05e+00 | 1.16e+00 | 249.6 | 19.57 | 7.45e-09 |
| both | postnorm | 1.17859 | 1.12e+00 | 1.19e+00 | 200.5 | 15.1 | 1.09e-02 |
| both | groupnorm | 1.16084 | 9.58e-01 | 1.13e+00 | 215.9 | 17.66 | 6.07e-03 |
| both | none | 1.10628 | 1.26e+00 | 1.38e+00 | 219.2 | 18.43 | 3.73e-09 |
| tcn-only | prenorm | 1.15515 | 1.22e+00 | 1.39e+00 | 233.1 | 16.1 | 7.45e-09 |
| tcn-only | postnorm | 1.12235 | 1.44e+00 | 1.59e+00 | 182.3 | 13.6 | 1.31e-04 |
| tcn-only | groupnorm | 1.20881 | 1.09e+00 | 1.33e+00 | 244.3 | 21.98 | 9.87e-03 |
| tcn-only | none | 1.12248 | 1.44e+00 | 1.60e+00 | 206.8 | 13.98 | 3.73e-09 |
| minconv-only | prenorm | 1.27683 | 1.40e+00 | 1.49e+00 | 270.1 | 14.52 | 7.45e-09 |

## Decision (2026-10-09)

**Keep TCN-only + prenorm. Deleted: `MinConvExpLSTM` aggregator
(`models/convgamer/minconv.py`, whole file), the `use_aggregator` flag, and
the `aggregator` field of `MixerState`.**

Why:

- **Loss is within noise for TCN variants** (1.12–1.21) while `minconv-only`
  is worst (1.28). The recurrent gates start near-averaging (forget bias 1)
  and need longer schedules to adapt — an 8-step proxy favors fast adapters,
  which counts against MinConv here, but nothing in the table favors it
  either: not loss, not train speed (270 ms, slowest), not streaming latency
  (14.5 ms vs 13.6 ms TCN).
- **TCN-only is cheaper and simpler**: drops 3 conv gate projections per
  mixer, keeps streaming state to fixed conv windows (O(1) in T), keeps
  training fully parallel. The TCN receptive field (15 frames at dilations
  (1,2,4)) already covers the T=12–16 clip regime; MinConv's long-horizon
  memory advantage (MinConv paper, closed-loop forecasting) does not apply to
  short-clip JEPA pretraining.
- **Norm: keep prenorm (status quo).** `none` wins at 8 steps (1.106/1.122),
  but unnormalized stacks winning *early* is a known artifact — norms pay off
  over long schedules. Prenorm has exact streaming parity (drift ~1e-9) and
  healthy grad norms, so it stays until a GPU-length run says otherwise.
- **Drift column caveat**: postnorm/groupnorm drift (1e-4–1e-2) is a harness
  artifact — those wrappers patched `forward` but not `step`. Only the
  prenorm/none drift values (~1e-9) are valid parity evidence.

Overturn conditions: a GPU full-schedule JEPA run where MinConv (or a
non-prenorm) wins val loss with healthy stability reopens this. Rerun
`scripts/bench_temporal.py` with more steps on real data first — it now
covers the norms grid on the surviving TCN mixer.
