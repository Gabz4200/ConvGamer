# Stem design: patchify-first downsampler + correction kernel choice

## Problem

The old `LearnedSpatialTemporalDownsampler` ran its correction conv
(`channel_multiple=85`, i.e. 3→255 channels, kernel (7,7,3)) at **full input
resolution before downsampling**:

- ~45 GFLOPs/sample at 224×224×12 (vs ~54 GFLOPs for the whole backbone),
- ~0.6 GB fp32 activations per sample,
- the exact memory-bound large-kernel pattern InceptionNeXt was designed to
  avoid (arXiv:2303.16900).

The 7×7 kernel was never the problem (~37k params); the 85× expansion at full
resolution was.

## Redesign

Downsample FIRST with a strided patchify conv `(1,P,P)` (temporal extent 1, so
stateless and trivially streaming-safe), then a small causal `(3,K,K)`
correction at low resolution with a residual add. Aspect ratio is preserved
(256×144 → 32×18); no forced square, no bilinear resize.

## Measurements (CPU, 1×3×12×224×224, 5-run mean)

| patch | kernel | out | params | fwd |
|---|---|---|---|---|
| 4 | dw7 (7×7 depthwise) | 56×56 | 11k | 82 ms |
| 4 | dense4 (4×4 dense) | 56×56 | 22k | 35 ms |
| 8 | dw7 (7×7 depthwise) | 28×28 | 12k | 12 ms |
| 8 | dense4 (4×4 dense) | 28×28 | 22k | 11 ms |

Depthwise-7×7 is 2–7× slower than dense-4×4 despite fewer params — the
InceptionNeXt memory-access argument reproducing inside our own stem.

## Decisions

- **Correction kernel: `dense4` default.** ~3× fewer MACs (~33M vs ~100M at
  patch 8) and 1–2.4× faster measured. `dw7` stays available as the
  large-receptive-field alternative.
- **Expansion 12** (midpoint of the approved 8–16 range, down from 85).
- **Patch 8 default** (plan); patch 4 available when denser maps are worth the
  3–7× stem cost. At 224px, patch 8 → 28×28 stem output → 7×7 frame maps.
- **Resolution:** native game resolution preferred; otherwise a 16:9 multiple
  (256×144 class), never a forced square. Data-default change lands with the
  Phase E data audit.

## Pending (needs GPU + real data)

JEPA-val tiebreak between `dense4` and `dw7`: run
`uv run train model=jepa data=jepa trainer=jepa optimizer=jepa` with
`model.encoder.correction_kernel=dense4` vs `=dw7` (same seed, short schedule)
and keep the higher val score. Throughput evidence above already favors
`dense4`; only a clear quality win should overturn it.
