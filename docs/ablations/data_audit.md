# Data pipeline audit (V-JEPA 2.1 App A alignment + HF verification)

Date: 2026-10-09. Verified with `hf` CLI / `huggingface_hub` API + `datasets`
streaming (1 row) + decord 0.6.0 in this environment.

## 1. Real-video decode was silently broken — fixed

`GameVideoDataset._load_video` and `HFVideoDataset._load_frames` called
`VideoReader.get_batch(...).as_tensor()`. **decord 0.6.0 has no `as_tensor`**
(only `asnumpy`), so every real video raised `AttributeError`, was caught by
the skip-unreadable handler, and **every video pipeline silently yielded zero
batches**. Fixed to `torch.from_numpy(...asnumpy())`. Lesson: the
skip-and-log handler needs a counter that fails loudly when a whole source
yields nothing (left as future work).

## 2. Dataset verification table

| Config ID | Contents | Verdict |
|---|---|---|
| `Infatoshi/terraria-gameplay-256x144-15fps` | 310 direct `.mp4`, 256×144 15fps | KEEP (workhorse; native 16:9) |
| `Xiaokunfeng2022/hollow_knight_dataset` | 73 `.mp4` of 74 files | KEEP |
| `teawhite/hollow-knight-boss-deaths` | 853 `.mp4` | KEEP |
| `markov-ai/gaming-500-hours` | 776 `.mp4` | KEEP |
| `ch1415926/cuphead` | 71 `.mp4` | KEEP |
| `datasetter458/hollow-knight-diverse-dataset` | CSV pointers only, no videos | DROPPED from defaults |
| `DylanRiden/smb-worldmodel-data` | single `smb_frames.zip` (frames, not mp4s) | DROPPED (image-candidate, unverified) |
| `asgaardlab/VideoGameBunny-Dataset` | split-zip parts (`images.z01`…) + `eval_images.zip` | DROPPED (not directly loadable) |
| `VideoGameBunny/Dataset` | 12 files, 1 image | DROPPED |
| `kiyoonkim/kinetics-400-splits` | 13 files, 0 videos; rows are `"<relpath> ..."` strings | DROPPED + `HFVideoDataset` deleted |

`HFVideoDataset` assumed rows carry video bytes/paths; the Kinetics repo
carries relative-path CSV rows with no videos in-repo, so the class could
never yield a batch. Deleted along with `regularization_dataset_ids` —
regularization is now just extra entries in `video_dataset_ids`.

## 3. Masking: multi-block, train-time only

`convgamer/data/masking.py::multiblock_mask_3d` samples unions of random 3D
blocks (spatial sides U(0.15, 0.7), temporal span U(1, T)) to the target
coverage — the V-JEPA 2 multi-block shape at token resolution, replacing the
old iid per-pixel noise. Masks are built in the datasets and never enter the
encoder or the streaming `step()` path (parity tests run unmasked).

## 4. Sampling: 16-frame 16:9 clips

Defaults are now `num_frames=16` (V-JEPA 2.1 App A) at 144×256 (native
terraria aspect; never a forced square). Deviations need VRAM math in the
config comment.

## 5. Modality handling mirrors the paper

V-JEPA 2.1 App A uses modality-specific tokenizers with separate per-modality
workers and aggregated gradients. Ours: `mixed` mode returns separate
video/image loaders (modality-homogeneous batches) paired with
`accumulate_grad_batches=2` in `configs/trainer/jepa.yaml`, so each optimizer
step aggregates one video + one image batch. `mixed` requires non-empty
video AND image IDs and fails loudly otherwise.

## 6. PNG decode segfault: root-caused and fixed

`torchvision.io.read_image` on PNGs segfaulted inside the native decoder late
in long-lived processes. Root cause (bisected 2026-10-09): **merely importing
decord breaks torchvision PNG decode process-wide** — decord's FFmpeg exports
`png_*` symbols that interpose torchvision's bundled libpng (order-dependent:
torchvision-first works, decord-first segfaults on read; JPEG never affected
because `libjpeg` symbols don't collide). Since `dataset.py` imports decord
at module top, every training process was affected.

Fix: `GameImageDataset` decodes with **PIL** (`Image.open` →
`convert("RGB")`), which survives decord-first import (verified). Side
benefit: `convert("RGB")` collapses the grayscale/RGBA channel-fixup
branches. The mixed-mode test is back to a PNG roundtrip and green in the
full-suite context. Production note: if other torchvision.io decoders ever
segfault in-process, suspect the same interposition and prefer PIL.

## 7. Synthetic data: deleted

`RandomSyntheticDataset`, `RandomImageDataset`, `RandomVideoDataset`,
`JEPADataset` (which pre-materialized ~0.5 GB of noise) are deleted. Offline
smoke uses the checked-in real clip (`tests/fixtures/terraria16_u8.pt`,
Terraria 256×144/15fps, 16 frames): `FixtureClipDataset` for JEPA triplets,
`FixtureFrameDataset`/`FixtureVideoDataset` for classification pairs.
`data.mode=synthetic` renamed to `data.mode=fixture`.
