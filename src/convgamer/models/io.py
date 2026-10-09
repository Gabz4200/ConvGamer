"""Typed state and output contracts for the causal video backbone.

A forward that returns a single tensor stays a bare ``torch.Tensor``. These
dataclasses cover the streaming seam:

- :class:`StreamCache`, the fixed-size frame-feature history carried across
  ``step()`` calls — O(1) in the sequence length, never a growing list;
- :class:`StepOutput`, the per-frame ``(features, logits)`` pair.

State is *explicit*: every ``step()`` receives the incoming cache and returns
the updated cache. No module keeps recurrent state on ``self``. All temporal
layers are stateless functions of their input clip; streaming works by
re-running them over the fixed-size cache window, so ``step()`` matches
``forward()`` on the same prefix exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "StepOutput",
    "StreamCache",
]


@dataclass
class StreamCache:
    """Fixed-size streaming cache of ``ConvGamerEncoder``.

    One ring window per temporal stage boundary — history is replayed through
    each stage's stateless ``forward`` over its window, so ``step()`` matches
    ``forward()`` on the same prefix exactly:

    Attributes
    ----------
    base:
        Post-patchify base history ``(B, Cb, Kb, H', W')`` feeding the
        downsampler correction conv (``Kb`` = its temporal extent).
    down:
        Downsampled history ``(B, Cd, Ks, H', W')`` feeding the multi-scale
        stem (``Ks`` = its max temporal extent).
    levels:
        Projected level-feature history ``(B, L, P, Km, h, w)`` feeding the
        shared temporal mixer (``Km`` = receptive field minus one).
    count:
        Number of frames already streamed.
    cumsum:
        Running sum of streamed fused frame features ``(B, P, 1)`` driving
        the causal cumulative-mean pooling; ``None`` before the first frame.

    All windows are fixed size: streaming memory is O(1) in T.
    """

    base: torch.Tensor
    down: torch.Tensor
    levels: torch.Tensor
    count: int = 0
    cumsum: torch.Tensor | None = None


@dataclass
class StepOutput:
    """Output of streaming one frame through ``ConvGamerEncoder``.

    Attributes
    ----------
    features:
        Frame features ``(B, P, 1)``, matching ``forward_features`` at that
        frame index.
    logits:
        Causal cumulative-mean pooled output ``(B, K)``, matching ``forward``
        pooled at that frame index. When no classification head is attached
        this is the pooled feature directly.
    new_state:
        Cache advanced by this frame, to be passed to the next :meth:`step`.
    """

    features: torch.Tensor
    logits: torch.Tensor
    new_state: StreamCache
