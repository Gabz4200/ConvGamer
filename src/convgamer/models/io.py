"""Typed state and output contracts for the causal video backbones.

A forward that returns a single tensor stays a bare ``torch.Tensor``.  These
dataclasses cover the multi-value seams:

- streaming state carried across ``step()`` calls (one dataclass per composite
  module, so adding a field never breaks a caller that only reads what it needs);
- :class:`StepOutput`, the per-frame ``(features, logits)`` pair.

State is *explicit*: every ``step()`` receives the incoming state and returns
the updated state.  No module keeps recurrent state on ``self``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "DownsamplerState",
    "MixerState",
    "StepOutput",
    "StemState",
    "StreamingState",
]


@dataclass
class MixerState:
    """Streaming state of ``CausalTemporalMixer``.

    Attributes
    ----------
    conv:
        One sliding-window tensor per dilation layer, each
        ``(B, C, pt_i, H, W)`` where ``pt_i`` is that layer's temporal
        receptive field minus one.
    aggregator:
        ``MinConvExpLSTM`` hidden state ``(B, Hid, H, W)``.
    """

    conv: tuple[torch.Tensor, ...]
    aggregator: torch.Tensor


@dataclass
class StemState:
    """Streaming state of ``ConvGamerStem`` — one window per causal conv.

    Each window is ``(B, C_in, pt, H, W)`` for that conv's input channels.
    """

    near: torch.Tensor
    local: torch.Tensor
    far: torch.Tensor
    fuse: torch.Tensor


@dataclass
class DownsamplerState:
    """Streaming state of ``LearnedSpatialTemporalDownsampler``."""

    correction: torch.Tensor
    out_correction: torch.Tensor


@dataclass
class StreamingState:
    """Full streaming state of ``ConvGamerEncoder``.

    Attributes
    ----------
    downsampler, stem, mixer:
        The per-composite states, in pipeline order.  ``stem`` and ``mixer``
        operate at the post-downsample spatial size; ``downsampler`` sees the
        full-resolution input.
    cumsum:
        Running sum of streamed frame features ``(B, F)`` driving the causal
        cumulative-mean pooling; ``None`` before the first frame.
    step_idx:
        Number of frames already streamed.  Needed because the pooling divisor
        is ``step_idx + 1``, not any tensor shape.
    """

    downsampler: DownsamplerState
    stem: StemState
    mixer: MixerState
    cumsum: torch.Tensor | None = None
    step_idx: int = 0


@dataclass
class StepOutput:
    """Output of streaming one frame through ``ConvGamerEncoder``.

    Attributes
    ----------
    features:
        Frame features ``(B, F, 1)``, matching ``forward_features`` at that
        frame index.
    logits:
        Causal cumulative-mean pooled output ``(B, K)`` or ``(B, 1, K)``,
        matching ``forward`` pooled at that frame index.  When no
        classification head is attached this is the pooled/normed feature.
    new_state:
        State advanced by this frame, to be passed to the next :meth:`step`.
    """

    features: torch.Tensor
    logits: torch.Tensor
    new_state: StreamingState
