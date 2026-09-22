# ruff: noqa: E501, E741, N802, N803, SIM300
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["srgb_to_oklab"]

pi = math.pi


def _cbrt(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.pow(torch.abs(x), 1.0 / 3.0)


def _move_channel_last(x: Tensor, dim: int):
    nd = x.ndim
    dim = dim % nd if dim < 0 else dim
    if dim == nd - 1:
        return x, dim, False
    return x.movedim(dim, -1), dim, True


def _move_back(x: Tensor, orig_dim: int, moved: bool) -> Tensor:
    if moved:
        return x.movedim(-1, orig_dim)
    return x


def srgb_transfer_function_inv(x: Tensor) -> Tensor:
    """Non-linear sRGB (0..1) -> linear sRGB."""
    return torch.where(x >= 0.04045, torch.pow((x + 0.055) / 1.055, 2.4), x / 12.92)


def _linear_srgb_to_oklab_elements(r: Tensor, g: Tensor, b: Tensor):
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = _cbrt(l)
    m_ = _cbrt(m)
    s_ = _cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b2 = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, a, b2


def linear_srgb_to_oklab(x: Tensor, dim: int = -3) -> Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    xm, od, moved = _move_channel_last(x, dim)
    r = xm[..., 0]
    g = xm[..., 1]
    b = xm[..., 2]
    L, a, b2 = _linear_srgb_to_oklab_elements(r, g, b)
    out = torch.stack([L, a, b2], dim=-1)
    return _move_back(out, od, moved)


def srgb_to_oklab(x: Tensor, dim: int = -3) -> Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    lin = srgb_transfer_function_inv(x)
    return linear_srgb_to_oklab(lin, dim=dim)
