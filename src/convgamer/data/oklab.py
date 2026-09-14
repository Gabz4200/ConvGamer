# ruff: noqa: E501, E741, N802, N803, SIM300
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "f",
    "f_inv",
    "srgb_transfer_function",
    "srgb_transfer_function_inv",
    "linear_srgb_to_oklab",
    "oklab_to_linear_srgb",
    "xyz_to_oklab",
    "oklab_to_xyz",
    "srgb_to_linear_srgb",
    "linear_srgb_to_srgb",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "convert_srgb_oklab",
    "oklab_to_lch",
    "lch_to_oklab",
    "compute_max_saturation",
    "find_cusp",
    "find_gamut_intersection",
    "toe",
    "toe_inv",
    "to_ST",
    "get_ST_mid",
    "get_Cs",
    "okhsv_to_srgb",
    "srgb_to_okhsv",
    "okhsl_to_srgb",
    "srgb_to_okhsl",
    "gamut_clip_preserve_chroma",
    "gamut_clip_project_to_0_5",
    "gamut_clip_project_to_L_cusp",
    "gamut_clip_adaptive_L0_0_5",
    "gamut_clip_adaptive_L0_L_cusp",
    "clamp",
    "sgn",
]

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


def clamp(x: Tensor, lo: float = 0.0, hi: float = 1.0) -> Tensor:  # noqa: A001
    return torch.clamp(x, min=lo, max=hi)


def sgn(x: Tensor) -> Tensor:
    return torch.sign(x)


def srgb_transfer_function(x: Tensor) -> Tensor:
    """Linear sRGB (0..1) -> non-linear sRGB (gamma encode)."""
    return torch.where(
        x >= 0.0031308, 1.055 * torch.pow(torch.clamp(x, min=0), 1 / 2.4) - 0.055, 12.92 * x
    )


def srgb_transfer_function_inv(x: Tensor) -> Tensor:
    """Non-linear sRGB (0..1) -> linear sRGB."""
    return torch.where(x >= 0.04045, torch.pow((x + 0.055) / 1.055, 2.4), x / 12.92)


def srgb_to_linear_srgb(x: Tensor) -> Tensor:
    return srgb_transfer_function_inv(x)


def linear_srgb_to_srgb(x: Tensor) -> Tensor:
    return srgb_transfer_function(x)


def toe(x: Tensor) -> Tensor:
    k1 = 0.206
    k2 = 0.03
    k3 = (1 + k1) / (1 + k2)
    return 0.5 * (k3 * x - k1 + torch.sqrt((k3 * x - k1) * (k3 * x - k1) + 4 * k2 * k3 * x))


def toe_inv(x: Tensor) -> Tensor:
    k1 = 0.206
    k2 = 0.03
    k3 = (1 + k1) / (1 + k2)
    return (x * x + k1 * x) / (k3 * (x + k2))


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


def _oklab_to_linear_srgb_elements(L: Tensor, a: Tensor, b: Tensor):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b2 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return r, g, b2


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


def oklab_to_linear_srgb(x: Tensor, dim: int = -3) -> Tensor:
    xm, od, moved = _move_channel_last(x, dim)
    L = xm[..., 0]
    a = xm[..., 1]
    b = xm[..., 2]
    r, g, b2 = _oklab_to_linear_srgb_elements(L, a, b)
    out = torch.stack([r, g, b2], dim=-1)
    return _move_back(out, od, moved)


def srgb_to_oklab(x: Tensor, dim: int = -3) -> Tensor:
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    lin = srgb_transfer_function_inv(x)
    return linear_srgb_to_oklab(lin, dim=dim)


def oklab_to_srgb(x: Tensor, dim: int = -3) -> Tensor:
    lin = oklab_to_linear_srgb(x, dim=dim)
    return srgb_transfer_function(lin)


def convert_srgb_oklab(x: Tensor, dim: int = -3, *, to_oklab: bool = True) -> Tensor:
    """Direct sRGB <-> Oklab without manual linear step.

    Wrapper over :func:`srgb_to_oklab` and :func:`oklab_to_srgb` so you
    don't call the linear intermediate yourself.

    Args:
        x: Tensor with 3-channel dim `dim` (`... ,3, ...`).
        dim: Channel dimension (default ``-3`` for ``C,H,W`` / ``B,C,H,W`` / video).
        to_oklab: If True ``sRGB -> Oklab``, else ``Oklab -> sRGB``.
    """
    return srgb_to_oklab(x, dim=dim) if to_oklab else oklab_to_srgb(x, dim=dim)


def xyz_to_oklab(x: Tensor, dim: int = -3) -> Tensor:
    xm, od, moved = _move_channel_last(x, dim)
    X = xm[..., 0]
    Y = xm[..., 1]
    Z = xm[..., 2]
    l = 0.8189330101 * X + 0.3618667424 * Y - 0.1288597137 * Z
    m = 0.0329845436 * X + 0.9293118715 * Y + 0.0361456387 * Z
    s = 0.0482003018 * X + 0.2643662691 * Y + 0.6338517070 * Z
    l_ = _cbrt(l)
    m_ = _cbrt(m)
    s_ = _cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    out = torch.stack([L, a, b], dim=-1)
    return _move_back(out, od, moved)


def oklab_to_xyz(x: Tensor, dim: int = -3) -> Tensor:
    xm, od, moved = _move_channel_last(x, dim)
    L = xm[..., 0]
    a = xm[..., 1]
    b = xm[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    X = 1.227013851103521 * l - 0.5577999806518223 * m + 0.2812561490159844 * s
    Y = -0.0405801784232806 * l + 1.1122568696168301 * m - 0.0716766786656012 * s
    Z = -0.0763812845057069 * l - 0.4214819784180127 * m + 1.5861632204407947 * s
    out = torch.stack([X, Y, Z], dim=-1)
    return _move_back(out, od, moved)


def oklab_to_lch(x: Tensor, dim: int = -3) -> Tensor:
    xm, od, moved = _move_channel_last(x, dim)
    L = xm[..., 0]
    a = xm[..., 1]
    b = xm[..., 2]
    C = torch.sqrt(a * a + b * b)
    h = torch.atan2(b, a) / (2 * pi)
    h = torch.where(h < 0, h + 1, h)
    out = torch.stack([L, C, h], dim=-1)
    return _move_back(out, od, moved)


def lch_to_oklab(x: Tensor, dim: int = -3) -> Tensor:
    xm, od, moved = _move_channel_last(x, dim)
    L = xm[..., 0]
    C = xm[..., 1]
    h = xm[..., 2]
    a = C * torch.cos(2 * pi * h)
    b = C * torch.sin(2 * pi * h)
    out = torch.stack([L, a, b], dim=-1)
    return _move_back(out, od, moved)


def compute_max_saturation(a: Tensor, b: Tensor) -> Tensor:
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    if a.dtype != b.dtype:
        b = b.to(a.dtype)
    a, b = torch.broadcast_tensors(a, b)
    cond_r = -1.88170328 * a - 0.80936493 * b > 1
    cond_g = (1.81444104 * a - 1.19445276 * b > 1) & (~cond_r)
    k0 = torch.where(
        cond_r,
        torch.tensor(1.19086277, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(0.73956515, dtype=a.dtype, device=a.device),
            torch.tensor(1.35733652, dtype=a.dtype, device=a.device),
        ),
    )
    k1 = torch.where(
        cond_r,
        torch.tensor(1.76576728, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(-0.45954404, dtype=a.dtype, device=a.device),
            torch.tensor(-0.00915799, dtype=a.dtype, device=a.device),
        ),
    )
    k2 = torch.where(
        cond_r,
        torch.tensor(0.59662641, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(0.08285427, dtype=a.dtype, device=a.device),
            torch.tensor(-1.15130210, dtype=a.dtype, device=a.device),
        ),
    )
    k3 = torch.where(
        cond_r,
        torch.tensor(0.75515197, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(0.12541070, dtype=a.dtype, device=a.device),
            torch.tensor(-0.50559606, dtype=a.dtype, device=a.device),
        ),
    )
    k4 = torch.where(
        cond_r,
        torch.tensor(0.56771245, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(0.14503204, dtype=a.dtype, device=a.device),
            torch.tensor(0.00692167, dtype=a.dtype, device=a.device),
        ),
    )
    wl = torch.where(
        cond_r,
        torch.tensor(4.0767416621, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(-1.2684380046, dtype=a.dtype, device=a.device),
            torch.tensor(-0.0041960863, dtype=a.dtype, device=a.device),
        ),
    )
    wm = torch.where(
        cond_r,
        torch.tensor(-3.3077115913, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(2.6097574011, dtype=a.dtype, device=a.device),
            torch.tensor(-0.7034186147, dtype=a.dtype, device=a.device),
        ),
    )
    ws = torch.where(
        cond_r,
        torch.tensor(0.2309699292, dtype=a.dtype, device=a.device),
        torch.where(
            cond_g,
            torch.tensor(-0.3413193965, dtype=a.dtype, device=a.device),
            torch.tensor(1.7076147010, dtype=a.dtype, device=a.device),
        ),
    )
    S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b
    k_l = 0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.2914855480 * b
    l_ = 1 + S * k_l
    m_ = 1 + S * k_m
    s_ = 1 + S * k_s
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    l_dS = 3 * k_l * l_ * l_
    m_dS = 3 * k_m * m_ * m_
    s_dS = 3 * k_s * s_ * s_
    l_dS2 = 6 * k_l * k_l * l_
    m_dS2 = 6 * k_m * k_m * m_
    s_dS2 = 6 * k_s * k_s * s_
    f = wl * l + wm * m + ws * s
    f1 = wl * l_dS + wm * m_dS + ws * s_dS
    f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2
    denom = f1 * f1 - 0.5 * f * f2
    denom = torch.where(
        torch.abs(denom) < 1e-12, torch.sign(denom) * 1e-12 + (denom == 0).float() * 1e-12, denom
    )
    S = S - f * f1 / denom
    return S


def find_cusp(a: Tensor, b: Tensor):
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    a, b = torch.broadcast_tensors(a, b)
    S_cusp = compute_max_saturation(a, b)
    l_ = 1 + S_cusp * (0.3963377774 * a + 0.2158037573 * b)
    m_ = 1 + S_cusp * (-0.1055613458 * a - 0.0638541728 * b)
    s_ = 1 + S_cusp * (-0.0894841775 * a - 1.2914855480 * b)
    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    max_rgb = torch.maximum(torch.maximum(r, g), bl)
    max_rgb = torch.clamp(max_rgb, min=1e-12)
    L_cusp = _cbrt(1.0 / max_rgb)
    C_cusp = L_cusp * S_cusp
    return L_cusp, C_cusp


def to_ST(L_cusp: Tensor, C_cusp: Tensor):
    S = C_cusp / torch.clamp(L_cusp, min=1e-12)
    T = C_cusp / torch.clamp(1 - L_cusp, min=1e-12)
    return S, T


def find_gamut_intersection(
    a: Tensor,
    b: Tensor,
    L1: Tensor,
    C1: Tensor,
    L0: Tensor,
    L_cusp: Tensor | None = None,
    C_cusp: Tensor | None = None,
) -> Tensor:
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    L1 = torch.as_tensor(L1)
    C1 = torch.as_tensor(C1)
    L0 = torch.as_tensor(L0)
    # handle tuple/list cusp passed as single arg (compat with C++ LC struct)
    if isinstance(L_cusp, (tuple, list)) and C_cusp is None:
        Lc, Cc = L_cusp
        L_cusp, C_cusp = Lc, Cc
    # if cusp passed as stacked tensor [...,2]
    if isinstance(L_cusp, Tensor) and L_cusp.ndim >= 1 and L_cusp.shape[-1] == 2 and C_cusp is None:
        Lc = L_cusp[..., 0]
        Cc = L_cusp[..., 1]
        L_cusp, C_cusp = Lc, Cc
    # broadcast all
    # if cusp not provided, compute
    if L_cusp is None or C_cusp is None:
        Lc, Cc = find_cusp(a, b)
    else:
        Lc = torch.as_tensor(L_cusp)
        Cc = torch.as_tensor(C_cusp)
        # broadcast Lc,Cc with others
    # broadcast
    a, b, L1, C1, L0, Lc, Cc = torch.broadcast_tensors(a, b, L1, C1, L0, Lc, Cc)
    lower = ((L1 - L0) * Cc - (Lc - L0) * C1) <= 0
    # lower half
    denom_low = C1 * Lc + Cc * (L0 - L1)
    denom_low = torch.where(
        torch.abs(denom_low) < 1e-12,
        torch.sign(denom_low) * 1e-12 + (denom_low == 0).float() * 1e-12,
        denom_low,
    )
    t_low = Cc * L0 / denom_low
    denom_up = C1 * (Lc - 1) + Cc * (L0 - L1)
    denom_up = torch.where(
        torch.abs(denom_up) < 1e-12,
        torch.sign(denom_up) * 1e-12 + (denom_up == 0).float() * 1e-12,
        denom_up,
    )
    t_up = Cc * (L0 - 1) / denom_up
    t = torch.where(lower, t_low, t_up)
    # Halley's refinement for upper only
    # compute only where not lower
    need = ~lower
    if torch.any(need):
        dL = L1 - L0
        dC = C1
        k_l = 0.3963377774 * a + 0.2158037573 * b
        k_m = -0.1055613458 * a - 0.0638541728 * b
        k_s = -0.0894841775 * a - 1.2914855480 * b
        l_dt = dL + dC * k_l
        m_dt = dL + dC * k_m
        s_dt = dL + dC * k_s
        L = L0 * (1 - t) + t * L1
        C = t * C1
        l_ = L + C * k_l
        m_ = L + C * k_m
        s_ = L + C * k_s
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        ldt = 3 * l_dt * l_ * l_
        mdt = 3 * m_dt * m_ * m_
        sdt = 3 * s_dt * s_ * s_
        ldt2 = 6 * l_dt * l_dt * l_
        mdt2 = 6 * m_dt * m_dt * m_
        sdt2 = 6 * s_dt * s_dt * s_
        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
        r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
        r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1
        g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
        g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2
        bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1
        b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt
        b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2
        # avoid div zero
        denom_r = r1 * r1 - 0.5 * r * r2
        denom_g = g1 * g1 - 0.5 * g * g2
        denom_b = b1 * b1 - 0.5 * bl * b2
        # small eps
        denom_r = torch.where(
            torch.abs(denom_r) < 1e-12,
            torch.sign(denom_r) * 1e-12 + (denom_r == 0).float() * 1e-12,
            denom_r,
        )
        denom_g = torch.where(
            torch.abs(denom_g) < 1e-12,
            torch.sign(denom_g) * 1e-12 + (denom_g == 0).float() * 1e-12,
            denom_g,
        )
        denom_b = torch.where(
            torch.abs(denom_b) < 1e-12,
            torch.sign(denom_b) * 1e-12 + (denom_b == 0).float() * 1e-12,
            denom_b,
        )
        u_r = r1 / denom_r
        u_g = g1 / denom_g
        u_b = b1 / denom_b
        t_r = -r * u_r
        t_g = -g * u_g
        t_b = -bl * u_b
        t_r = torch.where(u_r >= 0, t_r, torch.full_like(t_r, 1e10))
        t_g = torch.where(u_g >= 0, t_g, torch.full_like(t_g, 1e10))
        t_b = torch.where(u_b >= 0, t_b, torch.full_like(t_b, 1e10))
        t_delta = torch.minimum(t_r, torch.minimum(t_g, t_b))
        # where delta is huge (no valid), keep 0
        t_delta = torch.where(t_delta > 1e9, torch.zeros_like(t_delta), t_delta)
        t_refined = t + t_delta
        t = torch.where(need, t_refined, t)
    return t


def get_ST_mid(a_: Tensor, b_: Tensor):
    a_ = torch.as_tensor(a_)
    b_ = torch.as_tensor(b_)
    a_, b_ = torch.broadcast_tensors(a_, b_)
    S = 0.11516993 + 1.0 / (
        7.44778970
        + 4.15901240 * b_
        + a_
        * (
            -2.19557347
            + 1.75198401 * b_
            + a_
            * (
                -2.13704948
                - 10.02301043 * b_
                + a_ * (-4.24894561 + 5.38770819 * b_ + 4.69891013 * a_)
            )
        )
    )
    T = 0.11239642 + 1.0 / (
        1.61320320
        - 0.68124379 * b_
        + a_
        * (
            0.40370612
            + 0.90148123 * b_
            + a_
            * (
                -0.27087943
                + 0.61223990 * b_
                + a_ * (0.00299215 - 0.45399568 * b_ - 0.14661872 * a_)
            )
        )
    )
    return S, T


def get_Cs(L: Tensor, a_: Tensor, b_: Tensor):
    L = torch.as_tensor(L)
    a_ = torch.as_tensor(a_)
    b_ = torch.as_tensor(b_)
    L, a_, b_ = torch.broadcast_tensors(L, a_, b_)
    Lc, Cc = find_cusp(a_, b_)
    C_max = find_gamut_intersection(a_, b_, L, torch.ones_like(L), L, Lc, Cc)
    S_max, T_max = to_ST(Lc, Cc)
    k = C_max / torch.clamp(torch.minimum(L * S_max, (1 - L) * T_max), min=1e-12)
    S_mid, T_mid = get_ST_mid(a_, b_)
    C_a = L * S_mid
    C_b = (1 - L) * T_mid
    # soft min 4th order
    C_a4 = C_a * C_a * C_a * C_a
    C_b4 = C_b * C_b * C_b * C_b
    # avoid division by zero
    C_a4 = torch.clamp(C_a4, min=1e-12)
    C_b4 = torch.clamp(C_b4, min=1e-12)
    C_mid = 0.9 * k * torch.sqrt(torch.sqrt(1.0 / (1.0 / C_a4 + 1.0 / C_b4)))
    C_a0 = L * 0.4
    C_b0 = (1 - L) * 0.8
    # C0 soft min 2nd order
    C_a02 = C_a0 * C_a0
    C_b02 = C_b0 * C_b0
    C_a02 = torch.clamp(C_a02, min=1e-12)
    C_b02 = torch.clamp(C_b02, min=1e-12)
    C_0 = torch.sqrt(1.0 / (1.0 / C_a02 + 1.0 / C_b02))
    return C_0, C_mid, C_max


def okhsv_to_srgb(hsv: Tensor, dim: int = -3) -> Tensor:
    if hsv.dtype == torch.uint8:
        hsv = hsv.float() / 255.0
    xm, od, moved = _move_channel_last(hsv, dim)
    h = xm[..., 0]
    s = xm[..., 1]
    v = xm[..., 2]
    a_ = torch.cos(2 * pi * h)
    b_ = torch.sin(2 * pi * h)
    Lc, Cc = find_cusp(a_, b_)
    S_max, T_max = to_ST(Lc, Cc)
    S_0 = 0.5
    # avoid division by zero when S_max small
    S_max = torch.clamp(S_max, min=1e-6)
    k = 1 - S_0 / S_max
    denom = S_0 + T_max - T_max * k * s
    denom = torch.clamp(denom, min=1e-12)
    L_v = 1 - s * S_0 / denom
    C_v = s * T_max * S_0 / denom
    L = v * L_v
    C = v * C_v
    L_vt = toe_inv(L_v)
    # handle L_v ==0
    C_vt = torch.where(L_v > 1e-12, C_v * L_vt / L_v, torch.zeros_like(C_v))
    L_new = toe_inv(L)
    # when L small, C scaling may be nan
    C_scaled = torch.where(L > 1e-12, C * L_new / L, torch.zeros_like(C))
    C = C_scaled
    L = L_new
    # scale to fit gamut top
    r_s, g_s, b_s = _oklab_to_linear_srgb_elements(L_vt, a_ * C_vt, b_ * C_vt)
    max_rgb = torch.maximum(torch.maximum(r_s, g_s), torch.maximum(b_s, torch.zeros_like(b_s)))
    max_rgb = torch.clamp(max_rgb, min=1e-12)
    scale_L = _cbrt(1.0 / max_rgb)
    L = L * scale_L
    C = C * scale_L
    r, g, b = _oklab_to_linear_srgb_elements(L, C * a_, C * b_)
    r = srgb_transfer_function(r)
    g = srgb_transfer_function(g)
    b = srgb_transfer_function(b)
    out = torch.stack([r, g, b], dim=-1)
    return _move_back(out, od, moved)


def srgb_to_okhsv(rgb: Tensor, dim: int = -3) -> Tensor:
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    # srgb -> linear -> oklab
    lin = srgb_transfer_function_inv(rgb)
    # we need to avoid double movedim; do direct via helpers
    xm, od, moved = _move_channel_last(lin, dim)
    r = xm[..., 0]
    g = xm[..., 1]
    b = xm[..., 2]
    L_lab, a_lab, b_lab = _linear_srgb_to_oklab_elements(r, g, b)
    C = torch.sqrt(a_lab * a_lab + b_lab * b_lab)
    # avoid div by zero
    eps = 1e-8
    mask = C > eps
    a_ = torch.where(mask, a_lab / torch.clamp(C, min=eps), torch.ones_like(C))
    b_ = torch.where(mask, b_lab / torch.clamp(C, min=eps), torch.zeros_like(C))
    L = L_lab
    h = 0.5 + 0.5 * torch.atan2(-b_lab, -a_lab) / pi
    # normalize h to 0..1
    h = torch.where(h < 0, h + 1, torch.where(h >= 1, h - 1, h))
    h = torch.where(mask, h, torch.zeros_like(h))
    Lc, Cc = find_cusp(a_, b_)
    S_max, T_max = to_ST(Lc, Cc)
    S_max = torch.clamp(S_max, min=1e-6)
    T_max = torch.clamp(T_max, min=1e-6)
    S_0 = 0.5
    k = 1 - S_0 / S_max
    t = T_max / torch.clamp(C + L * T_max, min=1e-12)
    L_v = t * L
    C_v = t * C
    L_vt = toe_inv(L_v)
    C_vt = torch.where(L_v > 1e-12, C_v * L_vt / L_v, torch.zeros_like(C_v))
    r_s, g_s, b_s = _oklab_to_linear_srgb_elements(L_vt, a_ * C_vt, b_ * C_vt)
    max_rgb = torch.maximum(torch.maximum(r_s, g_s), torch.maximum(b_s, torch.zeros_like(b_s)))
    max_rgb = torch.clamp(max_rgb, min=1e-12)
    scale_L = _cbrt(1.0 / max_rgb)
    L = L / torch.clamp(scale_L, min=1e-12)
    C = C / torch.clamp(scale_L, min=1e-12)
    # invert toe scaling
    # C = C * toe(L)/L ; L = toe(L)
    # need where L>eps
    toe_L = toe(L)
    C = torch.where(L > 1e-12, C * toe_L / torch.clamp(L, min=1e-12), torch.zeros_like(C))
    L = toe_L
    v = torch.where(L_v > 1e-12, L / torch.clamp(L_v, min=1e-12), torch.zeros_like(L))
    # s = (S0+Tmax)*C_v / (Tmax*S0 + Tmax*k*C_v)
    denom_s = T_max * S_0 + T_max * k * C_v
    denom_s = torch.clamp(denom_s, min=1e-12)
    s = (S_0 + T_max) * C_v / denom_s
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)
    h = torch.clamp(h, 0, 1)
    out = torch.stack([h, s, v], dim=-1)
    return _move_back(out, od, moved)


def okhsl_to_srgb(hsl: Tensor, dim: int = -3) -> Tensor:
    if hsl.dtype == torch.uint8:
        hsl = hsl.float() / 255.0
    xm, od, moved = _move_channel_last(hsl, dim)
    h = xm[..., 0]
    s = xm[..., 1]
    l = xm[..., 2]
    # handle edge cases l==0 or 1 : return black/white directly per spec
    # we can compute but also handle via where later
    a_ = torch.cos(2 * pi * h)
    b_ = torch.sin(2 * pi * h)
    L = toe_inv(l)
    C_0, C_mid, C_max = get_Cs(L, a_, b_)
    mid = 0.8
    mid_inv = 1.25
    # need to compute C per element
    # branch s < mid
    t_mid = mid_inv * s
    k1_low = mid * C_0
    k2_low = 1 - k1_low / torch.clamp(C_mid, min=1e-12)
    C_low = t_mid * k1_low / torch.clamp(1 - k2_low * t_mid, min=1e-12)
    t_high = (s - mid) / (1 - mid)
    t_high = torch.clamp(t_high, min=0)
    k0_high = C_mid
    k1_high = (1 - mid) * C_mid * C_mid * mid_inv * mid_inv / torch.clamp(C_0, min=1e-12)
    k2_high = 1 - k1_high / torch.clamp(C_max - C_mid, min=1e-12)
    C_high = k0_high + t_high * k1_high / torch.clamp(1 - k2_high * t_high, min=1e-12)
    use_low = s < mid
    C = torch.where(use_low, C_low, C_high)
    # for s very small, ensure C near 0
    # l==0 or 1 edge: if l==1 => white, if l==0 => black (need to handle)
    r, g, b = _oklab_to_linear_srgb_elements(L, C * a_, C * b_)
    r = srgb_transfer_function(r)
    g = srgb_transfer_function(g)
    b = srgb_transfer_function(b)
    out = torch.stack([r, g, b], dim=-1)
    # handle l==1 and l==0 exactly as spec (return 1,1,1 and 0,0,0)
    # we can where l close to 1 or 0
    is_white = l >= 1 - 1e-6
    is_black = l <= 1e-6
    # expand masks to last dim
    out = torch.where(is_white[..., None], torch.ones_like(out), out)
    out = torch.where(is_black[..., None], torch.zeros_like(out), out)
    return _move_back(out, od, moved)


def srgb_to_okhsl(rgb: Tensor, dim: int = -3) -> Tensor:
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    lin = srgb_transfer_function_inv(rgb)
    xm, od, moved = _move_channel_last(lin, dim)
    r = xm[..., 0]
    g = xm[..., 1]
    b = xm[..., 2]
    L_lab, a_lab, b_lab = _linear_srgb_to_oklab_elements(r, g, b)
    C = torch.sqrt(a_lab * a_lab + b_lab * b_lab)
    eps = 1e-8
    mask = C > eps
    a_ = torch.where(mask, a_lab / torch.clamp(C, min=eps), torch.ones_like(C))
    b_ = torch.where(mask, b_lab / torch.clamp(C, min=eps), torch.zeros_like(C))
    L = L_lab
    h = 0.5 + 0.5 * torch.atan2(-b_lab, -a_lab) / pi
    h = torch.where(h < 0, h + 1, torch.where(h >= 1, h - 1, h))
    h = torch.where(mask, h, torch.zeros_like(h))
    C_0, C_mid, C_max = get_Cs(L, a_, b_)
    mid = 0.8
    mid_inv = 1.25
    # inverse interpolation
    # if C < C_mid
    k1_low = mid * C_0
    k2_low = 1 - k1_low / torch.clamp(C_mid, min=1e-12)
    t_low = C / torch.clamp(k1_low + k2_low * C, min=1e-12)
    s_low = t_low * mid
    k0_high = C_mid
    k1_high = (1 - mid) * C_mid * C_mid * mid_inv * mid_inv / torch.clamp(C_0, min=1e-12)
    k2_high = 1 - k1_high / torch.clamp(C_max - C_mid, min=1e-12)
    t_high = (C - k0_high) / torch.clamp(k1_high + k2_high * (C - k0_high), min=1e-12)
    s_high = mid + (1 - mid) * t_high
    s = torch.where(C < C_mid, s_low, s_high)
    s = torch.clamp(s, 0, 1)
    l = toe(L)
    l = torch.clamp(l, 0, 1)
    h = torch.clamp(h, 0, 1)
    out = torch.stack([h, s, l], dim=-1)
    return _move_back(out, od, moved)


def _gamut_clip_linear(rgb: Tensor, dim: int, L0_fn):
    xm, od, moved = _move_channel_last(rgb, dim)
    r = xm[..., 0]
    g = xm[..., 1]
    b = xm[..., 2]
    # check in gamut (0-1)
    in_gamut = (r < 1) & (g < 1) & (b < 1) & (r > 0) & (g > 0) & (b > 0)
    L_lab, a_lab, b_lab = _linear_srgb_to_oklab_elements(r, g, b)
    C = torch.sqrt(a_lab * a_lab + b_lab * b_lab)
    eps = 1e-5
    C = torch.maximum(C, torch.full_like(C, eps))
    # normalize for hue even when C small? Use eps version for a_,b_ but keep original C for clipping
    a_ = a_lab / C
    b_ = b_lab / C
    # need to recompute C eps for direction? already
    L = L_lab
    L0 = L0_fn(L, C, a_, b_)
    t = find_gamut_intersection(a_, b_, L, C, L0)
    L_clipped = L0 * (1 - t) + t * L
    C_clipped = t * C
    r_c, g_c, b_c = _oklab_to_linear_srgb_elements(L_clipped, C_clipped * a_, C_clipped * b_)
    out_c = torch.stack([r_c, g_c, b_c], dim=-1)
    out_orig = torch.stack([r, g, b], dim=-1)
    out = torch.where(in_gamut[..., None], out_orig, out_c)
    return _move_back(out, od, moved)


def gamut_clip_preserve_chroma(rgb: Tensor, dim: int = -3) -> Tensor:
    def L0_fn(L, _C, _a, _b):  # noqa: N803
        return torch.clamp(L, 0, 1)

    return _gamut_clip_linear(rgb, dim, L0_fn)


def gamut_clip_project_to_0_5(rgb: Tensor, dim: int = -3) -> Tensor:
    def L0_fn(L, _C, _a, _b):  # noqa: N803
        return torch.full_like(L, 0.5)

    return _gamut_clip_linear(rgb, dim, L0_fn)


def gamut_clip_project_to_L_cusp(rgb: Tensor, dim: int = -3) -> Tensor:
    def L0_fn(_L, _C, a_, b_):  # noqa: N803
        Lc, _ = find_cusp(a_, b_)
        return Lc

    return _gamut_clip_linear(rgb, dim, L0_fn)


def gamut_clip_adaptive_L0_0_5(rgb: Tensor, dim: int = -3, alpha: float = 0.05) -> Tensor:
    def L0_fn(L, C, _a, _b):  # noqa: N803
        Ld = L - 0.5
        e1 = 0.5 + torch.abs(Ld) + alpha * C
        inner = e1 * e1 - 2 * torch.abs(Ld)
        inner = torch.clamp(inner, min=0)
        return 0.5 * (1 + torch.sign(Ld) * (e1 - torch.sqrt(inner)))

    return _gamut_clip_linear(rgb, dim, L0_fn)


def gamut_clip_adaptive_L0_L_cusp(rgb: Tensor, dim: int = -3, alpha: float = 0.05) -> Tensor:
    def L0_fn(L, C, a_, b_):  # noqa: N803
        Lc, _ = find_cusp(a_, b_)
        Ld = L - Lc
        k = 2 * torch.where(Ld > 0, 1 - Lc, Lc)
        k = torch.clamp(k, min=1e-12)
        e1 = 0.5 * k + torch.abs(Ld) + alpha * C / k
        inner = e1 * e1 - 2 * k * torch.abs(Ld)
        inner = torch.clamp(inner, min=0)
        return Lc + 0.5 * (torch.sign(Ld) * (e1 - torch.sqrt(inner)))

    return _gamut_clip_linear(rgb, dim, L0_fn)


f = srgb_transfer_function
f_inv = srgb_transfer_function_inv


# torchvision-compatible nn.Module wrappers
class SRGBToOklab(torch.nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return srgb_to_oklab(x, dim=self.dim)


class OklabToSRGB(torch.nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return oklab_to_srgb(x, dim=self.dim)


class LinearSRGBToOklab(torch.nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return linear_srgb_to_oklab(x, dim=self.dim)


class OklabToLinearSRGB(torch.nn.Module):
    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return oklab_to_linear_srgb(x, dim=self.dim)
