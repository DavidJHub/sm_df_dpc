"""
Reconstrucción single-shot de atenuación, DPC y dark-field.

Implementa los algoritmos de recuperación del paper (Ecs. 5, 7 y 9 de
Yuan & Das, Optica 12, 1895, 2025) en forma generalizada: en lugar de los
escalares (w_e, α, α₁, α₂, α₃) se usan los coeficientes por píxel
(W_n, A_n, B_n) calculados desde el perfil real de la máscara proyectada
(ver :mod:`sm_df_dpc.mask`), de modo que las fórmulas siguen siendo válidas
para slits, penumbras u offsets arbitrarios. Para el patrón canónico se
reducen exactamente a las ecuaciones del paper.

Modelo por píxel (flat-field Ī_n = I_n^S / I_n^M, con I_n^M ∝ W_n):

    Ī_n = T(1−L) − T·D·A_n/W_n + T·S_n·B_n/W_n

Métodos
-------
* ``method='paper'`` — fórmulas cerradas por diferencias entre píxeles
  vecinos (ventana deslizante para N=2; grupos de 3 para DF-DPC).
* ``method='lstsq'`` — ajuste por mínimos cuadrados del modelo lineal en
  cada grupo de N píxeles (válido también con desalineamiento moderado).

Salidas
-------
* ``transmission``      T(1−L)   (atenuación con realce Laplaciano)
* ``dpc_um``            D_n      desplazamiento del beamlet [µm]
* ``refraction_urad``   D_n / z_eff  ángulo de refracción [µrad]
* ``darkfield``         señal adimensional del paper, ~(α₃/α₁)·S_n ∈ [0, 1]
* ``Sn_um2``            S_n      varianza de ensanchamiento [µm²]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .mask import MaskCoefficients, compute_coefficients

_EPS = 1e-12


@dataclass
class MultiContrastResult:
    """Imágenes multi-contraste reconstruidas (None si no aplica)."""

    geom: Geometry
    method: str
    transmission: np.ndarray
    dpc_um: np.ndarray | None = None
    refraction_urad: np.ndarray | None = None
    darkfield: np.ndarray | None = None
    Sn_um2: np.ndarray | None = None


def _as_2d(I):
    I = np.asarray(I, dtype=float)
    return I[None, :] if I.ndim == 1 else I


def _repeat_groups(vals: np.ndarray, N: int, n_pix: int) -> np.ndarray:
    """Expande valores por grupo (stride N) al ancho completo del detector."""
    full = np.full(vals.shape[:-1] + (n_pix,), np.nan)
    full[..., : vals.shape[-1] * N] = np.repeat(vals, N, axis=-1)
    return full


def _safe_div(num, den):
    return num / np.where(np.abs(den) < _EPS, np.nan, den)


# ===================================================================== #
# Fórmulas cerradas del paper (generalizadas con W, A, B por píxel)
# ===================================================================== #
def _retrieve_paper_n2(IS, IM, coef: MaskCoefficients):
    """Configuraciones DPC y DF (período 2p), ventana deslizante (Ecs. 5 y 7)."""
    geom = coef.geom
    W, A, B = coef.W, coef.A, coef.B
    Ibar = _safe_div(IS, IM)

    s0, s1 = Ibar[..., :-1], Ibar[..., 1:]
    att = np.full(IS.shape, np.nan)
    if geom.config == "DPC":
        att[..., :-1] = 0.5 * (s0 + s1)  # Ec. (5)
    else:
        # Ec. (7): suma ponderada por flujo, cancela el término B_n
        att[..., :-1] = _safe_div(IS[..., :-1] + IS[..., 1:], IM[..., :-1] + IM[..., 1:])

    if geom.config == "DPC":
        # Ec. (5): raw = (Ī_n − Ī_{n+1})/(Ī_n + Ī_{n+1}) = −D·Δ(A/W)/2
        raw = _safe_div(s0 - s1, s0 + s1)
        slope = (A / W)[:-1] - (A / W)[1:]
        D = np.full(IS.shape, np.nan)
        D[..., :-1] = -2.0 * raw / np.where(np.abs(slope) < _EPS, np.nan, slope)
        return att, D, None

    # DF — Ec. (7): df_raw = 1 − (ΔI^S/ΔI^M)·(ΣI^M/ΣI^S) = −S·ΔB/ΔW
    dIS = IS[..., :-1] - IS[..., 1:]
    dIM = IM[..., :-1] - IM[..., 1:]
    sIS = IS[..., :-1] + IS[..., 1:]
    sIM = IM[..., :-1] + IM[..., 1:]
    raw = 1.0 - _safe_div(dIS, dIM) * _safe_div(sIM, sIS)
    dW = W[:-1] - W[1:]
    dB = B[:-1] - B[1:]
    Sn = np.full(IS.shape, np.nan)
    Sn[..., :-1] = -raw * _safe_div(dW, dB)
    df = np.full(IS.shape, np.nan)
    df[..., :-1] = raw
    return att, None, (df, Sn)


def _retrieve_paper_n3(IS, IM, coef: MaskCoefficients):
    """Configuración DF-DPC (período 3p), grupos (n−1, n, n+1) — Ec. (9)."""
    geom = coef.geom
    N, phase = 3, coef.phase
    n_pix = IS.shape[-1]
    n_grp = (n_pix - phase) // N

    sl = slice(phase, phase + n_grp * N)
    gS = IS[..., sl].reshape(IS.shape[:-1] + (n_grp, N))
    gM = IM[..., sl].reshape(IM.shape[:-1] + (n_grp, N))
    gW = coef.W[sl].reshape(n_grp, N)
    gA = coef.A[sl].reshape(n_grp, N)
    gB = coef.B[sl].reshape(n_grp, N)

    sumS = gS.sum(axis=-1)
    sumM = gM.sum(axis=-1)
    att_g = _safe_div(sumS, sumM)

    # DPC desde los dos píxeles brillantes (índices 0 y 1 del grupo)
    raw_d = (_safe_div(gS[..., 0], gM[..., 0]) - _safe_div(gS[..., 1], gM[..., 1])) * _safe_div(
        sumM, sumS
    )
    slope = gA[:, 0] / gW[:, 0] - gA[:, 1] / gW[:, 1]
    D_g = -raw_d / np.where(np.abs(slope) < _EPS, np.nan, slope)

    # Dark-field con la combinación (1, 1, −2) — Ec. (9), tercera línea
    cS = gS[..., 0] + gS[..., 1] - 2.0 * gS[..., 2]
    cM = gM[..., 0] + gM[..., 1] - 2.0 * gM[..., 2]
    raw_s = 1.0 - _safe_div(cS, cM) * _safe_div(sumM, sumS)
    cW = gW[:, 0] + gW[:, 1] - 2.0 * gW[:, 2]
    cB = gB[:, 0] + gB[:, 1] - 2.0 * gB[:, 2]
    Sn_g = -raw_s * _safe_div(cW, cB)

    def expand(v):
        full = np.full(v.shape[:-1] + (n_pix,), np.nan)
        full[..., sl] = np.repeat(v, N, axis=-1)
        return full

    return expand(att_g), expand(D_g), (expand(raw_s), expand(Sn_g))


# ===================================================================== #
# Mínimos cuadrados generalizados
# ===================================================================== #
def _retrieve_lstsq(IS, IM, coef: MaskCoefficients):
    """Ajuste del modelo lineal Ī·W = u₁·W − u₂·A + u₃·B por grupos de N.

    Incógnitas u = (T(1−L), T·D, T·S_n); para N=2 se omite la columna no
    sensada por la configuración (B en DPC, A en DF).
    """
    geom = coef.geom
    N, phase = geom.N, coef.phase
    n_pix = IS.shape[-1]
    n_grp = (n_pix - phase) // N
    sl = slice(phase, phase + n_grp * N)

    flux = _safe_div(IM[..., sl], coef.W[sl])
    b = _safe_div(IS[..., sl], flux).reshape(IS.shape[:-1] + (n_grp, N))

    cols = {"DPC": (0, 1), "DF": (0, 2), "DF-DPC": (0, 1, 2)}[geom.config]
    design = np.stack([coef.W[sl], -coef.A[sl], coef.B[sl]], axis=-1)[:, list(cols)]
    design = design.reshape(n_grp, N, len(cols))
    pinv = np.linalg.pinv(design)  # (n_grp, n_unk, N)

    u = np.einsum("guk,...gk->...gu", pinv, b)
    att_g = u[..., 0]
    D_g = _safe_div(u[..., 1], att_g) if "DF" != geom.config else None
    if geom.config == "DPC":
        Sn_g = None
    else:
        Sn_g = _safe_div(u[..., -1], att_g)

    def expand(v):
        if v is None:
            return None
        full = np.full(v.shape[:-1] + (n_pix,), np.nan)
        full[..., sl] = np.repeat(v, N, axis=-1)
        return full

    df_g = None
    if Sn_g is not None:
        df_g = Sn_g * coef.df_sensitivity
    return expand(att_g), expand(D_g), (expand(df_g), expand(Sn_g)) if Sn_g is not None else None


# ===================================================================== #
# API pública
# ===================================================================== #
def retrieve(
    I_sample: np.ndarray,
    I_mask: np.ndarray,
    geom: Geometry,
    coef: MaskCoefficients | None = None,
    method: str = "paper",
) -> MultiContrastResult:
    """Reconstruye las imágenes multi-contraste desde un único disparo.

    Parameters
    ----------
    I_sample, I_mask : ndarray (..., n_pix)
        Imagen con muestra y referencia solo-máscara (mismo flujo).
    geom : Geometry
    coef : MaskCoefficients, optional
        Coeficientes precalculados; por defecto se calculan de ``geom``.
    method : {'paper', 'lstsq'}
    """
    IS, IM = _as_2d(I_sample), _as_2d(I_mask)
    if IS.shape != IM.shape:
        raise ValueError("I_sample e I_mask deben tener la misma forma")
    if coef is None:
        coef = compute_coefficients(geom, n_pix=IS.shape[-1])
    elif coef.W.shape[-1] != IS.shape[-1]:
        coef = compute_coefficients(geom, n_pix=IS.shape[-1])

    if method == "paper":
        if geom.N == 2:
            att, D, df_pack = _retrieve_paper_n2(IS, IM, coef)
        else:
            att, D, df_pack = _retrieve_paper_n3(IS, IM, coef)
    elif method == "lstsq":
        att, D, df_pack = _retrieve_lstsq(IS, IM, coef)
    else:
        raise ValueError("method debe ser 'paper' o 'lstsq'")

    refraction = None
    if D is not None:
        refraction = D / geom.z_eff_um * 1e6  # µm / µm → rad → µrad
    df = Sn = None
    if df_pack is not None:
        df, Sn = df_pack

    squeeze = np.asarray(I_sample).ndim == 1

    def sq(v):
        return None if v is None else (v[0] if squeeze else v)

    return MultiContrastResult(
        geom=geom,
        method=method,
        transmission=sq(att),
        dpc_um=sq(D),
        refraction_urad=sq(refraction),
        darkfield=sq(df),
        Sn_um2=sq(Sn),
    )
