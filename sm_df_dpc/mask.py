"""
Perfil de transmisión de la máscara y coeficientes del modelo de transporte.

La máscara (período p_mask, apertura slit, transmisión residual del oro) se
evalúa en coordenadas del plano muestra. La penumbra de la fuente extensa se
modela como una convolución gaussiana de sigma ``geom.sigma_pen_um``.

Los coeficientes por píxel del modelo de transporte (Ec. 3 del paper),

    I_n = T_n [ (1−L_n)·W_n − D_n·A_n + S_n·B_n ],

con
    W_n = (1/p) ∫_pix m(x) dx          (transparencia efectiva)
    A_n = (1/p) ∫_pix m'(x) dx         (sensibilidad DPC)   [1/µm]
    B_n = (1/p) ∫_pix m''(x) dx        (sensibilidad DF)    [1/µm²]

se calculan numéricamente desde el perfil proyectado, lo que generaliza las
constantes (w_e, α, α₁, α₂, α₃) del paper a cualquier geometría, slit,
penumbra o desalineamiento.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import Geometry


def fine_grid(geom: Geometry, n_pix: int, oversample: int) -> np.ndarray:
    """Centros de las muestras finas (plano muestra) para n_pix píxeles."""
    dx = geom.p_eff_um / oversample
    return (np.arange(n_pix * oversample) + 0.5) * dx


def mask_profile(
    geom: Geometry,
    x_um: np.ndarray,
    blur: bool = True,
) -> np.ndarray:
    """Transmisión de la máscara m(x) en el plano muestra.

    Aperturas de ancho ``slit_um`` centradas en
    ``mask_offset + j·p_mask``; el resto transmite ``mask_transmission``.
    Incluye anti-aliasing sub-muestra y, si ``blur``, la penumbra del foco.
    """
    x = np.asarray(x_um, dtype=float)
    dx = float(np.median(np.diff(x))) if x.size > 1 else geom.p_eff_um
    phase = np.mod(x - geom.mask_offset_resolved_um + geom.p_mask_um / 2, geom.p_mask_um)
    phase -= geom.p_mask_um / 2  # distancia (con signo) al centro de apertura
    # Cobertura fraccional de la apertura en cada muestra (anti-aliasing lineal)
    coverage = np.clip((geom.slit_um / 2 - np.abs(phase)) / dx + 0.5, 0.0, 1.0)
    m = geom.mask_transmission + (1.0 - geom.mask_transmission) * coverage
    if blur and geom.sigma_pen_um > 0:
        m = gaussian_filter1d(m, geom.sigma_pen_um / dx, mode="wrap")
    return m


@dataclass(frozen=True)
class MaskCoefficients:
    """Coeficientes del modelo de transporte para una geometría dada.

    Arrays por píxel (longitud n_pix): W, A, B. Escalares del paper según
    la configuración (los no aplicables quedan en nan).
    """

    geom: Geometry
    W: np.ndarray
    A: np.ndarray
    B: np.ndarray
    we: float
    alpha: float = np.nan    # DPC   : A_n = α(−1)ⁿ
    alpha1: float = np.nan   # DF / DF-DPC : modulación de W
    alpha2: float = np.nan   # DF-DPC: sensibilidad DPC
    alpha3: float = np.nan   # DF / DF-DPC : sensibilidad dark-field
    phase: int = 0           # corrimiento del patrón respecto al píxel 0

    @property
    def dpc_sensitivity(self) -> float:
        """Sensibilidad DPC del sistema (señal por µm de desplazamiento)."""
        if self.geom.config == "DPC":
            return abs(self.alpha) / self.we
        if self.geom.config == "DF-DPC":
            return abs(2 * self.alpha2 / (self.we + 0.5 * self.alpha1))
        return 0.0

    @property
    def df_sensitivity(self) -> float:
        """Sensibilidad dark-field (señal por µm² de varianza S_n)."""
        if self.geom.config in ("DF", "DF-DPC"):
            return abs(self.alpha3 / self.alpha1)
        return 0.0


def compute_coefficients(
    geom: Geometry, n_pix: int = 64, oversample: int = 64
) -> MaskCoefficients:
    """Calcula (W, A, B) por píxel y los escalares (w_e, α…) del paper.

    Integra numéricamente el perfil de máscara proyectado (con penumbra)
    sobre cada píxel. El patrón se asume periódico con período N píxeles
    para extraer los escalares; los arrays completos permiten la
    reconstrucción generalizada incluso con desalineamiento.
    """
    p = geom.p_eff_um
    dx = p / oversample
    x = fine_grid(geom, n_pix, oversample)
    m = mask_profile(geom, x)
    dm = np.gradient(m, dx)

    # W_n: promedio de m sobre el píxel
    W = m.reshape(n_pix, oversample).mean(axis=1)
    # m y m' en las fronteras de píxel x = n·p (interpolación lineal)
    xb = np.arange(n_pix + 1) * p
    m_b = np.interp(xb, x, m)
    dm_b = np.interp(xb, x, dm)
    A = (m_b[1:] - m_b[:-1]) / p
    B = (dm_b[1:] - dm_b[:-1]) / p

    N = geom.N
    we = float(W.mean())
    n_idx = np.arange(n_pix)

    if geom.config == "DPC":
        alpha = float(np.mean(A * (-1.0) ** n_idx))
        return MaskCoefficients(geom, W, A, B, we=we, alpha=alpha, phase=0)

    if geom.config == "DF":
        alpha1 = float(np.mean(W * (-1.0) ** n_idx))
        alpha3 = float(-np.mean(B * (-1.0) ** n_idx))
        # fase: el píxel brillante define n par
        phase = 0 if W[0] >= W[1] else 1
        if phase:
            alpha1, alpha3 = -alpha1, -alpha3
        return MaskCoefficients(
            geom, W, A, B, we=we, alpha1=alpha1, alpha3=alpha3, phase=phase
        )

    # DF-DPC: roles (n−1, n, n+1) = (brillante, brillante, oscuro)
    n_grp = (n_pix // N) * N
    Wg = W[:n_grp].reshape(-1, N).mean(axis=0)
    Ag = A[:n_grp].reshape(-1, N).mean(axis=0)
    Bg = B[:n_grp].reshape(-1, N).mean(axis=0)
    dark = int(np.argmin(Wg))
    phase = (dark - 2) % N  # corrimiento tal que el grupo empiece en n−1
    roll = lambda v: np.roll(v, -phase)
    W0, W1, W2 = roll(Wg)
    A0, A1, _ = roll(Ag)
    B0, B1, B2 = roll(Bg)
    alpha1 = float((W0 + W1 - 2 * W2) / 3)
    alpha2 = float((A1 - A0) / 2)
    alpha3 = float((2 * B2 - B0 - B1) / 3)
    return MaskCoefficients(
        geom, W, A, B, we=we, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, phase=phase
    )
