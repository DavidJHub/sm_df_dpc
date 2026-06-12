"""
Simulador de adquisición single-mask: genera las imágenes de detector con
muestra (I^S) y de referencia solo-máscara (I^M).

Pipeline (coordenadas del plano muestra, teorema de escalado de Fresnel):

1. Grilla fina de ``oversample`` muestras por píxel proyectado p_eff.
2. Mapas del fantoma T(x,y), φ(x,y), σ_θ²(x,y) y perfil de máscara m(x).
3. Transporte hasta el detector con z_eff:
   * ``model='fokker_planck'`` — Ec. (2) del paper:
       I_z = I₀ − (z_eff/k)·∂x(I₀ ∂xφ) + ∂x²(S_n·I₀),
     con I₀ = T·m y S_n(x) = σ_θ²·z_eff²/2 [µm²].
   * ``model='ray'`` — transporte geométrico (siempre positivo): cada
     muestra se desplaza u = z_eff·∂xφ/k y se ensancha con σ² = 2·S_n.
4. Convolución con la penumbra de la fuente (gaussiana, FWHM
   f_s·(M−1)/M en el plano muestra).
5. Integración sobre los píxeles, escala a fotones y ruido de Poisson.

Las verdades de terreno por píxel (T, D [µm], S_n [µm²], L) se devuelven
para validar la reconstrucción.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import Geometry
from .mask import fine_grid, mask_profile
from .phantom import Phantom


@dataclass
class SimConfig:
    """Parámetros de la adquisición simulada."""

    n_pix_x: int = 512          # píxeles del detector en x (modulación)
    n_rows: int = 32            # filas del detector
    oversample: int = 32        # muestras finas por píxel
    photons_per_pixel: float = 1000.0  # fotones/píxel con haz abierto
    add_noise: bool = True
    model: str = "fokker_planck"  # 'fokker_planck' | 'ray'
    seed: int | None = 0
    return_fine: bool = False   # guardar también los perfiles finos


@dataclass
class SimResult:
    """Resultado de la simulación."""

    geom: Geometry
    config: SimConfig
    I_sample: np.ndarray        # (n_rows, n_pix_x) imagen con muestra
    I_mask: np.ndarray          # (n_rows, n_pix_x) referencia solo-máscara
    x_pix_um: np.ndarray        # centros de píxel en el plano muestra
    gt: dict = field(default_factory=dict)  # verdades de terreno por píxel
    fine: dict = field(default_factory=dict)


def _pixel_average(arr_fine: np.ndarray, oversample: int) -> np.ndarray:
    """Promedio por píxel a lo largo del último eje."""
    shp = arr_fine.shape[:-1] + (arr_fine.shape[-1] // oversample, oversample)
    return arr_fine.reshape(shp).mean(axis=-1)


def _fokker_planck(I0, phi, Sn_map, z_eff, k, dx):
    """Transporte Fokker–Planck (Ec. 2 del paper) en forma conservativa.

    Nota: el campo fino linealizado puede ser localmente negativo en bordes
    abruptos; el modelo del paper está definido sobre las integrales por
    píxel, que permanecen válidas. No se recorta aquí (solo al nivel de
    píxel, antes del ruido de Poisson).
    """
    dphi = np.gradient(phi, dx, axis=-1)
    flow = np.gradient(I0 * dphi, dx, axis=-1)
    SI = Sn_map * I0
    diff = np.gradient(np.gradient(SI, dx, axis=-1), dx, axis=-1)
    return I0 - (z_eff / k) * flow + diff


def _ray(I0, phi, Sn_map, z_eff, k, dx, n_blur_levels: int = 6):
    """Transporte geométrico: desplazamiento + ensanchamiento gaussiano.

    Conserva energía y permanece positivo aun con señales fuertes.
    """
    ny, nx = I0.shape
    u = (z_eff / k) * np.gradient(phi, dx, axis=-1)  # desplazamiento [µm]
    # Depósito con interpolación lineal en las posiciones desplazadas
    pos = np.arange(nx)[None, :] + u / dx
    j0 = np.floor(pos).astype(int)
    f = pos - j0
    Iz = np.zeros_like(I0)
    rows = np.repeat(np.arange(ny)[:, None], nx, axis=1)
    j0c = np.clip(j0, 0, nx - 1)
    j1c = np.clip(j0 + 1, 0, nx - 1)
    np.add.at(Iz, (rows, j0c), I0 * (1.0 - f))
    np.add.at(Iz, (rows, j1c), I0 * f)
    # Ensanchamiento con sigma variable: mezcla de copias con blur uniforme
    sigma = np.sqrt(2.0 * np.maximum(Sn_map, 0.0)) / dx  # [muestras]
    smax = float(sigma.max())
    if smax < 1e-6:
        return Iz
    levels = np.linspace(0.0, smax, n_blur_levels)
    blurred = np.stack(
        [Iz if s == 0 else gaussian_filter1d(Iz, s, axis=-1, mode="nearest") for s in levels]
    )
    idx = np.clip(sigma / (levels[1] - levels[0]), 0, n_blur_levels - 1 - 1e-9)
    lo = np.floor(idx).astype(int)
    w = idx - lo
    rows3 = np.arange(ny)[:, None]
    cols3 = np.arange(nx)[None, :]
    return (1.0 - w) * blurred[lo, rows3, cols3] + w * blurred[lo + 1, rows3, cols3]


def simulate(geom: Geometry, phantom: Phantom | None = None, cfg: SimConfig | None = None) -> SimResult:
    """Simula una adquisición single-shot (imagen con muestra + referencia)."""
    cfg = cfg or SimConfig()
    if cfg.model not in ("fokker_planck", "ray"):
        raise ValueError("model debe ser 'fokker_planck' o 'ray'")
    rng = np.random.default_rng(cfg.seed)

    p = geom.p_eff_um
    dx = p / cfg.oversample
    x_fine = fine_grid(geom, cfg.n_pix_x, cfg.oversample)
    y_rows = (np.arange(cfg.n_rows) + 0.5) * p
    x_pix = (np.arange(cfg.n_pix_x) + 0.5) * p

    m = mask_profile(geom, x_fine, blur=False)  # penumbra se aplica al final
    z_eff, k = geom.z_eff_um, geom.k_um

    if phantom is not None and phantom.shapes:
        T, phi, var_theta = phantom.maps(x_fine, y_rows, geom)
    else:
        T = np.ones((cfg.n_rows, x_fine.size))
        phi = np.zeros_like(T)
        var_theta = np.zeros_like(T)
    Sn_map = 0.5 * var_theta * z_eff**2  # [µm²]

    def detect(I_fine: np.ndarray) -> np.ndarray:
        """Penumbra + integración en píxeles + fotones + ruido."""
        if geom.sigma_pen_um > 0:
            I_fine = gaussian_filter1d(I_fine, geom.sigma_pen_um / dx, axis=-1, mode="nearest")
        I_pix = _pixel_average(I_fine, cfg.oversample) * cfg.photons_per_pixel
        n_neg = int(np.sum(I_pix < 0))
        if n_neg:
            warnings.warn(
                f"{n_neg} píxeles con intensidad media negativa recortados a 0 "
                "(señales fuera del régimen lineal TIE; considerá model='ray').",
                stacklevel=2,
            )
            I_pix = np.clip(I_pix, 0.0, None)
        if cfg.add_noise:
            I_pix = rng.poisson(I_pix).astype(float)
        return I_pix

    transport = _fokker_planck if cfg.model == "fokker_planck" else _ray

    I0_sample = T * m[None, :]
    Iz_sample = transport(I0_sample, phi, Sn_map, z_eff, k, dx)
    I_sample = detect(Iz_sample)

    I0_mask = np.broadcast_to(m, I0_sample.shape).copy()
    zeros = np.zeros_like(I0_mask)
    I_mask = detect(zeros + I0_mask)

    # ------------------- verdades de terreno por píxel ------------------- #
    dphi = np.gradient(phi, dx, axis=-1)
    d2phi = np.gradient(dphi, dx, axis=-1)
    gt = {
        "T": _pixel_average(T, cfg.oversample),
        "D_um": _pixel_average((z_eff / k) * dphi, cfg.oversample),
        "L": _pixel_average((z_eff / k) * d2phi, cfg.oversample),
        "Sn_um2": _pixel_average(Sn_map, cfg.oversample),
        "refraction_urad": _pixel_average(dphi / k, cfg.oversample) * 1e6,
    }

    fine = {}
    if cfg.return_fine:
        fine = {
            "x_um": x_fine,
            "mask": m,
            "I_sample": Iz_sample,
            "T": T,
            "phi": phi,
            "Sn_um2": Sn_map,
        }

    return SimResult(geom=geom, config=cfg, I_sample=I_sample, I_mask=I_mask,
                     x_pix_um=x_pix, gt=gt, fine=fine)
