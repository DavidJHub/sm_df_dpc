"""
Geometría parametrizable del sistema de imagen single-mask DF-DPC.

Convenciones
------------
* Distancias axiales (fuente-máscara, máscara-detector) en metros.
* Longitudes transversales (pitches, slit, foco) en micrómetros.
* La muestra se asume apoyada contra la máscara (d_so = d_sm), como en el
  paper, de modo que la magnificación de la máscara y de la muestra coinciden.
* La simulación se realiza en coordenadas del plano de la muestra/máscara
  (teorema de escalado de Fresnel): el píxel del detector proyectado a ese
  plano mide ``p_eff = p_det / M`` y la distancia de propagación efectiva es
  ``z_eff = d_sm * d_md / (d_sm + d_md) = d_md / M``.

Condición de alineamiento (Sección EXPERIMENT del paper):
    p_mask * M = N * p_det
con N = 2 (configuraciones DPC y DF) o N = 3 (configuración DF-DPC).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

N_BY_CONFIG = {"DPC": 2, "DF": 2, "DF-DPC": 3}

#: h*c en keV·µm — lambda[µm] = HC_KEV_UM / E[keV]
HC_KEV_UM = 1.23984193e-3

TWO_PI = 6.283185307179586


@dataclass(frozen=True)
class Geometry:
    """Geometría completa del sistema single-mask.

    Parameters
    ----------
    config : str
        'DPC', 'DF' o 'DF-DPC'. Determina N (período proyectado = N·p_det)
        y el alineamiento canónico beamlet–píxel.
    p_mask_um, slit_um : float
        Período y apertura de la máscara [µm].
    mask_transmission : float
        Transmisión residual de las franjas de oro (≈0.09 para 100 µm de Au
        a 40 keV).
    p_det_um : float
        Pitch del detector [µm].
    d_sm_m : float
        Distancia fuente–máscara [m].
    d_md_m : float or None
        Distancia máscara–detector [m]. Si es None se calcula la distancia
        alineada (Moiré nulo) para la configuración elegida.
    focal_spot_um : float
        Diámetro (FWHM) del foco de la fuente [µm]; genera la penumbra.
    energy_kev : float
        Energía efectiva del haz [keV] (modelo monocromático efectivo).
    mask_offset_um : float or None
        Desplazamiento lateral de la máscara respecto a la rejilla de
        píxeles, en coordenadas del plano muestra. None → offset canónico
        de la configuración.
    """

    config: str = "DF-DPC"
    p_mask_um: float = 53.0
    slit_um: float = 20.0
    mask_transmission: float = 0.09
    p_det_um: float = 55.0
    d_sm_m: float = 0.60
    d_md_m: float | None = None
    focal_spot_um: float = 20.0
    energy_kev: float = 40.0
    mask_offset_um: float | None = None

    def __post_init__(self):
        cfg = self.config.upper()
        if cfg not in N_BY_CONFIG:
            raise ValueError(f"config debe ser uno de {list(N_BY_CONFIG)}, no {self.config!r}")
        object.__setattr__(self, "config", cfg)
        if not 0.0 < self.slit_um < self.p_mask_um:
            raise ValueError("slit_um debe cumplir 0 < slit < p_mask")
        if self.d_sm_m <= 0:
            raise ValueError("d_sm_m debe ser positiva")
        if self.d_md_m is None:
            # Distancia alineada: M* = N p_det / p_mask  →  d_md = d_sm (M*-1)
            m_star = self.N * self.p_det_um / self.p_mask_um
            if m_star <= 1:
                raise ValueError(
                    f"Magnificación alineada M*={m_star:.3f} ≤ 1: "
                    "no existe geometría válida (p_mask demasiado grande)."
                )
            object.__setattr__(self, "d_md_m", self.d_sm_m * (m_star - 1.0))
        elif self.d_md_m <= 0:
            raise ValueError("d_md_m debe ser positiva")

    # ------------------------------------------------------------------ #
    # Propiedades derivadas
    # ------------------------------------------------------------------ #
    @property
    def N(self) -> int:
        """Período proyectado de la máscara en unidades de p_det (2 o 3)."""
        return N_BY_CONFIG[self.config]

    @property
    def L_m(self) -> float:
        """Distancia total fuente–detector [m]."""
        return self.d_sm_m + self.d_md_m

    @property
    def M(self) -> float:
        """Magnificación geométrica de la máscara (y de la muestra)."""
        return self.L_m / self.d_sm_m

    @property
    def p_eff_um(self) -> float:
        """Pitch del detector proyectado al plano de la muestra [µm]."""
        return self.p_det_um / self.M

    @property
    def z_eff_um(self) -> float:
        """Distancia de propagación efectiva (escalado de Fresnel) [µm]."""
        return (self.d_sm_m * self.d_md_m / self.L_m) * 1e6

    @property
    def wavelength_um(self) -> float:
        return HC_KEV_UM / self.energy_kev

    @property
    def k_um(self) -> float:
        """Número de onda k = 2π/λ [1/µm]."""
        return TWO_PI / self.wavelength_um

    @property
    def penumbra_det_um(self) -> float:
        """Penumbra del foco en el plano del detector [µm]."""
        return self.focal_spot_um * (self.M - 1.0)

    @property
    def penumbra_sample_um(self) -> float:
        """Penumbra del foco referida al plano de la muestra [µm]."""
        return self.penumbra_det_um / self.M

    @property
    def sigma_pen_um(self) -> float:
        """Sigma gaussiana equivalente de la penumbra (plano muestra) [µm]."""
        return self.penumbra_sample_um / 2.3548200450309493  # FWHM → sigma

    @property
    def moire_mismatch(self) -> float:
        """Desviación relativa de la condición de Moiré nulo.

        (p_mask·M − N·p_det) / (N·p_det). Cero = alineamiento perfecto.
        """
        target = self.N * self.p_det_um
        return (self.p_mask_um * self.M - target) / target

    @property
    def canonical_offset_um(self) -> float:
        """Offset lateral canónico de la máscara (plano muestra) [µm].

        Posición del centro del primer beamlet:
        * DPC    : frontera de píxel (x = 0), un beamlet cada 2 píxeles.
        * DF     : centro de píxel (x = p_eff/2), un beamlet cada 2 píxeles.
        * DF-DPC : frontera entre los dos píxeles brillantes del grupo de 3
                   (x = p_eff), píxel n+1 del grupo queda oscuro.
        """
        if self.config == "DPC":
            return 0.0
        if self.config == "DF":
            return 0.5 * self.p_eff_um
        return self.p_eff_um  # DF-DPC

    @property
    def mask_offset_resolved_um(self) -> float:
        if self.mask_offset_um is None:
            return self.canonical_offset_um
        return float(self.mask_offset_um)

    @property
    def beamlet_span_pixels(self) -> int:
        """Píxeles que el beamlet debe iluminar según la configuración.

        * DF     : 1 (beamlet centrado en un píxel, vecinos oscuros).
        * DPC    : 2 (beamlet a caballo de una frontera de píxel).
        * DF-DPC : 2 (dos píxeles brillantes + uno oscuro por período).
        """
        return 1 if self.config == "DF" else 2

    @property
    def slit_max_um(self) -> float:
        """Máximo slit para que el beamlet (+penumbra) quepa en su zona.

        w·M + f_s·(M−1) ≤ n·p_det  →  w_max = (n·p_det − f_s(M−1)) / M,
        con n = ``beamlet_span_pixels``.
        """
        return (self.beamlet_span_pixels * self.p_det_um - self.penumbra_det_um) / self.M

    # ------------------------------------------------------------------ #
    # Constructores y utilidades
    # ------------------------------------------------------------------ #
    @classmethod
    def from_total_length(cls, L_m: float, **kwargs) -> "Geometry":
        """Geometría alineada dado el largo total L = d_sm + d_md.

        Reparte L según la magnificación alineada M* = N·p_det/p_mask:
        d_sm = L/M*, d_md = L (M*−1)/M*.
        """
        probe = cls(d_md_m=None, **kwargs)
        m_star = probe.N * probe.p_det_um / probe.p_mask_um
        return replace(probe, d_sm_m=L_m / m_star, d_md_m=L_m * (m_star - 1.0) / m_star)

    def aligned(self) -> "Geometry":
        """Devuelve la misma geometría con d_md ajustada a Moiré nulo."""
        return replace(self, d_md_m=None)

    def validate(self, moire_tol: float = 0.02, warn: bool = True) -> list[str]:
        """Chequeos físicos de la configuración. Devuelve lista de avisos."""
        msgs: list[str] = []
        if self.penumbra_det_um >= self.p_det_um:
            msgs.append(
                f"[CRÍTICO] Penumbra en detector ({self.penumbra_det_um:.1f} µm) ≥ "
                f"p_det ({self.p_det_um:.1f} µm): se pierde el contraste píxel a píxel. "
                f"Usá un foco < {self.p_det_um / (self.M - 1):.1f} µm o menor M."
            )
        elif self.slit_um > self.slit_max_um:
            msgs.append(
                f"[ADVERTENCIA] slit ({self.slit_um:.1f} µm) > w_max "
                f"({self.slit_max_um:.1f} µm): la zona iluminada "
                f"({self.slit_um * self.M + self.penumbra_det_um:.1f} µm) excede los "
                f"{self.beamlet_span_pixels} píxel(es) de diseño "
                f"({self.beamlet_span_pixels * self.p_det_um:.1f} µm): contraste reducido."
            )
        if abs(self.moire_mismatch) > moire_tol:
            msgs.append(
                f"[ADVERTENCIA] Desalineamiento de Moiré {self.moire_mismatch * 100:+.2f}% "
                f"(p_mask·M = {self.p_mask_um * self.M:.2f} µm vs N·p_det = "
                f"{self.N * self.p_det_um:.2f} µm). Aparecerán franjas de Moiré."
            )
        if warn:
            for m in msgs:
                warnings.warn(m, stacklevel=2)
        return msgs

    def summary(self) -> str:
        """Resumen legible de la geometría."""
        lines = [
            f"Configuración        : {self.config}  (N = {self.N})",
            f"d_sm / d_md / L      : {self.d_sm_m * 100:.2f} / {self.d_md_m * 100:.2f} / "
            f"{self.L_m * 100:.2f} cm",
            f"Magnificación M      : {self.M:.4f}",
            f"p_mask / slit        : {self.p_mask_um:.1f} / {self.slit_um:.1f} µm "
            f"(duty {self.slit_um / self.p_mask_um:.2f}, T_oro {self.mask_transmission:.2f})",
            f"p_det / p_eff        : {self.p_det_um:.1f} / {self.p_eff_um:.2f} µm",
            f"Período proyectado   : {self.p_mask_um * self.M:.2f} µm "
            f"(objetivo {self.N * self.p_det_um:.2f} µm, desvío "
            f"{self.moire_mismatch * 100:+.3f}%)",
            f"z_eff                : {self.z_eff_um * 1e-4:.2f} cm",
            f"Energía / k          : {self.energy_kev:.1f} keV / {self.k_um:.3e} µm⁻¹",
            f"Foco / penumbra det. : {self.focal_spot_um:.1f} / {self.penumbra_det_um:.1f} µm",
            f"Offset máscara       : {self.mask_offset_resolved_um:.2f} µm "
            f"({'canónico' if self.mask_offset_um is None else 'usuario'})",
        ]
        return "\n".join(lines)
