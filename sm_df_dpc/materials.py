"""
Materiales para la simulación: índice de refracción complejo n = 1 − δ + iβ
y microestructura generadora de dark-field (USAXS).

Cada material queda descrito por tres cantidades a la energía de trabajo:

* ``delta``  — decremento del índice de refracción (fase: φ = −k·δ·t).
* ``mu_um``  — coeficiente de atenuación lineal [1/µm] (T = exp(−μ·t)).
* ``usaxs_urad2_per_um`` — crecimiento de la varianza angular del USAXS por
  unidad de espesor [µrad²/µm]. Es el parámetro de microestructura: nulo en
  materiales homogéneos (agua, PMMA), grande en polvos y medios porosos.

La señal dark-field del modelo Fokker–Planck se construye como
``S_n = σ_θ² · z_eff² / 2`` (varianza del ensanchamiento del beamlet en el
plano del detector, referida al plano muestra), donde σ_θ² es la varianza
angular acumulada a través de la muestra.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Radio clásico del electrón [µm]
R_E_UM = 2.8179403262e-9
#: h*c [keV·µm]
HC_KEV_UM = 1.23984193e-3
TWO_PI = 6.283185307179586


@dataclass(frozen=True)
class Material:
    """Propiedades ópticas de un material a la energía de trabajo."""

    name: str
    delta: float
    mu_um: float
    usaxs_urad2_per_um: float = 0.0

    @classmethod
    def from_electron_density(
        cls,
        name: str,
        rho_e_per_m3: float,
        mu_um: float,
        energy_kev: float,
        usaxs_urad2_per_um: float = 0.0,
    ) -> "Material":
        """Crea un material a partir de su densidad electrónica.

        δ = r_e λ² ρ_e / (2π), con λ = hc/E.
        """
        lam_um = HC_KEV_UM / energy_kev
        rho_e_per_um3 = rho_e_per_m3 * 1e-18
        delta = R_E_UM * lam_um**2 * rho_e_per_um3 / TWO_PI
        return cls(name, delta, mu_um, usaxs_urad2_per_um)


# --------------------------------------------------------------------- #
# Biblioteca de materiales
# --------------------------------------------------------------------- #
# Densidades electrónicas [e/m³] y atenuación lineal de referencia a 40 keV
# [1/µm] (valores aproximados a partir de tablas NIST XCOM; suficientes para
# simulación). μ se escala con (40/E)³ como aproximación fotoeléctrica.
_LIBRARY = {
    #  nombre            rho_e [e/m³]   mu@40keV [1/µm]  usaxs [µrad²/µm]
    "air":             (3.59e26,        2.4e-8,           0.0),
    "water":           (3.34e29,        2.68e-5,          0.0),
    "pmma":            (3.83e29,        2.77e-5,          0.0),
    "graphite":        (6.32e29,        4.37e-5,          0.05),
    "diamond":         (1.05e30,        7.30e-5,          0.0),
    "diamond_powder":  (5.80e29,        4.00e-5,          0.25),
    "aluminum":        (7.83e29,        1.54e-4,          0.0),
    "soft_tissue":     (3.46e29,        2.72e-5,          0.005),
    "lung_tissue":     (1.05e29,        0.85e-5,          0.15),
}


def material_library(energy_kev: float = 40.0) -> dict[str, Material]:
    """Materiales predefinidos evaluados a la energía dada.

    δ se calcula exactamente desde la densidad electrónica (∝ 1/E²);
    μ se escala desde su valor a 40 keV con (40/E)³ (aproximación válida en
    el rango ~20–80 keV para elementos ligeros).
    """
    scale_mu = (40.0 / energy_kev) ** 3
    return {
        name: Material.from_electron_density(
            name, rho_e, mu40 * scale_mu, energy_kev, usaxs_urad2_per_um=usaxs
        )
        for name, (rho_e, mu40, usaxs) in _LIBRARY.items()
    }
