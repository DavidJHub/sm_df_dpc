"""
sm_df_dpc — Framework de simulación y reconstrucción para imagen de rayos X
multi-contraste con máscara única (single-mask DPC / DF / DF-DPC).

Implementa el modelo de transporte de luz (Fokker–Planck) y los algoritmos de
recuperación single-shot descritos en:

* J. Yuan & M. Das, "Single-shot, single-mask X-ray dark-field and
  phase-contrast imaging", Optica 12, 1895 (2025).
  https://doi.org/10.1364/OPTICA.578430
* J. Yuan & M. Das, "Transport-of-intensity model for single-mask x-ray
  differential phase contrast imaging", Optica 11, 478 (2024).

Módulos
-------
geometry   Geometría parametrizable (distancias, máscara, detector, fuente).
materials  Propiedades ópticas de materiales (delta, mu, USAXS).
phantom    Fantomas 2D proyectados (cilindros, esferas, cuñas, slabs...).
mask       Perfil de transmisión de la máscara proyectado al plano muestra.
forward    Simulador de adquisición (modelos 'fokker_planck' y 'ray').
retrieval  Reconstrucción de atenuación, DPC y dark-field (Ecs. 5, 7 y 9).
"""

from .geometry import Geometry, N_BY_CONFIG
from .materials import Material, material_library
from .phantom import Phantom, Slab, Wedge, Cylinder, Sphere, CustomShape
from .mask import mask_profile, compute_coefficients, MaskCoefficients
from .forward import SimConfig, SimResult, simulate
from .retrieval import retrieve, MultiContrastResult

__version__ = "0.1.0"

__all__ = [
    "Geometry",
    "N_BY_CONFIG",
    "Material",
    "material_library",
    "Phantom",
    "Slab",
    "Wedge",
    "Cylinder",
    "Sphere",
    "CustomShape",
    "mask_profile",
    "compute_coefficients",
    "MaskCoefficients",
    "SimConfig",
    "SimResult",
    "simulate",
    "retrieve",
    "MultiContrastResult",
]
