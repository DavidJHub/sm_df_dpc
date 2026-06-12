"""
Fantomas 2D para el simulador: colecciones de formas con espesor proyectado
t(x, y) a lo largo del eje óptico, evaluadas en el plano de la muestra.

El eje x es la dirección de modulación de la máscara (franjas verticales);
las filas y son independientes en el modelo de transporte 1D del paper.

A partir del espesor por material se construyen los tres mapas físicos:

* Transmisión        T(x,y)  = exp(−Σ μ_i t_i)
* Fase               φ(x,y)  = −k Σ δ_i t_i
* Varianza USAXS     σ_θ²(x,y) = Σ v_i t_i      [rad²]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geometry import Geometry
from .materials import Material


def _smoothstep(u: np.ndarray) -> np.ndarray:
    """Transición suave C¹ entre 0 y 1 (clampeada)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _edge(x: np.ndarray, x0: float, x1: float, taper_um: float) -> np.ndarray:
    """Ventana [x0, x1] con bordes suavizados en una longitud taper_um."""
    if taper_um <= 0:
        return ((x >= x0) & (x <= x1)).astype(float)
    rise = _smoothstep((x - x0) / taper_um + 0.5)
    fall = _smoothstep((x1 - x) / taper_um + 0.5)
    return rise * fall


class Shape:
    """Forma con espesor proyectado. Subclases implementan thickness()."""

    material: Material

    def thickness(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
        """Espesor proyectado [µm] en la grilla (len(y), len(x))."""
        raise NotImplementedError


@dataclass
class Slab(Shape):
    """Placa de espesor constante entre x0 y x1 (bordes suavizables)."""

    material: Material
    x0_um: float
    x1_um: float
    thickness_um: float
    taper_um: float = 0.0

    def thickness(self, x_um, y_um):
        prof = self.thickness_um * _edge(x_um, self.x0_um, self.x1_um, self.taper_um)
        return np.broadcast_to(prof, (len(y_um), len(x_um))).copy()


@dataclass
class Wedge(Shape):
    """Cuña: espesor lineal de t0 (en x0) a t1 (en x1), con bordes suaves.

    Útil para validación: produce un gradiente de fase constante y por lo
    tanto una señal DPC constante en su interior.
    """

    material: Material
    x0_um: float
    x1_um: float
    t0_um: float
    t1_um: float
    taper_um: float = 0.0

    def thickness(self, x_um, y_um):
        ramp = self.t0_um + (self.t1_um - self.t0_um) * (
            (x_um - self.x0_um) / (self.x1_um - self.x0_um)
        )
        prof = ramp * _edge(x_um, self.x0_um, self.x1_um, self.taper_um)
        return np.broadcast_to(prof, (len(y_um), len(x_um))).copy()


@dataclass
class Cylinder(Shape):
    """Cilindro con eje paralelo a y (vertical): t = 2√(r² − (x−cx)²)."""

    material: Material
    cx_um: float
    radius_um: float

    def thickness(self, x_um, y_um):
        dx2 = (x_um - self.cx_um) ** 2
        prof = 2.0 * np.sqrt(np.maximum(self.radius_um**2 - dx2, 0.0))
        return np.broadcast_to(prof, (len(y_um), len(x_um))).copy()


@dataclass
class Sphere(Shape):
    """Esfera centrada en (cx, cy): t = 2√(r² − (x−cx)² − (y−cy)²)."""

    material: Material
    cx_um: float
    cy_um: float
    radius_um: float

    def thickness(self, x_um, y_um):
        d2 = (x_um[None, :] - self.cx_um) ** 2 + (y_um[:, None] - self.cy_um) ** 2
        return 2.0 * np.sqrt(np.maximum(self.radius_um**2 - d2, 0.0))


@dataclass
class CustomShape(Shape):
    """Forma arbitraria definida por una función t(x_um, y_um) → [µm]."""

    material: Material
    thickness_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def thickness(self, x_um, y_um):
        t = np.asarray(self.thickness_fn(x_um, y_um), dtype=float)
        return np.broadcast_to(t, (len(y_um), len(x_um))).copy()


@dataclass
class Phantom:
    """Colección de formas que componen la muestra."""

    shapes: list[Shape] = field(default_factory=list)

    def add(self, shape: Shape) -> "Phantom":
        self.shapes.append(shape)
        return self

    def maps(
        self, x_um: np.ndarray, y_um: np.ndarray, geom: Geometry
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mapas físicos en la grilla del plano muestra.

        Returns
        -------
        T : transmisión (len(y), len(x))
        phi : fase acumulada [rad]
        var_theta : varianza angular USAXS [rad²]
        """
        ny, nx = len(y_um), len(x_um)
        mu_t = np.zeros((ny, nx))
        delta_t = np.zeros((ny, nx))
        var_theta = np.zeros((ny, nx))
        for shape in self.shapes:
            t = shape.thickness(np.asarray(x_um, float), np.asarray(y_um, float))
            m = shape.material
            mu_t += m.mu_um * t
            delta_t += m.delta * t
            var_theta += m.usaxs_urad2_per_um * 1e-12 * t  # µrad² → rad²
        T = np.exp(-mu_t)
        phi = -geom.k_um * delta_t
        return T, phi, var_theta
