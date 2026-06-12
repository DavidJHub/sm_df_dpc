# sm_df_dpc

Framework de **simulación de toma de imágenes de rayos X** y **reconstrucción
single-shot multi-contraste** para el método de máscara única (single-mask)
en sus tres configuraciones: **DPC**, **DF** y **DF-DPC**, con geometrías
totalmente personalizables.

Implementa el modelo de transporte de luz (Fokker–Planck) y los algoritmos de
recuperación de:

* J. Yuan & M. Das, *"Single-shot, single-mask X-ray dark-field and
  phase-contrast imaging"*, **Optica 12**, 1895 (2025).
  [doi:10.1364/OPTICA.578430](https://doi.org/10.1364/OPTICA.578430)
  — copias en [`papers/`](papers/).
* J. Yuan & M. Das, *"Transport-of-intensity model for single-mask x-ray
  differential phase contrast imaging"*, **Optica 11**, 478 (2024).

## Idea del método

Una única máscara absorbente (período ~53 µm) crea *beamlets* que se alinean
con los píxeles del detector de modo que la refracción (µrad) y el USAXS de la
muestra — efectos sub-píxel — quedan codificados como diferencias de intensidad
píxel a píxel. Tres alineamientos del mismo hardware seleccionan las señales:

| Config | Período proyectado | Beamlet alineado con | Señales recuperadas |
|--------|--------------------|----------------------|---------------------|
| `DPC`    | 2·p_det (M ≈ 2×) | fronteras de píxel (1 de cada 2) | atenuación + fase diferencial |
| `DF`     | 2·p_det (M ≈ 2×) | centros de píxel (1 de cada 2)   | atenuación + dark-field |
| `DF-DPC` | 3·p_det (M ≈ 3×) | 1 de cada 3 fronteras            | atenuación + DPC + dark-field |

Condición de alineamiento (Moiré nulo): `p_mask · M = N · p_det` con N = 2 o 3.

## Instalación

```bash
pip install -e .          # numpy + scipy (matplotlib opcional para los demos)
```

## Uso rápido

```python
from sm_df_dpc import (Geometry, Phantom, Cylinder, Slab, material_library,
                       SimConfig, simulate, retrieve)

# 1. Geometría (d_md se calcula sola para cumplir Moiré nulo)
geom = Geometry(config="DF-DPC", d_sm_m=0.60,
                p_mask_um=53.0, slit_um=20.0, p_det_um=55.0,
                focal_spot_um=20.0, energy_kev=40.0)
print(geom.summary())
geom.validate()          # avisos físicos: penumbra, slit, Moiré

# 2. Fantoma con materiales (delta, mu, microestructura USAXS)
mats = material_library(40.0)
phantom = Phantom([
    Cylinder(mats["pmma"], cx_um=3000, radius_um=1100),
    Slab(mats["diamond_powder"], 8000, 12000, thickness_um=900, taper_um=250),
])

# 3. Adquisición single-shot: imagen con muestra + referencia solo-máscara
res = simulate(geom, phantom, SimConfig(n_pix_x=900, n_rows=128,
                                        photons_per_pixel=1000, seed=42))

# 4. Reconstrucción (Ecs. 5/7/9 del paper, generalizadas)
out = retrieve(res.I_sample, res.I_mask, geom, method="paper")
out.transmission      # T(1−L)   atenuación
out.dpc_um            # D_n      desplazamiento del beamlet [µm]
out.refraction_urad   # θ = D/z_eff  ángulo de refracción [µrad]
out.darkfield         # señal adimensional del paper (≈ α3/α1 · S_n)
out.Sn_um2            # S_n      varianza de ensanchamiento [µm²]

# Verdades de terreno por píxel para validar: res.gt["T"], ["D_um"], ["Sn_um2"]
```

Demostración completa (las tres configuraciones, figuras en `examples/output/`):

```bash
python examples/demo_simulation.py
```

## Estructura

```
sm_df_dpc/
  geometry.py    Geometría parametrizable: pitches, slit, distancias, foco,
                 energía, offset de máscara. Alineamiento automático (Moiré
                 nulo), chequeos físicos y resumen.
  materials.py   Materiales: delta (fase), mu (atenuación) y varianza angular
                 USAXS por espesor (dark-field). Biblioteca a energía variable.
  phantom.py     Fantomas 2D: Slab, Wedge, Cylinder, Sphere, CustomShape →
                 mapas T(x,y), φ(x,y), σ_θ²(x,y).
  mask.py        Perfil de máscara proyectado al plano muestra (con penumbra)
                 y coeficientes por píxel (W_n, A_n, B_n) ≡ (w_e, α, α1, α2, α3).
  forward.py     Simulador: transporte 'fokker_planck' (Ec. 2 del paper) o
                 'ray' (desplazamiento+ensanchamiento, robusto a señal fuerte),
                 penumbra de la fuente, integración por píxel y ruido Poisson.
  retrieval.py   Reconstrucción single-shot: fórmulas cerradas del paper
                 (método 'paper') o mínimos cuadrados por grupos ('lstsq',
                 tolerante a desalineamientos).
notebooks/
  single_mask_df_dpc_params.ipynb   Calculadora de parámetros de laboratorio
                                    (distancias, slit óptimo, dithering).
  df_dpc_simulation_demo.ipynb      Demostración ejecutada del framework.
examples/demo_simulation.py         Demo CLI con las tres configuraciones.
tests/                              Validación: geometría, patrones de
                                    referencia, conservación de energía y
                                    circuito completo simulación→reconstrucción
                                    contra la verdad de terreno.
```

## Modelo numérico (resumen)

* Simulación en coordenadas del plano muestra (teorema de escalado de Fresnel):
  píxel efectivo `p_eff = p_det/M`, distancia efectiva `z_eff = d_sm·d_md/L`.
* Modelo por píxel: `I_n = T_n[(1−L_n)·W_n − D_n·A_n + S_n·B_n]`, con
  coeficientes integrados numéricamente del perfil real de la máscara — las
  constantes del paper son el caso particular del patrón canónico.
* `D_n = (z_eff/k)·∂xφ` (µm) y `S_n = ½·σ_θ²·z_eff²` (µm²); la señal dark-field
  adimensional del paper es `(α3/α1)·S_n ∈ [0, 1]`.
* Geometrías desalineadas producen franjas de Moiré reales en la simulación;
  `Geometry.validate()` las advierte y `method='lstsq'` las tolera moderadamente.

## Tests

```bash
python -m pytest tests/ -q
```
