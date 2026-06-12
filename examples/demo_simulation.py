"""
Demostración completa del framework single-mask DF-DPC.

Contiene dos ejemplos:

* ``--example paper``  (por defecto) — fantoma multi-material tipo paper
  (cilindro de PMMA, barra de grafito, cápsula de polvo de diamante y una
  esfera de tejido) en las tres configuraciones (DPC, DF y DF-DPC).

* ``--example biominerales`` — dos filas de cinco esferas (100→500 µm) de
  hidroxiapatita y de oxalato de calcio. Pensado para jugar con los
  PARÁMETROS DE MÁSCARA: editá el bloque ``MASK_PARAMS`` de abajo o pasalos
  por línea de comandos y observá cómo cambian el patrón, el contraste y la
  reconstrucción.

Uso:
    python examples/demo_simulation.py
    python examples/demo_simulation.py --example biominerales
    python examples/demo_simulation.py --example biominerales \\
        --p-mask 53 --slit 20 --p-det 55 --focal-spot 20 --config DF-DPC
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sm_df_dpc import (
    Cylinder,
    Geometry,
    Phantom,
    SimConfig,
    Slab,
    Sphere,
    material_library,
    retrieve,
    simulate,
)

# ===================================================================== #
#  ▼▼▼  BLOQUE EDITABLE — PARÁMETROS DE MÁSCARA (ejemplo biominerales) ▼▼▼
#
#  Cambiá estos valores (o pasalos por línea de comandos) y volvé a correr
#  para ver cómo afectan al patrón proyectado, al contraste y a la calidad
#  de la reconstrucción. La línea de comandos tiene prioridad sobre esto.
# ===================================================================== #
MASK_PARAMS = dict(
    config="DF-DPC",        # 'DPC' | 'DF' | 'DF-DPC'
    p_mask_um=53.0,         # período de la máscara [µm]
    slit_um=20.0,           # ancho de la apertura (slit) [µm]   (0 < slit < p_mask)
    mask_transmission=0.09, # transmisión residual del oro (0 = opaco)
    p_det_um=55.0,          # pitch del detector [µm]
    focal_spot_um=20.0,     # tamaño del foco de la fuente [µm]  (penumbra)
    energy_kev=40.0,        # energía efectiva del haz [keV]
    d_sm_m=0.60,            # distancia fuente–máscara [m] (d_md se alinea sola)
)

# Diámetros de las cinco esferas de cada material [µm]
SPHERE_DIAMETERS_UM = [100.0, 200.0, 300.0, 400.0, 500.0]
# ===================================================================== #
#  ▲▲▲  FIN DEL BLOQUE EDITABLE                                       ▲▲▲
# ===================================================================== #


def build_phantom(fov_x_um: float, fov_y_um: float) -> Phantom:
    """Fantoma multi-material inspirado en la Fig. 3/5/7 del paper."""
    mats = material_library(40.0)
    return Phantom(
        [
            # cilindro de PMMA (sin microestructura → solo atenuación y DPC)
            #Cylinder(mats["lung_tissue"], cx_um=0.18 * fov_x_um, radius_um=1100),
            # barra de grafito (atenuación + dark-field débil)
            #Cylinder(mats["lung_tissue"], cx_um=0.42 * fov_x_um, radius_um=1300),
            # cápsula de polvo de diamante (dark-field fuerte)
            #Slab(mats["lung_tissue"], 0.58 * fov_x_um, 0.76 * fov_x_um,
            #     thickness_um=900, taper_um=250),
            # esfera de tejido blando
            Sphere(mats["soft_tissue"], cx_um=0.18 * fov_x_um, cy_um=0.5 * fov_y_um,
                   radius_um=0.22 * fov_y_um),
            Sphere(mats["soft_tissue"], cx_um=0.42 * fov_x_um, cy_um=0.5 * fov_y_um,
                   radius_um=0.32 * fov_y_um),
            Sphere(mats["lung_tissue"], cx_um=0.58 * fov_x_um, cy_um=0.5 * fov_y_um,
                   radius_um=0.32 * fov_y_um),
            Sphere(mats["lung_tissue"], cx_um=0.88 * fov_x_um, cy_um=0.5 * fov_y_um,
                   radius_um=0.42 * fov_y_um),
        ]
    )


def run_config(config: str, photons: float, outdir: pathlib.Path) -> None:
    geom = Geometry(config=config, d_sm_m=0.70, focal_spot_um=20.0, energy_kev=40.0)
    print(f"\n=== Configuración {config} ===")
    print(geom.summary())
    geom.validate()

    cfg = SimConfig(
        n_pix_x=900,
        n_rows=160,
        oversample=32,
        photons_per_pixel=photons,
        add_noise=True,
        model="fokker_planck",
        seed=42,
    )
    fov_x = cfg.n_pix_x * geom.p_eff_um
    fov_y = cfg.n_rows * geom.p_eff_um
    phantom = build_phantom(fov_x, fov_y)

    res = simulate(geom, phantom, cfg)
    out = retrieve(res.I_sample, res.I_mask, geom, method="paper")

    extent = [0, fov_x / 1000, 0, fov_y / 1000]  # mm en el plano muestra

    # ---------------- figura: imágenes crudas del detector ---------------- #
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ax, img, title in [
        (axes[0], res.I_mask, "Reference with mask $I^{(M)}$"),
        (axes[1], res.I_sample, "with sample $I^{(S)}$"),
    ]:
        im = ax.imshow(img, extent=extent, origin="lower", cmap="gray", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="cuentas")
    axes[1].set_xlabel("x en plano muestra [mm]")
    fig.suptitle(f"Detector images {config} (M = {geom.M:.3f})")
    fig.tight_layout()
    fig.savefig(outdir / f"detector_{config.lower().replace('-', '_')}.png", dpi=140)
    plt.close(fig)

    # ------------- figura: reconstrucción vs verdad de terreno ------------- #
    panels = [("transmission", "T(1−L) — Attenuation", "T")]
    if out.dpc_um is not None:
        panels.append(("dpc_um", "Differential phase contrast [µm]", "D_um"))
    if out.Sn_um2 is not None:
        panels.append(("Sn_um2", "Dark-field — $S_n$ [µm²]", "Sn_um2"))

    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.8 * n), squeeze=False)
    for i, (attr, title, gt_key) in enumerate(panels):
        ret_img = getattr(out, attr)
        gt_img = res.gt[gt_key]
        finite = np.isfinite(ret_img)
        vmin, vmax = np.percentile(gt_img, [1, 99])
        if vmax <= vmin:
            vmin, vmax = gt_img.min() - 1e-6, gt_img.max() + 1e-6
        for j, (img, tag) in enumerate([(ret_img, "reconstructed"), (gt_img, "ref")]):
            ax = axes[i, j]
            im = ax.imshow(img, extent=extent, origin="lower", cmap="viridis",
                           aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} — {tag}", fontsize=10)
            ax.set_ylabel("y [mm]")
            fig.colorbar(im, ax=ax)
        rms = np.sqrt(np.nanmean((ret_img[finite] - gt_img[finite]) ** 2))
        print(f"  {attr:13s}: RMS = {rms:.4g}")
    for ax in axes[-1]:
        ax.set_xlabel("x [mm]")
    fig.suptitle(f" single-shot reconstruction —  {config}")
    fig.tight_layout()
    fig.savefig(outdir / f"retrieval_{config.lower().replace('-', '_')}.png", dpi=140)
    plt.close(fig)

    # --------- figura: perfiles centrales (validación cuantitativa) --------- #
    row = cfg.n_rows // 2
    x_mm = res.x_pix_um / 1000
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True, squeeze=False)
    for i, (attr, title, gt_key) in enumerate(panels):
        ax = axes[i, 0]
        ax.plot(x_mm, res.gt[gt_key][row], "k-", lw=1.6, label="verdad de terreno")
        ax.plot(x_mm, getattr(out, attr)[row], color="tab:orange", lw=0.9,
                alpha=0.9, label="reconstruida")
        ax.set_ylabel(title.split("—")[0].strip(), fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("x en plano muestra [mm]")
    fig.suptitle(f"Perfiles centrales — configuración {config}")
    fig.tight_layout()
    fig.savefig(outdir / f"profiles_{config.lower().replace('-', '_')}.png", dpi=140)
    plt.close(fig)


def build_biomineral_phantom(geom: Geometry, diameters_um, row_y_um, gap_um=600.0):
    """Dos filas de esferas: hidroxiapatita (arriba) y oxalato de calcio (abajo).

    Cada fila tiene una esfera por diámetro de ``diameters_um``, alineadas y
    centradas verticalmente en ``row_y_um = (y_HA, y_oxalato)``.

    Returns
    -------
    phantom, centers_x_um : el fantoma y los centros x de cada columna.
    """
    mats = material_library(geom.energy_kev)
    rmax = max(diameters_um) / 2.0
    pitch_x = 2 * rmax + gap_um             # separación entre columnas
    x0 = gap_um + rmax
    centers_x = [x0 + i * pitch_x for i in range(len(diameters_um))]

    shapes = []
    for cx, d in zip(centers_x, diameters_um):
        r = d / 2.0
        shapes.append(Sphere(mats["hydroxyapatite"], cx_um=cx, cy_um=row_y_um[0], radius_um=r))
        shapes.append(Sphere(mats["calcium_oxalate"], cx_um=cx, cy_um=row_y_um[1], radius_um=r))
    return Phantom(shapes), centers_x


def run_biomineral_spheres(params: dict, diameters_um, photons: float,
                           outdir: pathlib.Path) -> None:
    """Ejemplo configurable: 5 esferas (100–500 µm) de HA y de oxalato de Ca."""
    geom = Geometry(d_md_m=None, **params)
    print("\n=== Ejemplo biominerales — esferas de hidroxiapatita y oxalato de calcio ===")
    print(geom.summary())
    geom.validate()
    print(f"Diámetros de esfera : {', '.join(f'{d:.0f}' for d in diameters_um)} µm")

    # Geometría del fantoma: dos filas separadas, FOV ajustado al contenido
    rmax = max(diameters_um) / 2.0
    gap_um = 600.0
    fov_x = 2 * gap_um + len(diameters_um) * (2 * rmax + gap_um)
    row_sep = 2 * rmax + gap_um
    fov_y = row_sep + 2 * rmax + 2 * gap_um
    row_y = (fov_y / 2 + row_sep / 2, fov_y / 2 - row_sep / 2)  # HA arriba, oxalato abajo

    n_pix_x = int(np.ceil(fov_x / geom.p_eff_um))
    n_rows = int(np.ceil(fov_y / geom.p_eff_um))
    cfg = SimConfig(n_pix_x=n_pix_x, n_rows=n_rows, oversample=32,
                    photons_per_pixel=photons, add_noise=True,
                    model="fokker_planck", seed=42)

    phantom, centers_x = build_biomineral_phantom(geom, diameters_um, row_y, gap_um)
    res = simulate(geom, phantom, cfg)
    out = retrieve(res.I_sample, res.I_mask, geom, method="paper")

    extent = [0, fov_x / 1000, 0, fov_y / 1000]  # mm

    # --- panel de contrastes reconstruidos (los que la config provee) --- #
    panels = [("transmission", "Atenuación  T(1−L)")]
    if out.dpc_um is not None:
        panels.append(("dpc_um", "DPC — desplazamiento [µm]"))
    if out.Sn_um2 is not None:
        panels.append(("Sn_um2", "Dark-field — $S_n$ [µm²]"))

    n = len(panels)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.5 * (n + 1)), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(res.I_sample, extent=extent, origin="lower", cmap="gray", aspect="auto")
    ax.set_title("Imagen de detector con muestra $I^{(S)}$ (single-shot)", fontsize=10)
    ax.set_ylabel("y [mm]")
    for i, (attr, title) in enumerate(panels):
        ax = axes[i + 1, 0]
        img = getattr(out, attr)
        vmin, vmax = np.nanpercentile(img, [1, 99])
        im = ax.imshow(img, extent=extent, origin="lower", cmap="viridis",
                       aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax)
        # etiqueta de diámetro sobre cada columna
        for cx, d in zip(centers_x, diameters_um):
            ax.text(cx / 1000, extent[3] * 0.97, f"{d:.0f}", ha="center", va="top",
                    fontsize=7, color="white")
    axes[0, 0].text(0.01, 0.5, "HA →\noxalato →", transform=axes[0, 0].transAxes,
                    fontsize=7, color="yellow", va="center")
    axes[-1, 0].set_xlabel("x en plano muestra [mm]   (números = diámetro de esfera, µm)")
    title = (f"Biominerales — {geom.config}  |  p_mask={geom.p_mask_um:.0f} µm, "
             f"slit={geom.slit_um:.0f} µm, p_det={geom.p_det_um:.0f} µm, "
             f"foco={geom.focal_spot_um:.0f} µm, M={geom.M:.3f}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fname = outdir / "biominerales.png"
    fig.savefig(fname, dpi=140)
    plt.close(fig)

    # --- perfiles por la fila central de cada material (señal vs tamaño) --- #
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True, squeeze=False)
    x_mm = res.x_pix_um / 1000
    row_ha = int(round(row_y[0] / geom.p_eff_um))
    row_ox = int(round(row_y[1] / geom.p_eff_um))
    for i, (attr, title) in enumerate(panels):
        ax = axes[i, 0]
        img = getattr(out, attr)
        ax.plot(x_mm, img[min(row_ha, cfg.n_rows - 1)], color="tab:blue",
                lw=0.9, label="hidroxiapatita")
        ax.plot(x_mm, img[min(row_ox, cfg.n_rows - 1)], color="tab:red",
                lw=0.9, label="oxalato de calcio")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel("x en plano muestra [mm]")
    fig.suptitle("Perfiles por material — señal vs diámetro de esfera", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "biominerales_perfiles.png", dpi=140)
    plt.close(fig)
    print(f"  Figuras: {fname.name}, biominerales_perfiles.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", choices=["paper", "biominerales"], default="paper",
                        help="ejemplo a ejecutar")
    parser.add_argument("--photons", type=float, default=1000.0,
                        help="fotones por píxel (el paper usa 800–1200)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "output")
    parser.add_argument("--configs", nargs="+", default=["DPC", "DF", "DF-DPC"],
                        help="(ejemplo 'paper') configuraciones a simular")
    # Overrides de los parámetros de máscara (ejemplo 'biominerales')
    parser.add_argument("--config", default=None, choices=["DPC", "DF", "DF-DPC"])
    parser.add_argument("--p-mask", type=float, default=None, help="período de máscara [µm]")
    parser.add_argument("--slit", type=float, default=None, help="ancho de slit [µm]")
    parser.add_argument("--mask-transmission", type=float, default=None,
                        help="transmisión residual del oro")
    parser.add_argument("--p-det", type=float, default=None, help="pitch del detector [µm]")
    parser.add_argument("--focal-spot", type=float, default=None, help="tamaño del foco [µm]")
    parser.add_argument("--energy", type=float, default=None, help="energía [keV]")
    parser.add_argument("--d-sm", type=float, default=None, help="distancia fuente–máscara [m]")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.example == "paper":
        for config in args.configs:
            run_config(config, args.photons, args.out)
    else:
        params = dict(MASK_PARAMS)
        cli = {
            "config": args.config, "p_mask_um": args.p_mask, "slit_um": args.slit,
            "mask_transmission": args.mask_transmission, "p_det_um": args.p_det,
            "focal_spot_um": args.focal_spot, "energy_kev": args.energy,
            "d_sm_m": args.d_sm,
        }
        params.update({k: v for k, v in cli.items() if v is not None})
        run_biomineral_spheres(params, SPHERE_DIAMETERS_UM, args.photons, args.out)

    print(f"\nFiguras guardadas en {args.out}")


if __name__ == "__main__":
    main()
