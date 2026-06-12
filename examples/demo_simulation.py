"""
Demostración completa del framework single-mask DF-DPC.

Simula la adquisición single-shot de un fantoma multi-material (similar al
del paper: cilindro de PMMA, barra de grafito, cápsula de polvo de diamante
y una esfera de tejido) en las tres configuraciones (DPC, DF y DF-DPC), y
reconstruye atenuación, fase diferencial y dark-field comparando con la
verdad de terreno.

Uso:
    python examples/demo_simulation.py [--photons 2000] [--out examples/output]
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


def build_phantom(fov_x_um: float, fov_y_um: float) -> Phantom:
    """Fantoma multi-material inspirado en la Fig. 3/5/7 del paper."""
    mats = material_library(40.0)
    return Phantom(
        [
            # cilindro de PMMA (sin microestructura → solo atenuación y DPC)
            Cylinder(mats["pmma"], cx_um=0.18 * fov_x_um, radius_um=1100),
            # barra de grafito (atenuación + dark-field débil)
            Cylinder(mats["graphite"], cx_um=0.42 * fov_x_um, radius_um=1300),
            # cápsula de polvo de diamante (dark-field fuerte)
            Slab(mats["diamond_powder"], 0.58 * fov_x_um, 0.76 * fov_x_um,
                 thickness_um=900, taper_um=250),
            # esfera de tejido blando
            Sphere(mats["soft_tissue"], cx_um=0.88 * fov_x_um, cy_um=0.5 * fov_y_um,
                   radius_um=0.42 * fov_y_um),
        ]
    )


def run_config(config: str, photons: float, outdir: pathlib.Path) -> None:
    geom = Geometry(config=config, d_sm_m=0.60, focal_spot_um=20.0, energy_kev=40.0)
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
        (axes[0], res.I_mask, "Referencia solo-máscara $I^{(M)}$"),
        (axes[1], res.I_sample, "Con muestra $I^{(S)}$"),
    ]:
        im = ax.imshow(img, extent=extent, origin="lower", cmap="gray", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, label="cuentas")
    axes[1].set_xlabel("x en plano muestra [mm]")
    fig.suptitle(f"Imágenes de detector — configuración {config} (M = {geom.M:.3f})")
    fig.tight_layout()
    fig.savefig(outdir / f"detector_{config.lower().replace('-', '_')}.png", dpi=140)
    plt.close(fig)

    # ------------- figura: reconstrucción vs verdad de terreno ------------- #
    panels = [("transmission", "T(1−L) — atenuación", "T")]
    if out.dpc_um is not None:
        panels.append(("dpc_um", "DPC — desplazamiento del beamlet [µm]", "D_um"))
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
        for j, (img, tag) in enumerate([(ret_img, "reconstruida"), (gt_img, "verdad de terreno")]):
            ax = axes[i, j]
            im = ax.imshow(img, extent=extent, origin="lower", cmap="viridis",
                           aspect="auto", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} — {tag}", fontsize=10)
            ax.set_ylabel("y [mm]")
            fig.colorbar(im, ax=ax)
        rms = np.sqrt(np.nanmean((ret_img[finite] - gt_img[finite]) ** 2))
        print(f"  {attr:13s}: RMS(reconstrucción − GT) = {rms:.4g}")
    for ax in axes[-1]:
        ax.set_xlabel("x en plano muestra [mm]")
    fig.suptitle(f"Reconstrucción single-shot — configuración {config}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=float, default=1000.0,
                        help="fotones por píxel (el paper usa 800–1200)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path(__file__).parent / "output")
    parser.add_argument("--configs", nargs="+", default=["DPC", "DF", "DF-DPC"])
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for config in args.configs:
        run_config(config, args.photons, args.out)
    print(f"\nFiguras guardadas en {args.out}")


if __name__ == "__main__":
    main()
