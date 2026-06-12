"""Validación de circuito completo: simulación → reconstrucción → comparación
con la verdad de terreno, para las tres configuraciones del paper."""

import numpy as np
import pytest

from sm_df_dpc import (
    Geometry,
    Material,
    Phantom,
    SimConfig,
    Slab,
    Wedge,
    material_library,
    retrieve,
    simulate,
)

NOISELESS = dict(add_noise=False, photons_per_pixel=1000.0)


def _phantom_multicontraste(mats):
    """Cuña de PMMA (DPC), slab absorbente (atenuación) y slab dispersor (DF)."""
    absorber = Material("absorber", delta=1.65e-7, mu_um=2.77e-5, usaxs_urad2_per_um=0.0)
    scatterer = Material("scatterer", delta=1.0e-7, mu_um=1.5e-5, usaxs_urad2_per_um=0.06)
    return Phantom(
        [
            Wedge(mats["pmma"], 800, 5800, 0, 4000, taper_um=400),
            Slab(absorber, 7000, 10000, 8000, taper_um=400),
            Slab(scatterer, 11500, 15500, 2500, taper_um=400),
        ]
    )


def _interior(x, lo, hi):
    return (x > lo) & (x < hi)


@pytest.fixture(scope="module")
def sim_dfdpc():
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    ph = _phantom_multicontraste(mats)
    res = simulate(g, ph, SimConfig(n_pix_x=960, n_rows=2, **NOISELESS))
    return g, res


@pytest.mark.parametrize("method", ["paper", "lstsq"])
def test_dfdpc_full_roundtrip(sim_dfdpc, method):
    g, res = sim_dfdpc
    out = retrieve(res.I_sample, res.I_mask, g, method=method)
    x = res.x_pix_um

    # --- atenuación dentro del slab absorbente: T = exp(−μt) ---
    zone = _interior(x, 8000, 9000)
    T_ret = np.nanmean(out.transmission[0, zone])
    T_gt = np.nanmean(res.gt["T"][0, zone])
    assert T_ret == pytest.approx(T_gt, rel=0.02)

    # --- DPC dentro de la cuña: D constante ---
    zone = _interior(x, 2200, 4500)
    D_ret = np.nanmean(out.dpc_um[0, zone])
    D_gt = np.nanmean(res.gt["D_um"][0, zone])
    assert D_gt != 0
    assert D_ret == pytest.approx(D_gt, rel=0.10)

    # --- dark-field dentro del slab dispersor: S_n constante ---
    zone = _interior(x, 12500, 14500)
    S_ret = np.nanmean(out.Sn_um2[0, zone])
    S_gt = np.nanmean(res.gt["Sn_um2"][0, zone])
    assert S_gt > 0.5
    assert S_ret == pytest.approx(S_gt, rel=0.15)

    # --- sin señal espuria: DF ≈ 0 en la cuña, DPC ≈ 0 en el dispersor ---
    zone_w = _interior(x, 2200, 4500)
    assert abs(np.nanmean(out.Sn_um2[0, zone_w])) < 0.2 * S_gt
    zone_s = _interior(x, 12500, 14500)
    assert abs(np.nanmean(out.dpc_um[0, zone_s])) < 0.2 * abs(D_gt)


def test_dpc_config_roundtrip():
    g = Geometry(config="DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    ph = Phantom([Wedge(mats["pmma"], 800, 5800, 0, 4000, taper_um=400)])
    res = simulate(g, ph, SimConfig(n_pix_x=640, n_rows=2, **NOISELESS))
    out = retrieve(res.I_sample, res.I_mask, g)
    zone = _interior(res.x_pix_um, 2200, 4500)
    D_ret = np.nanmean(out.dpc_um[0, zone])
    D_gt = np.nanmean(res.gt["D_um"][0, zone])
    assert D_ret == pytest.approx(D_gt, rel=0.10)
    assert out.darkfield is None  # DPC no mide dark-field
    # ángulo de refracción coherente: θ = D/z_eff
    th = np.nanmean(out.refraction_urad[0, zone])
    assert th == pytest.approx(D_gt / g.z_eff_um * 1e6, rel=0.10)


def test_df_config_roundtrip():
    g = Geometry(config="DF", d_sm_m=0.6)
    scatterer = Material("scatterer", delta=1.0e-7, mu_um=1.5e-5, usaxs_urad2_per_um=0.06)
    ph = Phantom([Slab(scatterer, 4000, 9000, 2500, taper_um=400)])
    res = simulate(g, ph, SimConfig(n_pix_x=640, n_rows=2, **NOISELESS))
    out = retrieve(res.I_sample, res.I_mask, g)
    zone = _interior(res.x_pix_um, 5000, 8000)
    S_ret = np.nanmean(out.Sn_um2[0, zone])
    S_gt = np.nanmean(res.gt["Sn_um2"][0, zone])
    assert S_gt > 0.5
    assert S_ret == pytest.approx(S_gt, rel=0.15)
    assert out.dpc_um is None  # DF no mide fase diferencial
    # señal adimensional del paper entre 0 y 1
    assert 0 < np.nanmean(out.darkfield[0, zone]) < 1


def test_methods_agree(sim_dfdpc):
    g, res = sim_dfdpc
    a = retrieve(res.I_sample, res.I_mask, g, method="paper")
    b = retrieve(res.I_sample, res.I_mask, g, method="lstsq")
    zone = _interior(res.x_pix_um, 2200, 4500)
    assert np.nanmean(a.dpc_um[0, zone]) == pytest.approx(
        np.nanmean(b.dpc_um[0, zone]), rel=0.05
    )


def test_retrieval_with_noise_converges():
    """Con ruido Poisson, promediar filas converge a la verdad de terreno."""
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    ph = Phantom([Wedge(mats["pmma"], 800, 5800, 0, 4000, taper_um=400)])
    cfg = SimConfig(n_pix_x=640, n_rows=64, photons_per_pixel=5000, add_noise=True, seed=7)
    res = simulate(g, ph, cfg)
    out = retrieve(res.I_sample, res.I_mask, g)
    zone = _interior(res.x_pix_um, 2200, 4500)
    D_ret = np.nanmean(out.dpc_um[:, zone])
    D_gt = np.nanmean(res.gt["D_um"][:, zone])
    assert D_ret == pytest.approx(D_gt, rel=0.15)


def test_1d_input_accepted():
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    res = simulate(g, None, SimConfig(n_pix_x=96, n_rows=1, **NOISELESS))
    out = retrieve(res.I_sample[0], res.I_mask[0], g)
    assert out.transmission.ndim == 1
    assert np.nanmean(out.transmission) == pytest.approx(1.0, abs=0.02)
