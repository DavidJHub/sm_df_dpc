import numpy as np
import pytest

from sm_df_dpc import (
    Geometry,
    Material,
    Phantom,
    SimConfig,
    Slab,
    Wedge,
    compute_coefficients,
    material_library,
    simulate,
)

NOISELESS = dict(add_noise=False, photons_per_pixel=1000.0)


def test_dpc_reference_is_uniform():
    """En la configuración DPC la máscara es invisible sin muestra (Fig. 2b)."""
    g = Geometry(config="DPC", d_sm_m=0.6)
    res = simulate(g, None, SimConfig(n_pix_x=64, n_rows=2, **NOISELESS))
    inner = res.I_mask[:, 4:-4]
    assert np.std(inner) / np.mean(inner) < 5e-3


def test_df_reference_has_high_contrast():
    """En la configuración DF la referencia alterna píxeles claros/oscuros."""
    g = Geometry(config="DF", d_sm_m=0.6)
    res = simulate(g, None, SimConfig(n_pix_x=64, n_rows=2, **NOISELESS))
    row = res.I_mask[0, 4:-4]
    bright, dark = row[::2], row[1::2]
    if bright.mean() < dark.mean():
        bright, dark = dark, bright
    visibility = (bright.mean() - dark.mean()) / (bright.mean() + dark.mean())
    assert visibility > 0.4  # el paper reporta ~50%


def test_dfdpc_reference_three_pixel_period():
    """DF-DPC: patrón de período 3 con dos píxeles brillantes y uno oscuro."""
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    res = simulate(g, None, SimConfig(n_pix_x=66, n_rows=2, **NOISELESS))
    row = res.I_mask[0, 3:-3]
    n = (len(row) // 3) * 3
    groups = row[:n].reshape(-1, 3).mean(axis=0)
    g_sorted = np.sort(groups)
    # un píxel oscuro y dos brillantes parecidos entre sí
    assert g_sorted[2] / g_sorted[0] > 3.0
    assert abs(g_sorted[2] - g_sorted[1]) / g_sorted[2] < 0.1


def test_energy_conservation_ray_model():
    """El modelo de rayos conserva la energía total (sin atenuación)."""
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    # fase pura + dark-field, sin absorción
    mat = Material("test", delta=2e-7, mu_um=0.0, usaxs_urad2_per_um=0.1)
    ph = Phantom([Wedge(mat, 1000, 5000, 0, 2000, taper_um=300)])
    cfg = SimConfig(n_pix_x=384, n_rows=2, model="ray", **NOISELESS)
    res = simulate(g, ph, cfg)
    inner = slice(8, -8)
    assert res.I_sample[:, inner].sum() == pytest.approx(
        res.I_mask[:, inner].sum(), rel=2e-2
    )


def test_fokker_planck_matches_ray_for_weak_signals():
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    ph = Phantom([Wedge(mats["pmma"], 1500, 6500, 0, 3000, taper_um=400)])
    base = dict(n_pix_x=512, n_rows=2, **NOISELESS)
    r_fp = simulate(g, ph, SimConfig(model="fokker_planck", **base))
    r_ray = simulate(g, ph, SimConfig(model="ray", **base))
    inner = slice(6, -6)
    diff = r_fp.I_sample[:, inner] - r_ray.I_sample[:, inner]
    mean = r_fp.I_mask.mean()
    # ambos modelos coinciden a primer orden; el residuo es O(u²) en los bordes
    assert np.sqrt(np.mean(diff**2)) / mean < 0.01
    assert np.abs(diff).max() / mean < 0.05


def test_ground_truth_wedge_constant_dpc():
    """Una cuña produce señal DPC (desplazamiento del beamlet) constante."""
    g = Geometry(config="DF-DPC", d_sm_m=0.6)
    mats = material_library(40.0)
    wedge = Wedge(mats["pmma"], 1000, 7000, 0, 3000, taper_um=300)
    res = simulate(g, Phantom([wedge]), SimConfig(n_pix_x=512, n_rows=2, **NOISELESS))
    interior = (res.x_pix_um > 2500) & (res.x_pix_um < 5500)
    D = res.gt["D_um"][0, interior]
    # D = −z_eff·δ·dt/dx, pendiente = 3000/6000 = 0.5
    expected = -g.z_eff_um * mats["pmma"].delta * 0.5
    assert D == pytest.approx(expected, rel=1e-2)
    assert np.ptp(D) / abs(expected) < 0.02


def test_mask_coefficients_canonical_patterns():
    # DPC: W uniforme, A alternante; DF: W alternante; DF-DPC: período 3
    g_dpc = Geometry(config="DPC", d_sm_m=0.6)
    c = compute_coefficients(g_dpc, n_pix=32)
    assert np.std(c.W) / c.we < 1e-3
    assert abs(c.alpha) > 0.01
    assert c.dpc_sensitivity > 0

    g_df = Geometry(config="DF", d_sm_m=0.6)
    c = compute_coefficients(g_df, n_pix=32)
    assert c.alpha1 > 0.1
    assert c.alpha3 > 0
    assert c.df_sensitivity > 0

    g_3 = Geometry(config="DF-DPC", d_sm_m=0.6)
    c = compute_coefficients(g_3, n_pix=33)
    assert c.alpha1 > 0.1 and c.alpha2 != 0 and c.alpha3 > 0
    # transparencia efectiva ≈ (slit + T_au·(p−slit))/p
    we_teo = (20 + 0.09 * 33) / 53
    assert c.we == pytest.approx(we_teo, rel=0.02)


def test_poisson_noise_statistics():
    g = Geometry(config="DF", d_sm_m=0.6)
    cfg = SimConfig(n_pix_x=64, n_rows=256, photons_per_pixel=2000, add_noise=True, seed=1)
    res = simulate(g, None, cfg)
    bright = res.I_mask[:, 10]  # columna brillante, estadística entre filas
    assert np.var(bright) == pytest.approx(np.mean(bright), rel=0.2)
