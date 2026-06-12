import numpy as np
import pytest

from sm_df_dpc import Geometry, N_BY_CONFIG


def test_aligned_distances_satisfy_moire_condition():
    for cfg in ("DPC", "DF", "DF-DPC"):
        g = Geometry(config=cfg, d_sm_m=0.6, d_md_m=None)
        assert g.p_mask_um * g.M == pytest.approx(g.N * g.p_det_um, rel=1e-12)
        assert g.moire_mismatch == pytest.approx(0.0, abs=1e-12)


def test_from_total_length():
    g = Geometry.from_total_length(1.8, config="DF-DPC")
    assert g.L_m == pytest.approx(1.8)
    assert abs(g.moire_mismatch) < 1e-12


def test_paper_geometry_values():
    # Paper: d_sm ≈ 60 cm; DPC → M ≈ 2×, DF-DPC → M ≈ 3×
    g2 = Geometry(config="DPC", d_sm_m=0.6)
    g3 = Geometry(config="DF-DPC", d_sm_m=0.6)
    assert g2.M == pytest.approx(2 * 55.0 / 53.0)
    assert g3.M == pytest.approx(3 * 55.0 / 53.0)
    assert g3.d_md_m == pytest.approx(0.6 * (g3.M - 1))
    # z_eff = d_md / M
    assert g3.z_eff_um == pytest.approx(g3.d_md_m * 1e6 / g3.M)


def test_misaligned_geometry_warns():
    g = Geometry(config="DPC", d_sm_m=0.6, d_md_m=0.9)
    msgs = g.validate(warn=False)
    assert any("Moiré" in m for m in msgs)


def test_penumbra_critical_warning():
    g = Geometry(config="DF-DPC", d_sm_m=0.6, focal_spot_um=50.0)
    msgs = g.validate(warn=False)
    assert any("CRÍTICO" in m for m in msgs)


def test_slit_max_matches_formula():
    # DF: el beamlet cabe en 1 píxel; DPC y DF-DPC: a caballo de 2 píxeles
    g_df = Geometry(config="DF", d_sm_m=0.6, focal_spot_um=20.0)
    expected = (g_df.p_det_um - g_df.focal_spot_um * (g_df.M - 1)) / g_df.M
    assert g_df.slit_max_um == pytest.approx(expected)
    g3 = Geometry(config="DF-DPC", d_sm_m=0.6, focal_spot_um=20.0)
    expected = (2 * g3.p_det_um - g3.focal_spot_um * (g3.M - 1)) / g3.M
    assert g3.slit_max_um == pytest.approx(expected)
    # con los parámetros del paper (slit 20 µm) la geometría DF-DPC es válida
    assert g3.slit_um < g3.slit_max_um


def test_wavenumber_at_40kev():
    g = Geometry(energy_kev=40.0)
    assert g.k_um == pytest.approx(2 * np.pi * 40.0 / 1.23984193e-3, rel=1e-9)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        Geometry(config="XYZ")
    with pytest.raises(ValueError):
        Geometry(slit_um=60.0)  # slit > p_mask
    assert "N = 3" in Geometry(config="DF-DPC").summary()
