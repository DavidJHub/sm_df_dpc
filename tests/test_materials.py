import numpy as np
import pytest

from sm_df_dpc import material_library
from sm_df_dpc.materials import Material


def test_biomineral_materials_present():
    mats = material_library(40.0)
    assert "hydroxyapatite" in mats
    assert "calcium_oxalate" in mats


def test_biominerals_more_attenuating_than_water():
    mats = material_library(40.0)
    # Minerales con Ca: mayor δ y μ que el agua / tejido blando
    for name in ("hydroxyapatite", "calcium_oxalate"):
        assert mats[name].delta > mats["water"].delta
        assert mats[name].mu_um > mats["water"].mu_um
    # La hidroxiapatita (mineral óseo) es más densa y atenuante que el oxalato
    assert mats["hydroxyapatite"].delta > mats["calcium_oxalate"].delta
    assert mats["hydroxyapatite"].mu_um > mats["calcium_oxalate"].mu_um


def test_biominerals_have_darkfield_microstructure():
    mats = material_library(40.0)
    assert mats["hydroxyapatite"].usaxs_urad2_per_um > 0
    assert mats["calcium_oxalate"].usaxs_urad2_per_um > 0


def test_delta_scales_as_inverse_energy_squared():
    m40 = material_library(40.0)["hydroxyapatite"]
    m80 = material_library(80.0)["hydroxyapatite"]
    # δ ∝ λ² ∝ 1/E²  →  δ(40)/δ(80) = 4
    assert m40.delta / m80.delta == pytest.approx(4.0, rel=1e-6)


def test_mu_scales_as_inverse_energy_cubed():
    m40 = material_library(40.0)["calcium_oxalate"]
    m80 = material_library(80.0)["calcium_oxalate"]
    # μ ∝ (40/E)³  →  μ(40)/μ(80) = 8
    assert m40.mu_um / m80.mu_um == pytest.approx(8.0, rel=1e-6)


def test_from_electron_density_roundtrip():
    # δ = r_e λ² ρ_e / (2π); verificamos contra el cálculo manual del agua
    m = material_library(40.0)["water"]
    lam = 1.23984193e-3 / 40.0
    r_e, rho_e = 2.8179403262e-9, 3.34e29 * 1e-18
    expected = r_e * lam**2 * rho_e / (2 * np.pi)
    assert m.delta == pytest.approx(expected, rel=1e-9)
