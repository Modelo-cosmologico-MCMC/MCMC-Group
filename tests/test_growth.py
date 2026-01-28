"""Tests para el módulo growth.

CORRECCIÓN ONTOLÓGICA (2025): S ∈ [0, 100]
- Pre-geométrico: S ∈ [0, 1.001)
- Post-Big Bang: S ∈ [1.001, 95.07]
"""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.growth.linear_growth import (
    GrowthParams,
    LinearGrowthSolver,
    D_of_z_LCDM,
    f_of_z_LCDM,
)
from mcmc.growth.f_sigma8 import (
    FSigma8Data,
    FSigma8Predictor,
    S8_parameter,
    get_BOSS_data,
    get_6dFGS_data,
    get_combined_RSD_data,
    compare_growth,
)
from mcmc.growth.mu_eta import (
    MuEtaParams,
    MuEtaFromS,
    mu_CPL,
    eta_CPL,
    Sigma_lensing,
    Upsilon_RSD,
    solve_growth_modified,
    compare_modified_gravity,
)


class TestLinearGrowthSolver:
    """Tests para el solver de crecimiento lineal."""

    @pytest.fixture
    def solver(self):
        """Solver con parámetros ΛCDM."""
        params = GrowthParams(Omega_m0=0.315, sigma8_0=0.811)
        return LinearGrowthSolver(params)

    def test_D_normalized_at_z0(self, solver):
        """D(z=0) = 1."""
        z, D = solver.solve_growth_z((0, 2), 100)
        D_z0 = D[0]  # z=0 es el primer punto
        np.testing.assert_allclose(D_z0, 1.0, rtol=1e-4)

    def test_D_decreases_with_z(self, solver):
        """D(z) decrece con z."""
        z, D = solver.solve_growth_z((0, 2), 50)
        assert np.all(np.diff(D) <= 0)

    def test_f_positive(self, solver):
        """f = d ln D / d ln a > 0."""
        z = np.array([0.0, 0.5, 1.0, 2.0])
        f = solver.f_of_z(z)
        assert np.all(f > 0)

    def test_f_finite(self, solver):
        """f es finito en todo el rango."""
        z = np.array([0.0, 0.5, 1.0, 2.0])
        f = solver.f_of_z(z)
        assert np.all(np.isfinite(f))


class TestDofzLCDM:
    """Tests para D(z) ΛCDM de referencia."""

    def test_D_LCDM_normalized(self):
        """D_LCDM(z=0) = 1."""
        D = D_of_z_LCDM(0.0, Omega_m0=0.315)
        np.testing.assert_allclose(D, 1.0, rtol=1e-6)

    def test_D_LCDM_high_z(self):
        """D_LCDM(z>>1) → 0."""
        D = D_of_z_LCDM(100.0, Omega_m0=0.315)
        assert D < 0.1

    def test_f_LCDM_reasonable(self):
        """f_LCDM en rango razonable."""
        z = np.array([0.0, 1.0, 2.0])
        f = f_of_z_LCDM(z, Omega_m0=0.315)
        assert np.all(f > 0.3)
        assert np.all(f < 1.0)


class TestFSigma8:
    """Tests para fσ₈(z)."""

    @pytest.fixture
    def predictor(self):
        """Predictor con parámetros ΛCDM."""
        params = GrowthParams(Omega_m0=0.315, sigma8_0=0.811)
        return FSigma8Predictor(params=params)

    def test_f_sigma8_finite(self, predictor):
        """fσ₈(z) es finito."""
        z = np.array([0.0, 0.5, 1.0])
        fs8 = predictor.f_sigma8(z)
        assert np.all(np.isfinite(fs8))

    def test_sigma8_of_z_decreases(self, predictor):
        """σ₈(z) decrece con z."""
        z = np.array([0.0, 0.5, 1.0, 2.0])
        sigma8 = predictor.sigma8_of_z(z)
        assert np.all(np.diff(sigma8) < 0)

    def test_chi2_reasonable(self, predictor):
        """χ² con datos sintéticos."""
        data = FSigma8Data(
            z=np.array([0.5]),
            f_sigma8=np.array([0.45]),
            sigma=np.array([0.05]),
            survey="test"
        )
        chi2 = predictor.chi2(data)
        assert chi2 > 0

    def test_S8_parameter(self):
        """S₈ = σ₈ (Ω_m/0.3)^0.5."""
        sigma8 = 0.811
        Omega_m = 0.315
        S8 = S8_parameter(sigma8, Omega_m)
        expected = sigma8 * (Omega_m / 0.3) ** 0.5
        np.testing.assert_allclose(S8, expected)


class TestObservationalData:
    """Tests para datos observacionales."""

    def test_BOSS_data_shape(self):
        """BOSS tiene 3 puntos."""
        data = get_BOSS_data()
        assert len(data.z) == 3
        assert len(data.f_sigma8) == 3
        assert len(data.sigma) == 3

    def test_6dFGS_data_shape(self):
        """6dFGS tiene 1 punto."""
        data = get_6dFGS_data()
        assert len(data.z) == 1

    def test_combined_data_shape(self):
        """Datos combinados tienen 4 puntos."""
        data = get_combined_RSD_data()
        assert len(data.z) == 4

    def test_data_values_positive(self):
        """fσ₈ y σ positivos."""
        data = get_combined_RSD_data()
        assert np.all(data.f_sigma8 > 0)
        assert np.all(data.sigma > 0)


class TestMuEta:
    """Tests para gravedad modificada μ(a), η(a)."""

    @pytest.fixture
    def params(self):
        """Parámetros por defecto (GR)."""
        return MuEtaParams(mu_0=1.0, mu_a=0.0, eta_0=1.0, eta_a=0.0)

    def test_mu_GR(self, params):
        """μ = 1 para GR."""
        a = np.array([0.5, 1.0])
        mu = mu_CPL(a, params)
        np.testing.assert_allclose(mu, 1.0)

    def test_eta_GR(self, params):
        """η = 1 para GR."""
        a = np.array([0.5, 1.0])
        eta = eta_CPL(a, params)
        np.testing.assert_allclose(eta, 1.0)

    def test_mu_CPL_deviation(self):
        """μ CPL con desviación."""
        params = MuEtaParams(mu_0=1.1, mu_a=0.1)
        mu_today = mu_CPL(1.0, params)
        mu_past = mu_CPL(0.5, params)

        np.testing.assert_allclose(mu_today, 1.1)
        np.testing.assert_allclose(mu_past, 1.15)

    def test_Sigma_lensing(self):
        """Σ = μ(1+η)/2."""
        mu = 1.1
        eta = 0.9
        Sigma = Sigma_lensing(mu, eta)
        expected = 1.1 * (1 + 0.9) / 2
        np.testing.assert_allclose(Sigma, expected)

    def test_Upsilon_RSD(self):
        """Υ = μ·f."""
        mu = 1.1
        f = 0.5
        Y = Upsilon_RSD(mu, f)
        np.testing.assert_allclose(Y, 1.1 * 0.5)


class TestMuEtaFromS:
    """Tests para μ, η desde el mapa entrópico.

    CORRECCIÓN: S ∈ [0, 100], post-Big Bang S ∈ [1.001, 95.07]
    """

    @pytest.fixture
    def mu_eta_S(self):
        """MuEtaFromS con parámetros pequeños."""
        return MuEtaFromS(alpha_mu=0.01, alpha_eta=0.01)

    def test_mu_of_S_near_one(self, mu_eta_S):
        """μ(S) ≈ 1 para α pequeño.

        CORRECCIÓN: S ∈ [1.001, 95.07] post-Big Bang
        """
        S = np.array([10.0, 50.0, 90.0])
        mu = mu_eta_S.mu_of_S(S)
        np.testing.assert_allclose(mu, 1.0, atol=0.1)

    def test_eta_of_S_near_one(self, mu_eta_S):
        """η(S) ≈ 1 lejos de transiciones.

        CORRECCIÓN: S ∈ [1.001, 95.07], transiciones en S_BB=1.001 y S_peak=47.5
        """
        S = np.array([20.0, 80.0])  # Lejos de S_BB y S_peak
        eta = mu_eta_S.eta_of_S(S)
        np.testing.assert_allclose(eta, 1.0, atol=0.1)


class TestGrowthModified:
    """Tests para crecimiento con gravedad modificada."""

    def test_solve_growth_GR(self):
        """Crecimiento GR."""
        a, D, f = solve_growth_modified(
            a_range=(0.01, 1.0),
            params=__import__('mcmc.growth.mu_eta', fromlist=['PerturbationParams']).PerturbationParams(),
            n_points=100
        )
        # D(a=1) = 1 por normalización
        np.testing.assert_allclose(D[-1], 1.0, rtol=1e-4)

    def test_compare_modified_gravity(self):
        """Comparación mod vs GR."""
        params = MuEtaParams(mu_0=1.0, mu_a=0.0)
        result = compare_modified_gravity(params, n_points=50)

        # Para GR, D_mod = D_GR
        np.testing.assert_allclose(result.D_mod, result.D_GR, rtol=0.05)


class TestCompareGrowth:
    """Tests para comparación MCMC vs ΛCDM."""

    def test_compare_growth_output(self):
        """compare_growth retorna estructura correcta."""
        params = GrowthParams()
        result = compare_growth(params, n_points=50)

        assert len(result.z) == 50
        assert len(result.f_sigma8_mcmc) == 50
        assert len(result.f_sigma8_lcdm) == 50
        assert len(result.ratio) == 50

    def test_compare_growth_ratio_finite(self):
        """Ratio es finito."""
        params = GrowthParams()
        result = compare_growth(params, n_points=50)

        # Ratio debe ser finito
        assert np.all(np.isfinite(result.ratio))
