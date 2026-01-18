"""
Tests de regresión para observables cosmológicos.

Verifica que los cálculos de distancias, BAO, H(z) y SNe
producen resultados consistentes.
"""

import pytest
import numpy as np

from src.mcmc.observables.distances import (
    C_LIGHT,
    comoving_distance,
    angular_diameter_distance,
    luminosity_distance,
    distance_modulus,
    DistanceCalculator,
    H_lcdm,
    get_lcdm_distances,
)
from src.mcmc.observables.bao import (
    D_H, D_M, D_V,
    compute_bao_observables,
    get_boss_dr12_data,
    get_combined_bao_data,
)
from src.mcmc.observables.hz import (
    chi2_Hz,
    get_cosmic_chronometers_data,
    H_lcdm as H_lcdm_hz,
)
from src.mcmc.observables.sne import (
    mu_from_dL,
    chi2_sne_simple,
    get_pantheon_binned_data,
)


class TestDistances:
    """Tests para cálculos de distancias."""

    def setup_method(self):
        """Setup para cada test."""
        self.H0 = 67.4
        self.Omega_m = 0.3
        self.z = np.array([0.1, 0.5, 1.0, 2.0])

    def test_H_lcdm_at_z0(self):
        """Verifica H(z=0) = H0 para ΛCDM."""
        H0_calc = H_lcdm(0.0, self.H0, self.Omega_m)
        assert np.isclose(H0_calc, self.H0)

    def test_H_lcdm_increases(self):
        """Verifica que H(z) crece con z."""
        H = np.array([H_lcdm(z, self.H0, self.Omega_m) for z in self.z])
        assert np.all(np.diff(H) > 0), "H(z) debe crecer con z"

    def test_comoving_distance_positive(self):
        """Verifica d_C > 0 para z > 0."""
        def H_func(z):
            return H_lcdm(z, self.H0, self.Omega_m)

        d_C = comoving_distance(self.z, H_func)
        assert np.all(d_C > 0), "d_C debe ser positivo"

    def test_comoving_distance_monotonic(self):
        """Verifica que d_C crece con z."""
        def H_func(z):
            return H_lcdm(z, self.H0, self.Omega_m)

        d_C = comoving_distance(self.z, H_func)
        assert np.all(np.diff(d_C) > 0), "d_C debe crecer con z"

    def test_angular_diameter_distance(self):
        """Verifica d_A = d_C / (1+z)."""
        d_C = np.array([100, 500, 1000, 1500])  # Mpc arbitrarios
        d_A = angular_diameter_distance(self.z, d_C)

        expected = d_C / (1 + self.z)
        np.testing.assert_array_almost_equal(d_A, expected)

    def test_luminosity_distance(self):
        """Verifica d_L = (1+z) * d_C."""
        d_C = np.array([100, 500, 1000, 1500])
        d_L = luminosity_distance(self.z, d_C)

        expected = (1 + self.z) * d_C
        np.testing.assert_array_almost_equal(d_L, expected)

    def test_distance_modulus(self):
        """Verifica μ = 5 log10(d_L) + 25."""
        d_L = np.array([10, 100, 1000])  # Mpc
        mu = distance_modulus(d_L)

        expected = 5 * np.log10(d_L) + 25
        np.testing.assert_array_almost_equal(mu, expected)

    def test_get_lcdm_distances(self):
        """Verifica función de conveniencia para ΛCDM."""
        result = get_lcdm_distances(self.z, self.H0, self.Omega_m)

        assert 'z' in result
        assert 'd_C' in result
        assert 'd_A' in result
        assert 'd_L' in result
        assert 'mu' in result

        # Verificar relaciones
        np.testing.assert_array_almost_equal(
            result['d_A'],
            result['d_C'] / (1 + result['z'])
        )


class TestDistanceCalculator:
    """Tests para DistanceCalculator."""

    def setup_method(self):
        """Setup para cada test."""
        self.H0 = 67.4
        self.Omega_m = 0.3

        def H_func(z):
            return H_lcdm(z, self.H0, self.Omega_m)

        self.calc = DistanceCalculator(H_func=H_func, H0=self.H0)

    def test_d_C(self):
        """Verifica cálculo de d_C."""
        d_C = self.calc.d_C(0.5)
        assert d_C > 0

    def test_d_A(self):
        """Verifica cálculo de d_A."""
        d_A = self.calc.d_A(0.5)
        assert d_A > 0

    def test_d_L(self):
        """Verifica cálculo de d_L."""
        d_L = self.calc.d_L(0.5)
        assert d_L > 0

    def test_mu(self):
        """Verifica cálculo de μ."""
        mu = self.calc.mu(0.5)
        assert np.isfinite(mu)


class TestBAO:
    """Tests para observables BAO."""

    def setup_method(self):
        """Setup para cada test."""
        self.z = 0.5
        self.d_C = 1500  # Mpc
        self.H = 80  # km/s/Mpc
        self.r_d = 147.09  # Mpc

    def test_D_H(self):
        """Verifica D_H = c/H."""
        D_H_val = D_H(self.z, self.H)
        expected = C_LIGHT / self.H
        assert np.isclose(D_H_val, expected)

    def test_D_M(self):
        """Verifica D_M = d_C (universo plano)."""
        D_M_val = D_M(self.z, self.d_C)
        assert D_M_val == self.d_C

    def test_D_V(self):
        """Verifica D_V = [z * D_M^2 * D_H]^(1/3)."""
        D_V_val = D_V(self.z, self.d_C, self.H)
        D_H_val = C_LIGHT / self.H
        expected = (self.z * self.d_C**2 * D_H_val) ** (1/3)
        assert np.isclose(D_V_val, expected)

    def test_compute_bao_observables(self):
        """Verifica función de cálculo de observables BAO."""
        z = np.array([0.38, 0.51, 0.61])
        d_C = np.array([1000, 1300, 1500])
        H = np.array([80, 85, 90])

        result = compute_bao_observables(z, d_C, H, self.r_d)

        assert 'D_V' in result
        assert 'D_M' in result
        assert 'D_H' in result
        assert 'D_V/r_d' in result

    def test_boss_data_loading(self):
        """Verifica carga de datos BOSS."""
        data = get_boss_dr12_data()
        assert data.n_points > 0
        assert len(data.z_values) > 0

    def test_combined_bao_data(self):
        """Verifica datos BAO combinados."""
        data = get_combined_bao_data()
        assert data.n_points > 0


class TestHz:
    """Tests para H(z)."""

    def test_chi2_Hz(self):
        """Verifica cálculo de χ² para H(z)."""
        H_model = np.array([70, 80, 90])
        H_data = np.array([72, 78, 92])
        sigma = np.array([5, 5, 5])

        chi2 = chi2_Hz(H_model, H_data, sigma)
        expected = np.sum(((H_model - H_data) / sigma)**2)

        assert np.isclose(chi2, expected)

    def test_cosmic_chronometers_data(self):
        """Verifica carga de datos de cosmic chronometers."""
        data = get_cosmic_chronometers_data()
        assert data.n_points > 0
        assert len(data.z_values) == len(data.H_values)


class TestSNe:
    """Tests para supernovas."""

    def test_mu_from_dL(self):
        """Verifica μ desde d_L."""
        d_L = np.array([100, 1000])  # Mpc
        mu = mu_from_dL(d_L)

        expected = 5 * np.log10(d_L) + 25
        np.testing.assert_array_almost_equal(mu, expected)

    def test_chi2_sne_simple(self):
        """Verifica χ² simple para SNe."""
        mu_model = np.array([35, 40, 42])
        mu_data = np.array([35.1, 39.8, 42.2])
        sigma = np.array([0.1, 0.1, 0.1])

        chi2 = chi2_sne_simple(mu_model, mu_data, sigma)
        expected = np.sum(((mu_model - mu_data) / sigma)**2)

        assert np.isclose(chi2, expected)

    def test_pantheon_binned_data(self):
        """Verifica carga de datos Pantheon binneados."""
        data = get_pantheon_binned_data()
        assert data.n_sne > 0


class TestRegressionLCDM:
    """
    Tests de regresión: verificar que ΛCDM produce valores conocidos.
    """

    def setup_method(self):
        """Setup para cada test."""
        # Parámetros Planck 2018
        self.H0 = 67.4
        self.Omega_m = 0.315

    def test_d_L_at_z1_order_of_magnitude(self):
        """Verifica orden de magnitud de d_L(z=1)."""
        result = get_lcdm_distances(np.array([1.0]), self.H0, self.Omega_m)
        d_L = result['d_L'][0]

        # d_L(z=1) para ΛCDM Planck es ~6700 Mpc
        assert 6000 < d_L < 7500, f"d_L(z=1) = {d_L} Mpc fuera de rango esperado"

    def test_mu_at_z1_order_of_magnitude(self):
        """Verifica orden de magnitud de μ(z=1)."""
        result = get_lcdm_distances(np.array([1.0]), self.H0, self.Omega_m)
        mu = result['mu'][0]

        # μ(z=1) para ΛCDM Planck es ~44
        assert 43 < mu < 45, f"μ(z=1) = {mu} fuera de rango esperado"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
