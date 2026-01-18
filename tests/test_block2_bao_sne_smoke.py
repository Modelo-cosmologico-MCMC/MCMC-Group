"""Smoke tests para observables BAO y SNe del Bloque II."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.channels.rho_id_refined import RhoIDRefinedParams
from mcmc.core.friedmann_effective import EffectiveParams, H_of_z
from mcmc.observables.bao_distances import (
    angular_diameter_distance,
    volume_distance,
    dv_over_rd,
    da_over_rd,
)
from mcmc.observables.distances import luminosity_distance, distance_modulus


class TestBAODistances:
    """Tests para distancias BAO."""

    @pytest.fixture
    def setup_model(self):
        """Prepara modelo efectivo para tests."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)
        z_arr = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        H_arr = H_of_z(z_arr, p)
        return z_arr, H_arr

    def test_angular_diameter_distance_finite(self, setup_model):
        """D_A(z) debe ser finito y positivo (excepto primer punto de integracion)."""
        z_arr, H_arr = setup_model
        DA = angular_diameter_distance(z_arr, H_arr)

        assert np.all(np.isfinite(DA)), "D_A contiene valores no finitos"
        # El primer punto es 0 por construccion de cumulative_trapezoid
        assert np.all(DA[1:] > 0), "D_A debe ser positivo para z > z_min"

    def test_volume_distance_finite(self, setup_model):
        """D_V(z) debe ser finito y positivo."""
        z_arr, H_arr = setup_model
        DV = volume_distance(z_arr, H_arr)

        assert np.all(np.isfinite(DV)), "D_V contiene valores no finitos"
        assert np.all(DV > 0), "D_V debe ser positivo"

    def test_dv_over_rd_finite(self, setup_model):
        """D_V/r_d debe ser finito y positivo."""
        z_arr, H_arr = setup_model
        rd = 147.0  # Mpc (tipico)
        dvrd = dv_over_rd(z_arr, H_arr, rd)

        assert np.all(np.isfinite(dvrd)), "D_V/r_d contiene valores no finitos"
        assert np.all(dvrd > 0), "D_V/r_d debe ser positivo"

    def test_da_over_rd_finite(self, setup_model):
        """D_A/r_d debe ser finito y positivo (excepto primer punto)."""
        z_arr, H_arr = setup_model
        rd = 147.0
        dard = da_over_rd(z_arr, H_arr, rd)

        assert np.all(np.isfinite(dard)), "D_A/r_d contiene valores no finitos"
        # El primer punto es 0 por construccion de cumulative_trapezoid
        assert np.all(dard[1:] > 0), "D_A/r_d debe ser positivo para z > z_min"

    def test_dv_over_rd_invalid_rd(self, setup_model):
        """rd <= 0 debe lanzar ValueError."""
        z_arr, H_arr = setup_model

        with pytest.raises(ValueError, match="rd debe ser > 0"):
            dv_over_rd(z_arr, H_arr, rd=0.0)

        with pytest.raises(ValueError, match="rd debe ser > 0"):
            dv_over_rd(z_arr, H_arr, rd=-10.0)

    def test_da_over_rd_invalid_rd(self, setup_model):
        """rd <= 0 debe lanzar ValueError."""
        z_arr, H_arr = setup_model

        with pytest.raises(ValueError, match="rd debe ser > 0"):
            da_over_rd(z_arr, H_arr, rd=0.0)


class TestSNeDistances:
    """Tests para distancias SNe."""

    @pytest.fixture
    def setup_model(self):
        """Prepara modelo efectivo para tests."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)
        z_arr = np.array([0.01, 0.1, 0.5, 1.0, 1.5])
        H_arr = H_of_z(z_arr, p)
        return z_arr, H_arr

    def test_luminosity_distance_finite(self, setup_model):
        """D_L(z) debe ser finito y positivo (excepto primer punto)."""
        z_arr, H_arr = setup_model
        DL = luminosity_distance(z_arr, H_arr)

        assert np.all(np.isfinite(DL)), "D_L contiene valores no finitos"
        # El primer punto es 0 por construccion de cumulative_trapezoid
        assert np.all(DL[1:] > 0), "D_L debe ser positivo para z > z_min"

    def test_distance_modulus_finite(self, setup_model):
        """mu(z) debe ser finito."""
        z_arr, H_arr = setup_model
        M = -19.3  # Magnitud absoluta tipica
        mu = distance_modulus(z_arr, H_arr, M=M)

        assert np.all(np.isfinite(mu)), "mu contiene valores no finitos"

    def test_distance_modulus_increases_with_z(self, setup_model):
        """mu(z) debe crecer con z."""
        z_arr, H_arr = setup_model
        M = -19.3
        mu = distance_modulus(z_arr, H_arr, M=M)

        # mu debe ser monotonamente creciente
        dmu = np.diff(mu)
        assert np.all(dmu > 0), "mu(z) debe crecer con z"


class TestCombinedObservables:
    """Tests combinados de observables."""

    def test_full_observable_chain(self):
        """Cadena completa: params -> H(z) -> observables."""
        # Parametros
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)

        # Redshifts de prueba
        z_hz = np.array([0.1, 0.3, 0.5, 0.7])
        z_sne = np.array([0.01, 0.1, 0.5, 1.0])
        z_bao = np.array([0.38, 0.51, 0.70])

        # Evaluar H(z)
        H_hz = H_of_z(z_hz, p)
        H_sne = H_of_z(z_sne, p)
        H_bao = H_of_z(z_bao, p)

        # Observables
        mu = distance_modulus(z_sne, H_sne, M=-19.3)
        dvrd = dv_over_rd(z_bao, H_bao, rd=147.0)

        # Todos deben ser finitos
        assert np.all(np.isfinite(H_hz)), "H(z) no finito"
        # El primer punto de mu puede ser -inf por D_L=0 (primer punto de integracion)
        assert np.all(np.isfinite(mu[1:])), "mu(z) no finito para z > z_min"
        assert np.all(np.isfinite(dvrd)), "DV/rd no finito"

        # Valores razonables (excluyendo primer punto por integracion acumulativa)
        assert np.all(H_hz > 60) and np.all(H_hz < 500), "H(z) fuera de rango"
        assert np.all(mu[1:] > 30) and np.all(mu[1:] < 50), "mu(z) fuera de rango"
        assert np.all(dvrd[1:] > 1) and np.all(dvrd[1:] < 50), "DV/rd fuera de rango"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
