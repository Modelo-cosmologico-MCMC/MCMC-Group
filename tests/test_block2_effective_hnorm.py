"""Tests para normalizacion H(0)=H0 del modelo efectivo Bloque II."""
from __future__ import annotations

import numpy as np
import pytest

from mcmc.channels.rho_id_refined import RhoIDRefinedParams
from mcmc.core.friedmann_effective import EffectiveParams, H_of_z, E_of_z, rho_total


class TestEffectiveNormalization:
    """Verifica que H(z=0) = H0 exactamente."""

    def test_effective_H_normalized_at_z0(self):
        """H(0) debe ser exactamente H0."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)

        H0_eval = H_of_z(np.array([0.0]), p)[0]

        assert np.isfinite(H0_eval)
        assert abs(H0_eval - 70.0) < 1e-10, f"H(0)={H0_eval} != H0=70.0"

    def test_effective_H_normalized_various_H0(self):
        """H(0) = H0 para varios valores de H0."""
        for H0_val in [50.0, 67.4, 70.0, 75.0, 80.0]:
            rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
            p = EffectiveParams(H0=H0_val, rho_b0=0.30, rho_id=rid)

            H0_eval = H_of_z(np.array([0.0]), p)[0]

            assert np.isfinite(H0_eval)
            assert abs(H0_eval - H0_val) < 1e-10, f"H(0)={H0_eval} != H0={H0_val}"

    def test_effective_E_normalized_at_z0(self):
        """E(0) debe ser 1.0."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)

        E0 = E_of_z(np.array([0.0]), p)[0]

        assert np.isfinite(E0)
        assert abs(E0 - 1.0) < 1e-10, f"E(0)={E0} != 1.0"

    def test_rho_total_positive(self):
        """rho_total debe ser positivo para todo z >= 0."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)

        z_arr = np.linspace(0.0, 3.0, 100)
        rho = rho_total(z_arr, p)

        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0), "rho_total debe ser > 0"

    def test_H_monotonic_increasing(self):
        """H(z) debe crecer con z (universo en expansion)."""
        rid = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        p = EffectiveParams(H0=70.0, rho_b0=0.30, rho_id=rid)

        z_arr = np.linspace(0.0, 2.0, 50)
        H_arr = H_of_z(z_arr, p)

        # H debe crecer (o al menos no decrecer significativamente)
        dH = np.diff(H_arr)
        # Permitimos pequenas fluctuaciones numericas
        assert np.all(dH >= -1e-8), "H(z) no debe decrecer"


class TestRhoIDRefined:
    """Tests para rho_id_refined."""

    def test_rho_id_positive(self):
        """rho_id debe ser positivo."""
        from mcmc.channels.rho_id_refined import rho_id_refined

        p = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)
        z_arr = np.linspace(0.0, 3.0, 100)
        rho = rho_id_refined(z_arr, p)

        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)

    def test_rho_id_transition(self):
        """Verifica comportamiento en transicion z_trans."""
        from mcmc.channels.rho_id_refined import rho_id_refined

        p = RhoIDRefinedParams(rho0=0.70, z_trans=1.0, eps=0.05)

        # Debajo de z_trans: regimen efectivo
        z_low = np.array([0.0, 0.5])
        rho_low = rho_id_refined(z_low, p)
        assert np.all(rho_low > 0)

        # Arriba de z_trans: regimen (1+z)^3
        z_high = np.array([1.5, 2.0, 3.0])
        rho_high = rho_id_refined(z_high, p)
        expected = p.rho0 * (1.0 + z_high) ** 3
        np.testing.assert_allclose(rho_high, expected, rtol=1e-10)

    def test_rho_id_scalar_input(self):
        """Acepta escalar como entrada."""
        from mcmc.channels.rho_id_refined import rho_id_refined

        p = RhoIDRefinedParams()
        result = rho_id_refined(0.5, p)
        assert np.isfinite(result)
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
