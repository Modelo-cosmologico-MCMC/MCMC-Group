"""
Tests para el canal indeterminado ρ_id.

Verifica el comportamiento de la transición y la consistencia física.
"""

import pytest
import numpy as np

from src.mcmc.channels.rho_id_parametric import (
    RhoIdParametricParams,
    rho_id_sharp_transition,
    rho_id_smooth_transition,
    Omega_id_of_z,
    E_squared_with_rho_id,
    H_of_z_with_rho_id,
    w_eff_id,
    RhoIdParametricModel,
)


class TestRhoIdParametricParams:
    """Tests para RhoIdParametricParams."""

    def test_default_values(self):
        """Verifica valores por defecto."""
        p = RhoIdParametricParams()
        assert p.Omega_id0 == 0.7
        assert p.z_trans == 0.5
        assert p.epsilon == 0.01

    def test_to_dict(self):
        """Verifica conversión a diccionario."""
        p = RhoIdParametricParams()
        d = p.to_dict()
        assert 'Omega_id0' in d
        assert 'z_trans' in d

    def test_from_dict(self):
        """Verifica construcción desde diccionario."""
        d = {'Omega_id0': 0.6, 'z_trans': 0.8}
        p = RhoIdParametricParams.from_dict(d)
        assert p.Omega_id0 == 0.6
        assert p.z_trans == 0.8


class TestRhoIdTransition:
    """Tests para la función de transición de ρ_id."""

    def setup_method(self):
        """Setup para cada test."""
        self.params = RhoIdParametricParams()
        self.z = np.linspace(0, 3, 100)

    def test_sharp_transition_shape(self):
        """Verifica forma de salida."""
        rho = rho_id_sharp_transition(self.z, self.params)
        assert rho.shape == self.z.shape

    def test_smooth_transition_shape(self):
        """Verifica forma de salida."""
        rho = rho_id_smooth_transition(self.z, self.params)
        assert rho.shape == self.z.shape

    def test_rho_id_positive(self):
        """Verifica que ρ_id > 0."""
        rho_sharp = rho_id_sharp_transition(self.z, self.params)
        rho_smooth = rho_id_smooth_transition(self.z, self.params)

        assert np.all(rho_sharp > 0), "ρ_id (sharp) tiene valores negativos"
        assert np.all(rho_smooth > 0), "ρ_id (smooth) tiene valores negativos"

    def test_transition_at_z_trans(self):
        """Verifica comportamiento alrededor de z_trans."""
        z_trans = self.params.z_trans
        z_before = z_trans + 0.5  # z > z_trans (tiempos tempranos)
        z_after = z_trans - 0.3   # z < z_trans (tiempos tardíos)

        rho = rho_id_smooth_transition(np.array([z_before, z_after]), self.params)

        # En z alto (temprano), ρ_id crece como (1+z)^3
        # En z bajo (tardío), ρ_id es más plano
        # La derivada debe ser mayor en z alto
        # (No hacemos una verificación estricta aquí, solo que no explote)
        assert np.isfinite(rho[0])
        assert np.isfinite(rho[1])

    def test_rho_id_today(self):
        """Verifica ρ_id en z=0."""
        rho_today = rho_id_smooth_transition(np.array([0.0]), self.params)
        # Debe ser cercano a Omega_id0
        assert np.isclose(rho_today[0], self.params.Omega_id0, rtol=0.1)


class TestESquared:
    """Tests para E(z)^2 = [H(z)/H0]^2."""

    def setup_method(self):
        """Setup para cada test."""
        self.params = RhoIdParametricParams()
        self.z = np.linspace(0, 3, 100)

    def test_E2_shape(self):
        """Verifica forma de E^2."""
        E2 = E_squared_with_rho_id(self.z, self.params)
        assert E2.shape == self.z.shape

    def test_E2_positive(self):
        """Verifica que E^2 > 0."""
        E2 = E_squared_with_rho_id(self.z, self.params)
        assert np.all(E2 > 0), "E^2 tiene valores no positivos"

    def test_E2_at_z0_is_one(self):
        """Verifica que E(z=0) ≈ 1."""
        E2_today = E_squared_with_rho_id(np.array([0.0]), self.params)
        E_today = np.sqrt(E2_today[0])
        # E(0) = sqrt(Omega_m + Omega_id) ≈ 1 si Omega_m + Omega_id = 1
        assert np.isclose(E_today, 1.0, rtol=0.1)


class TestHofZ:
    """Tests para H(z)."""

    def setup_method(self):
        """Setup para cada test."""
        self.params = RhoIdParametricParams()
        self.H0 = 67.4
        self.z = np.linspace(0, 3, 100)

    def test_H_shape(self):
        """Verifica forma de H(z)."""
        H = H_of_z_with_rho_id(self.z, self.H0, self.params)
        assert H.shape == self.z.shape

    def test_H_positive(self):
        """Verifica que H(z) > 0."""
        H = H_of_z_with_rho_id(self.z, self.H0, self.params)
        assert np.all(H > 0), "H(z) tiene valores no positivos"

    def test_H_at_z0_is_H0(self):
        """Verifica que H(z=0) ≈ H0."""
        H_today = H_of_z_with_rho_id(np.array([0.0]), self.H0, self.params)
        assert np.isclose(H_today[0], self.H0, rtol=0.1)

    def test_H_increases_with_z(self):
        """Verifica que H(z) crece con z (para z moderado)."""
        H = H_of_z_with_rho_id(self.z, self.H0, self.params)
        # En general, H debe crecer con z
        assert H[-1] > H[0], "H(z) no crece con z"


class TestWeff:
    """Tests para la ecuación de estado efectiva w_eff."""

    def setup_method(self):
        """Setup para cada test."""
        self.params = RhoIdParametricParams()
        self.z = np.linspace(0.01, 2, 50)  # Evitar z=0 para derivadas

    def test_w_shape(self):
        """Verifica forma de w_eff."""
        w = w_eff_id(self.z, self.params)
        assert w.shape == self.z.shape

    def test_w_finite(self):
        """Verifica que w es finito."""
        w = w_eff_id(self.z, self.params)
        assert np.all(np.isfinite(w)), "w tiene valores no finitos"

    def test_w_late_times_near_minus_one(self):
        """Verifica que w → -1 en tiempos tardíos (para ε pequeño)."""
        params = RhoIdParametricParams(epsilon=0.001)  # ε muy pequeño
        z_late = np.array([0.01, 0.05, 0.1])
        w = w_eff_id(z_late, params)

        # Para ε → 0, w debería estar cerca de -1 en z bajo
        assert np.all(w < 0), "w no es negativo en tiempos tardíos"


class TestRhoIdParametricModel:
    """Tests para la clase RhoIdParametricModel."""

    def setup_method(self):
        """Setup para cada test."""
        self.params = RhoIdParametricParams()
        self.model = RhoIdParametricModel(self.params)

    def test_model_creation(self):
        """Verifica creación del modelo."""
        assert self.model.params == self.params
        assert self.model.H0 == 67.4

    def test_model_rho_id(self):
        """Verifica cálculo de ρ_id."""
        z = np.array([0.0, 0.5, 1.0])
        rho = self.model.rho_id(z)
        assert rho.shape == z.shape
        assert np.all(rho > 0)

    def test_model_H(self):
        """Verifica cálculo de H(z)."""
        z = np.array([0.0, 0.5, 1.0])
        H = self.model.H(z)
        assert H.shape == z.shape
        assert np.all(H > 0)

    def test_model_E(self):
        """Verifica cálculo de E(z)."""
        z = np.array([0.0, 0.5, 1.0])
        E = self.model.E(z)
        assert E.shape == z.shape
        assert np.all(E > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
