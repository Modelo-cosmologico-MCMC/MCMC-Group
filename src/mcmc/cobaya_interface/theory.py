"""Wrapper cobaya.theory.Theory para el componente de fondo MCMC.

Requiere cobaya >= 3.6.2 (opcional; no es dependencia del paquete).
El motor numerico (BackgroundCalculator) no depende de cobaya, de modo que
todo el contenido fisico es testeable sin instalar cobaya.

Uso (con cobaya instalado):

    from mcmc.cobaya_interface.theory import get_theory_class
    MCMCBackground = get_theory_class()

    info = {
        "theory": {"mcmc_background": MCMCBackground},
        "params": {
            "H0": {"prior": {"min": 50, "max": 90}, "ref": 67.4},
            "rho_b0": {"prior": {"min": 0.05, "max": 1.0}, "ref": 0.30},
            "rho0": {"prior": {"min": 0.05, "max": 1.0}, "ref": 0.70},
            "z_trans": {"prior": {"min": 0.0, "max": 5.0}, "ref": 1.0},
            "eps": {"prior": {"min": 0.0, "max": 1.0}, "ref": 0.05},
        },
        "likelihood": {...},
        "sampler": {"mcmc": {}},
    }

Fase 1: solo fondo (H, distancias). Fase 2 (pendiente): espectros CMB.
"""
from __future__ import annotations

import numpy as np

from mcmc.cobaya_interface.background import BackgroundCalculator, BackgroundParams


def get_theory_class():
    """Construye la clase Theory de cobaya bajo demanda.

    Se define dentro de una funcion para que importar este modulo no
    requiera tener cobaya instalado.

    Returns:
        Subclase de cobaya.theory.Theory que publica el fondo MCMC.

    Raises:
        ImportError: si cobaya no esta instalado.
    """
    try:
        from cobaya.theory import Theory
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "cobaya no esta instalado. Instalar con: pip install 'cobaya>=3.6.2'"
        ) from exc

    class MCMCBackground(Theory):
        """Fondo MCMC (post-BB): H(z) y distancias para likelihoods BAO/SNe."""

        # Malla del calculador (configurable desde el yaml de cobaya)
        z_max: float = 5.0
        n_grid: int = 4001

        params = {
            "H0": None,
            "rho_b0": None,
            "rho0": None,
            "z_trans": None,
            "eps": None,
        }

        def initialize(self):
            self._calc = None
            self._z_pool = np.linspace(0.0, self.z_max, 500)

        def get_can_provide(self):
            return [
                "Hubble",
                "comoving_radial_distance",
                "angular_diameter_distance",
                "luminosity_distance",
            ]

        def get_can_provide_params(self):
            return ["Omega_m_eff", "w_eff_id_0"]

        def must_provide(self, **requirements):
            for key in ("Hubble", "comoving_radial_distance",
                        "angular_diameter_distance", "luminosity_distance"):
                req = requirements.get(key)
                if req and "z" in req:
                    z_req = np.atleast_1d(np.asarray(req["z"], dtype=float))
                    self._z_pool = np.union1d(self._z_pool, z_req)
            return {}

        def calculate(self, state, want_derived=True, **params_values_dict):
            p = BackgroundParams(
                H0=float(params_values_dict["H0"]),
                rho_b0=float(params_values_dict["rho_b0"]),
                rho0=float(params_values_dict["rho0"]),
                z_trans=float(params_values_dict["z_trans"]),
                eps=float(params_values_dict["eps"]),
            )
            try:
                calc = BackgroundCalculator(p, z_max=self.z_max, n_grid=self.n_grid)
            except ValueError:
                # Fondo invalido (p.ej. H^2 <= 0): punto rechazado
                return False

            z = self._z_pool
            state["Hubble"] = {"z": z, "value": calc.H(z)}
            state["comoving_radial_distance"] = {"z": z, "value": calc.D_C(z)}
            state["angular_diameter_distance"] = {"z": z, "value": calc.D_A(z)}
            state["luminosity_distance"] = {"z": z, "value": calc.D_L(z)}
            self._calc = calc

            if want_derived:
                total = p.rho_b0 + p.rho0
                state["derived"] = {
                    "Omega_m_eff": p.rho_b0 / total if total > 0 else np.nan,
                    "w_eff_id_0": float(calc.w_eff_id(0.0)[0]),
                }
            return True

        def _interp(self, key, z):
            data = self.current_state[key]
            return np.interp(np.atleast_1d(z), data["z"], data["value"])

        def get_Hubble(self, z, units="km/s/Mpc"):
            if units != "km/s/Mpc":
                raise ValueError("Solo units='km/s/Mpc' soportado en fase 1")
            return self._interp("Hubble", z)

        def get_comoving_radial_distance(self, z):
            return self._interp("comoving_radial_distance", z)

        def get_angular_diameter_distance(self, z):
            return self._interp("angular_diameter_distance", z)

        def get_luminosity_distance(self, z):
            return self._interp("luminosity_distance", z)

    return MCMCBackground
