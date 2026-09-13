"""Referencia LCDM independiente para verificar el componente de fondo.

Implementacion deliberadamente separada del motor MCMC:
- E(z) analitico de LCDM plano
- distancias por cuadratura adaptativa (scipy.integrate.quad) punto a punto

No comparte codigo de integracion con BackgroundCalculator, de modo que la
comparacion en tests es una verificacion cruzada real y no una tautologia.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

C_LIGHT = 299792.458  # km/s


@dataclass(frozen=True)
class LCDMReference:
    """LCDM plano de referencia.

    Attributes:
        H0: Constante de Hubble [km/s/Mpc]
        Omega_m: Fraccion de materia hoy (Omega_Lambda = 1 - Omega_m)
    """
    H0: float = 67.4
    Omega_m: float = 0.315

    def E(self, z) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return np.sqrt(self.Omega_m * (1.0 + z) ** 3 + (1.0 - self.Omega_m))

    def H(self, z) -> np.ndarray:
        return self.H0 * self.E(z)

    def D_C(self, z) -> np.ndarray:
        """Distancia comovil radial por cuadratura adaptativa [Mpc]."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.empty_like(z)
        for i, zi in enumerate(z):
            val, _ = quad(lambda x: 1.0 / float(self.E(x)[0]), 0.0, zi,
                          epsabs=1e-10, epsrel=1e-10, limit=200)
            out[i] = (C_LIGHT / self.H0) * val
        return out

    def D_M(self, z) -> np.ndarray:
        return self.D_C(z)

    def D_H(self, z) -> np.ndarray:
        return C_LIGHT / self.H(z)

    def D_L(self, z) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return (1.0 + z) * self.D_M(z)

    def D_A(self, z) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return self.D_M(z) / (1.0 + z)

    def D_V(self, z) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        term = self.D_M(z) ** 2 * C_LIGHT * z / self.H(z)
        return np.maximum(term, 0.0) ** (1.0 / 3.0)

    def mu(self, z, M: float = -19.3) -> np.ndarray:
        dL = np.maximum(self.D_L(z), 1e-30)
        return 5.0 * np.log10(dL) + 25.0 + (M + 19.3)
