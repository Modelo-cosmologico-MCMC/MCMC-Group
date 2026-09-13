"""Componente de fondo del MCMC (regimen post-BB) con salida completa de distancias.

Contrato (Prioridad 1 del plan de validacion):
  - Entrada: parametros efectivos MCMC (H0, rho_b0, rho0, z_trans, eps, rd, M).
  - Salida: H(z), E(z), D_C, D_M, D_H, D_A, D_L, D_V, mu(z), DV/rd, DM/rd, DH/rd
    y w_eff(z) del canal indeterminado.
  - Limite LCDM verificable: con eps = 0 y z_trans >= z_max el canal
    rho_id(z) es exactamente constante, de modo que

        E^2(z) = [rho_b0 (1+z)^3 + rho0] / [rho_b0 + rho0]

    coincide con LCDM plano de Omega_m = rho_b0 / (rho_b0 + rho0).

Nota de degeneracion: por la normalizacion H(0) = H0, solo la razon
rho_b0 : rho0 es observable en el fondo; una reescala conjunta de ambas
densidades deja H(z) invariante. Ver docs/implementation/cobaya_background.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid

from mcmc.channels.rho_id_refined import RhoIDRefinedParams, rho_id_refined, drho_id_dz
from mcmc.core.friedmann_effective import EffectiveParams, H_of_z

C_LIGHT = 299792.458  # km/s


@dataclass(frozen=True)
class BackgroundParams:
    """Parametros de entrada del componente de fondo.

    Attributes:
        H0: Constante de Hubble [km/s/Mpc]
        rho_b0: Coeficiente efectivo de materia agrupada
        rho0: Amplitud del canal indeterminado rho_id
        z_trans: Redshift de transicion de rho_id
        eps: Pendiente tardia de rho_id (eps=0 -> vacio constante)
        rd: Horizonte de sonido en drag epoch [Mpc] (nuisance BAO)
        M: Magnitud absoluta SNe (nuisance, degenerada con H0)
    """
    H0: float = 67.4
    rho_b0: float = 0.30
    rho0: float = 0.70
    z_trans: float = 1.0
    eps: float = 0.05
    rd: float = 147.0
    M: float = -19.3


class BackgroundCalculator:
    """Motor de fondo: H(z) + distancias sobre una malla densa interpolada.

    Todas las distancias se obtienen integrando c/H(z) con trapecio
    acumulado sobre una malla densa en [0, z_max] y evaluando por
    interpolacion lineal. Universo plano: D_M = D_C.
    """

    def __init__(
        self,
        params: BackgroundParams,
        *,
        z_max: float = 5.0,
        n_grid: int = 4001,
    ):
        if z_max <= 0:
            raise ValueError("z_max debe ser > 0")
        if n_grid < 100:
            raise ValueError("n_grid demasiado pequeno (>= 100)")

        self.params = params
        self.z_max = float(z_max)
        self._z_grid = np.linspace(0.0, self.z_max, int(n_grid))

        self._eff = EffectiveParams(
            H0=params.H0,
            rho_b0=params.rho_b0,
            rho_id=RhoIDRefinedParams(
                rho0=params.rho0, z_trans=params.z_trans, eps=params.eps
            ),
        )
        self._H_grid = H_of_z(self._z_grid, self._eff)
        self._dc_grid = cumulative_trapezoid(
            C_LIGHT / self._H_grid, self._z_grid, initial=0.0
        )

        self.validate()

    # ------------------------------------------------------------------
    # Constructores alternativos
    # ------------------------------------------------------------------
    @classmethod
    def lcdm(
        cls,
        H0: float = 67.4,
        Omega_m: float = 0.315,
        *,
        rd: float = 147.0,
        M: float = -19.3,
        z_max: float = 5.0,
        n_grid: int = 4001,
    ) -> "BackgroundCalculator":
        """Limite LCDM exacto del componente MCMC.

        Se construye con eps=0 y z_trans > z_max, de modo que rho_id es
        constante en toda la malla y E^2(z) = Omega_m (1+z)^3 + (1-Omega_m).
        """
        if not (0.0 < Omega_m < 1.0):
            raise ValueError("Omega_m debe estar en (0, 1)")
        p = BackgroundParams(
            H0=H0,
            rho_b0=Omega_m,
            rho0=1.0 - Omega_m,
            z_trans=z_max + 1.0,
            eps=0.0,
            rd=rd,
            M=M,
        )
        return cls(p, z_max=z_max, n_grid=n_grid)

    # ------------------------------------------------------------------
    # Validacion / estabilidad
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Invariantes de estabilidad del fondo.

        - H(z) finito y H^2(z) > 0 en toda la malla
        - Normalizacion H(0) = H0 exacta
        - D_C(z) estrictamente creciente
        """
        if not np.isfinite(self._H_grid).all():
            raise ValueError("H(z) contiene NaN/Inf en la malla")
        if np.any(self._H_grid <= 0):
            raise ValueError("H^2(z) > 0 violado: H(z) <= 0 en la malla")
        if abs(self._H_grid[0] - self.params.H0) > 1e-8 * self.params.H0:
            raise ValueError(
                f"Normalizacion rota: H(0)={self._H_grid[0]} != H0={self.params.H0}"
            )
        if np.any(np.diff(self._dc_grid) < 0):
            raise ValueError("D_C(z) no es monotona creciente")

    # ------------------------------------------------------------------
    # Nucleo de evaluacion
    # ------------------------------------------------------------------
    def _check_range(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if np.any(z < 0):
            raise ValueError("Se requiere z >= 0 (regimen post-BB)")
        if np.any(z > self.z_max):
            raise ValueError(
                f"z fuera de la malla: max(z)={z.max():.4f} > z_max={self.z_max}"
            )
        return z

    def H(self, z) -> np.ndarray:
        """H(z) [km/s/Mpc]."""
        z = self._check_range(z)
        return np.interp(z, self._z_grid, self._H_grid)

    def E(self, z) -> np.ndarray:
        """E(z) = H(z)/H0."""
        return self.H(z) / self.params.H0

    def D_H(self, z) -> np.ndarray:
        """Distancia de Hubble D_H(z) = c / H(z) [Mpc]."""
        return C_LIGHT / self.H(z)

    def D_C(self, z) -> np.ndarray:
        """Distancia comovil radial [Mpc]."""
        z = self._check_range(z)
        return np.interp(z, self._z_grid, self._dc_grid)

    def D_M(self, z) -> np.ndarray:
        """Distancia comovil transversal [Mpc]. Universo plano: D_M = D_C."""
        return self.D_C(z)

    def D_A(self, z) -> np.ndarray:
        """Distancia de diametro angular [Mpc]."""
        z = self._check_range(z)
        return self.D_M(z) / (1.0 + z)

    def D_L(self, z) -> np.ndarray:
        """Distancia de luminosidad [Mpc]."""
        z = self._check_range(z)
        return (1.0 + z) * self.D_M(z)

    def D_V(self, z) -> np.ndarray:
        """Distancia de volumen BAO: D_V = [D_M^2 c z / H]^(1/3) [Mpc]."""
        z = self._check_range(z)
        term = self.D_M(z) ** 2 * C_LIGHT * z / self.H(z)
        return np.maximum(term, 0.0) ** (1.0 / 3.0)

    def mu(self, z) -> np.ndarray:
        """Modulo de distancia mu(z) = 5 log10(D_L/Mpc) + 25 + (M + 19.3)."""
        dL = np.maximum(self.D_L(z), 1e-30)
        return 5.0 * np.log10(dL) + 25.0 + (self.params.M + 19.3)

    # Observables BAO adimensionales
    def DV_rd(self, z) -> np.ndarray:
        return self.D_V(z) / self.params.rd

    def DM_rd(self, z) -> np.ndarray:
        return self.D_M(z) / self.params.rd

    def DH_rd(self, z) -> np.ndarray:
        return self.D_H(z) / self.params.rd

    def w_eff_id(self, z) -> np.ndarray:
        """Ecuacion de estado efectiva del canal indeterminado.

        De la conservacion covariante d(rho)/dz = 3 (1+w) rho / (1+z):

            w(z) = (1+z) rho_id'(z) / (3 rho_id(z)) - 1

        En el limite LCDM (eps=0, z <= z_trans) devuelve exactamente -1.
        """
        z = self._check_range(z)
        rid = self._eff.rho_id
        rho = rho_id_refined(z, rid)
        drho = drho_id_dz(z, rid)
        return (1.0 + z) * drho / (3.0 * rho) - 1.0

    # ------------------------------------------------------------------
    # Exportacion tabular
    # ------------------------------------------------------------------
    def table(self, z) -> dict:
        """Tabla completa de fondo para un array de z."""
        z = self._check_range(z)
        return {
            "z": z,
            "H": self.H(z),
            "E": self.E(z),
            "D_C": self.D_C(z),
            "D_M": self.D_M(z),
            "D_H": self.D_H(z),
            "D_A": self.D_A(z),
            "D_L": self.D_L(z),
            "D_V": self.D_V(z),
            "mu": self.mu(z),
            "DV_rd": self.DV_rd(z),
            "w_eff_id": self.w_eff_id(z),
        }
