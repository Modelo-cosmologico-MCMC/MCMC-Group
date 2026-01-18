#!/usr/bin/env python3
"""
Demo: Distancias Cosmologicas MCMC
==================================

Script de demostracion que calcula distancias cosmologicas usando el modelo
MCMC (Modelo Cosmologico de Multiples Colapsos).

Compara:
- Modelo MCMC refinado (con rho_id parametrico)
- Modelo LCDM estandar

Uso:
    python scripts/run_distance_demo.py

Salida:
    - Tabla de distancias a varios redshifts
    - Comparacion MCMC vs LCDM
"""

import numpy as np
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from mcmc.observables.distances import (
    comoving_distance,
    luminosity_distance,
    angular_diameter_distance,
    distance_modulus,
    DistanceCalculator
)
from mcmc.channels.rho_id_parametric import (
    RhoIdParametricParams,
    H_of_z_with_rho_id,
    E_squared_with_rho_id
)


# Constantes
C_KM_S = 299792.458  # km/s


def E_LCDM(z: np.ndarray, Omega_m: float = 0.315) -> np.ndarray:
    """E(z) para LCDM plano."""
    Omega_Lambda = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)


def H_LCDM(z: np.ndarray, H0: float = 67.4, Omega_m: float = 0.315) -> np.ndarray:
    """H(z) para LCDM [km/s/Mpc]."""
    return H0 * E_LCDM(z, Omega_m)


def H_MCMC(z: np.ndarray, H0: float, params: RhoIdParametricParams,
           Omega_m0: float) -> np.ndarray:
    """H(z) para modelo MCMC con rho_id."""
    return H_of_z_with_rho_id(z, H0, params, Omega_m0)


def main():
    print("=" * 70)
    print("DEMO: Distancias Cosmologicas - MCMC vs LCDM")
    print("=" * 70)
    print()

    # Parametros
    H0 = 67.4  # km/s/Mpc
    Omega_m0 = 0.315

    # Parametros MCMC
    mcmc_params = RhoIdParametricParams(
        Omega_id0=0.685,    # 1 - Omega_m0
        z_trans=0.5,
        epsilon=0.02,
        delta_z_trans=0.2,
        gamma=0.0
    )

    # Redshifts de prueba
    z_test = np.array([0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])

    print(f"Parametros cosmologicos:")
    print(f"  H0 = {H0} km/s/Mpc")
    print(f"  Omega_m0 = {Omega_m0}")
    print()
    print(f"Parametros MCMC:")
    print(f"  Omega_id0 = {mcmc_params.Omega_id0}")
    print(f"  z_trans = {mcmc_params.z_trans}")
    print(f"  epsilon = {mcmc_params.epsilon}")
    print()

    # Calcular distancias para LCDM
    print("-" * 70)
    print("LCDM - Distancias:")
    print("-" * 70)

    # Crear grid de integracion
    z_grid = np.linspace(0, 3.0, 1000)
    H_grid_lcdm = H_LCDM(z_grid, H0, Omega_m0)

    print(f"{'z':>6} | {'d_C [Mpc]':>12} | {'d_L [Mpc]':>12} | {'d_A [Mpc]':>12} | {'mu [mag]':>10}")
    print("-" * 70)

    for z in z_test:
        d_C = comoving_distance(z, H_LCDM, z_grid, H_grid_lcdm)
        d_L = luminosity_distance(z, d_C)
        d_A = angular_diameter_distance(z, d_C)
        mu = distance_modulus(d_L)
        print(f"{z:6.2f} | {d_C[0]:12.2f} | {d_L[0]:12.2f} | {d_A[0]:12.2f} | {mu[0]:10.4f}")

    print()

    # Calcular distancias para MCMC
    print("-" * 70)
    print("MCMC - Distancias:")
    print("-" * 70)

    def H_mcmc_func(z):
        return H_MCMC(np.atleast_1d(z), H0, mcmc_params, Omega_m0)

    H_grid_mcmc = H_mcmc_func(z_grid)

    print(f"{'z':>6} | {'d_C [Mpc]':>12} | {'d_L [Mpc]':>12} | {'d_A [Mpc]':>12} | {'mu [mag]':>10}")
    print("-" * 70)

    for z in z_test:
        d_C = comoving_distance(z, H_mcmc_func, z_grid, H_grid_mcmc)
        d_L = luminosity_distance(z, d_C)
        d_A = angular_diameter_distance(z, d_C)
        mu = distance_modulus(d_L)
        print(f"{z:6.2f} | {d_C[0]:12.2f} | {d_L[0]:12.2f} | {d_A[0]:12.2f} | {mu[0]:10.4f}")

    print()

    # Comparacion
    print("-" * 70)
    print("Diferencias relativas MCMC vs LCDM (%):")
    print("-" * 70)

    print(f"{'z':>6} | {'Delta d_C':>12} | {'Delta d_L':>12} | {'Delta mu':>12}")
    print("-" * 70)

    for z in z_test:
        d_C_lcdm = comoving_distance(z, H_LCDM, z_grid, H_grid_lcdm)[0]
        d_C_mcmc = comoving_distance(z, H_mcmc_func, z_grid, H_grid_mcmc)[0]

        d_L_lcdm = luminosity_distance(z, np.array([d_C_lcdm]))[0]
        d_L_mcmc = luminosity_distance(z, np.array([d_C_mcmc]))[0]

        mu_lcdm = distance_modulus(np.array([d_L_lcdm]))[0]
        mu_mcmc = distance_modulus(np.array([d_L_mcmc]))[0]

        delta_dC = 100 * (d_C_mcmc - d_C_lcdm) / d_C_lcdm
        delta_dL = 100 * (d_L_mcmc - d_L_lcdm) / d_L_lcdm
        delta_mu = mu_mcmc - mu_lcdm  # diferencia absoluta en magnitudes

        print(f"{z:6.2f} | {delta_dC:+11.4f}% | {delta_dL:+11.4f}% | {delta_mu:+11.5f}")

    print()
    print("=" * 70)
    print("Demo completado exitosamente.")
    print("=" * 70)


if __name__ == "__main__":
    main()
