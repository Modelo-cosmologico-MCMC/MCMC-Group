#!/usr/bin/env python3
"""
Script principal para ejecutar el ajuste MCMC con emcee.

Uso:
    python scripts/run_fit_emcee.py
    python scripts/run_fit_emcee.py --config custom_config.yaml
    python scripts/run_fit_emcee.py --model lcdm --n-walkers 64 --n-steps 10000

Este script:
1. Carga configuración y datos
2. Construye el modelo y likelihood
3. Ejecuta el sampler emcee
4. Genera reportes y plots
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcmc.observables.distances import DistanceCalculator, H_lcdm
from src.mcmc.observables.bao import get_combined_bao_data
from src.mcmc.observables.hz import get_cosmic_chronometers_data
from src.mcmc.observables.sne import get_pantheon_binned_data
from src.mcmc.observables.likelihoods import CombinedLikelihood, LikelihoodConfig
from src.mcmc.observables.info_criteria import compute_all_criteria, compare_models
from src.mcmc.channels.rho_id_parametric import (
    RhoIdParametricParams,
    H_of_z_with_rho_id,
)
from src.mcmc.inference.emcee_fit import (
    Parameter,
    MCMCConfig,
    MCMCFitter,
    run_mcmc_fit,
    get_mcmc_refined_parameters,
    HAS_EMCEE,
)


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Ejecutar ajuste MCMC del modelo cosmológico'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Archivo de configuración YAML'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='mcmc_refined',
        choices=['mcmc_refined', 'lcdm', 'wcdm'],
        help='Modelo a ajustar'
    )

    parser.add_argument(
        '--n-walkers',
        type=int,
        default=32,
        help='Número de walkers'
    )

    parser.add_argument(
        '--n-steps',
        type=int,
        default=5000,
        help='Número de pasos por walker'
    )

    parser.add_argument(
        '--n-burnin',
        type=int,
        default=1000,
        help='Pasos de burn-in a descartar'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directorio de salida'
    )

    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Desactivar barra de progreso'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Semilla para reproducibilidad'
    )

    return parser.parse_args()


def build_lcdm_model():
    """Construye modelo ΛCDM."""
    parameters = [
        Parameter(
            name='H0',
            latex=r'$H_0$',
            initial=67.4,
            prior_min=60.0,
            prior_max=80.0
        ),
        Parameter(
            name='Omega_m',
            latex=r'$\Omega_m$',
            initial=0.3,
            prior_min=0.1,
            prior_max=0.5
        ),
    ]

    def model_builder(params):
        H0, Omega_m = params

        def H_func(z):
            return H_lcdm(z, H0, Omega_m)

        dist_calc = DistanceCalculator(H_func=H_func, H0=H0)
        return dist_calc, H_func

    return parameters, model_builder


def build_mcmc_refined_model():
    """Construye modelo MCMC refinado (Nivel A)."""
    parameters = get_mcmc_refined_parameters()
    Omega_m0 = 0.3  # Fijo para simplificar

    def model_builder(params):
        H0, Omega_id0, z_trans, epsilon, gamma = params

        rho_params = RhoIdParametricParams(
            Omega_id0=Omega_id0,
            z_trans=z_trans,
            epsilon=epsilon,
            gamma=gamma
        )

        def H_func(z):
            return H_of_z_with_rho_id(
                np.atleast_1d(z), H0, rho_params, Omega_m0
            )[0] if np.isscalar(z) else H_of_z_with_rho_id(
                z, H0, rho_params, Omega_m0
            )

        dist_calc = DistanceCalculator(H_func=H_func, H0=H0)
        return dist_calc, H_func

    return parameters, model_builder


def main():
    """Función principal."""
    args = parse_args()

    # Verificar emcee
    if not HAS_EMCEE:
        print("ERROR: emcee no está instalado.")
        print("Instalar con: pip install emcee")
        sys.exit(1)

    # Semilla
    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 60)
    print("MCMC - Ajuste Bayesiano")
    print("=" * 60)

    # Cargar datos
    print("\nCargando datos...")
    bao_data = get_combined_bao_data()
    Hz_data = get_cosmic_chronometers_data()
    sne_data = get_pantheon_binned_data()

    print(f"  BAO: {bao_data.n_points} puntos")
    print(f"  H(z): {Hz_data.n_points} puntos")
    print(f"  SNe: {sne_data.n_sne} puntos")

    # Construir likelihood
    config = LikelihoodConfig()
    likelihood = CombinedLikelihood(
        config=config,
        bao_data=bao_data,
        Hz_data=Hz_data,
        sne_data=sne_data
    )

    # Construir modelo
    print(f"\nModelo: {args.model}")
    if args.model == 'lcdm':
        parameters, model_builder = build_lcdm_model()
    elif args.model == 'mcmc_refined':
        parameters, model_builder = build_mcmc_refined_model()
    else:
        print(f"Modelo {args.model} no implementado aún")
        sys.exit(1)

    print(f"  Parámetros: {[p.name for p in parameters]}")

    # Función de log-likelihood
    def log_likelihood(params):
        try:
            dist_calc, H_func = model_builder(params)
            return likelihood.log_likelihood(dist_calc, H_func)
        except Exception:
            return -np.inf

    # Configurar MCMC
    mcmc_config = MCMCConfig(
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burnin=args.n_burnin,
        progress=not args.no_progress
    )

    print(f"\nConfiguración MCMC:")
    print(f"  Walkers: {mcmc_config.n_walkers}")
    print(f"  Pasos: {mcmc_config.n_steps}")
    print(f"  Burn-in: {mcmc_config.n_burnin}")

    # Ejecutar
    print("\nEjecutando MCMC...")
    fitter = MCMCFitter(parameters, log_likelihood, mcmc_config)
    result = fitter.run()

    # Mostrar resultados
    print("\n" + result.summary())

    # Calcular criterios de información
    n_params = len(parameters)
    n_data = likelihood.n_bao + likelihood.n_Hz + likelihood.n_sne
    chi2_best = -2 * np.max(result.log_prob)

    criteria = compute_all_criteria(chi2_best, n_params, n_data)
    print(f"\n{criteria}")

    # Guardar resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f'{args.model}_samples.npy', result.samples)
    np.save(output_dir / f'{args.model}_log_prob.npy', result.log_prob)

    # Resumen
    with open(output_dir / f'{args.model}_summary.txt', 'w') as f:
        f.write(result.summary())
        f.write(f"\n\n{criteria}")

    print(f"\nResultados guardados en: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
