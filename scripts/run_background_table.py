#!/usr/bin/env python3
"""
Script para generar tablas del fondo cosmológico MCMC.

Uso:
    python scripts/run_background_table.py
    python scripts/run_background_table.py --output tables/background.csv
    python scripts/run_background_table.py --plot

Este script:
1. Genera la rejilla entrópica S
2. Integra las ecuaciones de fondo
3. Calcula canales oscuros (ρ_id, ρ_lat)
4. Exporta tablas y genera plots
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcmc.core.s_grid import SGrid, Seals, create_default_grid
from src.mcmc.core.background import (
    BackgroundParams,
    solve_background,
    solve_background_default,
)
from src.mcmc.core.checks import validate_background_solution
from src.mcmc.core.mapping import SZMapper
from src.mcmc.channels.rho_id_parametric import (
    RhoIdParametricParams,
    RhoIdParametricModel,
)


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Generar tablas del fondo cosmológico MCMC'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/background_table.csv',
        help='Archivo de salida'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generar plots'
    )

    parser.add_argument(
        '--plot-output',
        type=str,
        default='results/background_plots.png',
        help='Archivo de plots'
    )

    parser.add_argument(
        '--H0',
        type=float,
        default=67.4,
        help='Constante de Hubble [km/s/Mpc]'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verbose'
    )

    return parser.parse_args()


def generate_background_table(params: BackgroundParams, verbose: bool = False):
    """
    Genera tabla completa del fondo.

    Returns:
        dict con todas las cantidades de fondo
    """
    # Crear rejilla
    grid, S = create_default_grid()

    if verbose:
        print(f"Rejilla S: {len(S)} puntos, S ∈ [{S[0]:.3f}, {S[-1]:.3f}]")
        print(f"Sellos: S1={grid.seals.S1}, S2={grid.seals.S2}, "
              f"S3={grid.seals.S3}, S4={grid.seals.S4}")

    # Resolver fondo
    if verbose:
        print("Integrando ecuaciones de fondo...")

    sol = solve_background(S, params, grid.seals)

    # Validar
    report = validate_background_solution(
        sol.S, sol.a, sol.H, params.H0, grid.seals, sol.t_rel
    )

    if verbose:
        print(report)

    if not report.all_passed:
        print("ADVERTENCIA: Algunas validaciones fallaron")

    # Construir tabla
    table = {
        'S': sol.S,
        'a': sol.a,
        'z': sol.z,
        't_rel': sol.t_rel,
        'H': sol.H,
        'C': sol.C,
        'T': sol.T,
        'N': sol.N,
        'Phi_ten': sol.Phi_ten,
    }

    return table, sol, grid


def add_dark_sector(table, sol, verbose=False):
    """
    Añade canales oscuros a la tabla.
    """
    if verbose:
        print("Calculando canal indeterminado ρ_id...")

    # Modelo paramétrico de ρ_id
    rho_params = RhoIdParametricParams()
    rho_model = RhoIdParametricModel(rho_params, H0=sol.params.H0)

    # Evaluar en z
    z = table['z']
    rho_id = rho_model.rho_id(z)
    w_eff = rho_model.w_eff(z)

    table['rho_id'] = rho_id
    table['w_eff'] = w_eff

    return table


def save_table(table, output_path, verbose=False):
    """
    Guarda la tabla en CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Construir array
    keys = list(table.keys())
    n_rows = len(table[keys[0]])

    # Header
    header = ','.join(keys)

    # Data
    data = np.column_stack([table[k] for k in keys])

    np.savetxt(
        output_path,
        data,
        delimiter=',',
        header=header,
        comments='',
        fmt='%.10e'
    )

    if verbose:
        print(f"Tabla guardada en: {output_path}")
        print(f"  Columnas: {keys}")
        print(f"  Filas: {n_rows}")


def make_plots(table, sol, output_path, verbose=False):
    """
    Genera plots del fondo.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no está instalado, omitiendo plots")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    S = table['S']
    z = table['z']

    # a(S)
    ax = axes[0, 0]
    ax.plot(S, table['a'])
    ax.set_xlabel('S')
    ax.set_ylabel('a(S)')
    ax.set_title('Factor de escala')
    ax.axvline(sol.params.H0, color='gray', linestyle='--', alpha=0.3)

    # z(S)
    ax = axes[0, 1]
    ax.plot(S, z)
    ax.set_xlabel('S')
    ax.set_ylabel('z(S)')
    ax.set_title('Redshift')
    ax.set_yscale('log')

    # H(z)
    ax = axes[0, 2]
    # Limitar a z < 3 para visualización
    mask = z < 3
    ax.plot(z[mask], table['H'][mask])
    ax.set_xlabel('z')
    ax.set_ylabel('H(z) [km/s/Mpc]')
    ax.set_title('Parámetro de Hubble')

    # C(S), T(S)
    ax = axes[1, 0]
    ax.plot(S, table['C'], label='C(S)')
    ax.plot(S, table['T'], label='T(S)')
    ax.set_xlabel('S')
    ax.set_ylabel('Valor')
    ax.set_title('Funciones auxiliares')
    ax.legend()

    # ρ_id(z)
    ax = axes[1, 1]
    if 'rho_id' in table:
        ax.plot(z[mask], table['rho_id'][mask])
        ax.set_xlabel('z')
        ax.set_ylabel(r'$\rho_{id}/\rho_{crit,0}$')
        ax.set_title('Canal indeterminado')

    # w_eff(z)
    ax = axes[1, 2]
    if 'w_eff' in table:
        ax.plot(z[mask], table['w_eff'][mask])
        ax.axhline(-1, color='gray', linestyle='--', label=r'$w=-1$')
        ax.set_xlabel('z')
        ax.set_ylabel(r'$w_{eff}(z)$')
        ax.set_title('Ecuación de estado')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

    if verbose:
        print(f"Plots guardados en: {output_path}")


def main():
    """Función principal."""
    args = parse_args()

    print("=" * 60)
    print("MCMC - Generador de Tablas de Fondo")
    print("=" * 60)

    # Parámetros
    params = BackgroundParams(H0=args.H0)

    if args.verbose:
        print(f"\nParámetros:")
        print(f"  H0 = {params.H0} km/s/Mpc")

    # Generar tabla
    print("\nGenerando tabla de fondo...")
    table, sol, grid = generate_background_table(params, args.verbose)

    # Añadir sector oscuro
    print("Añadiendo canales oscuros...")
    table = add_dark_sector(table, sol, args.verbose)

    # Guardar
    save_table(table, args.output, args.verbose)

    # Plots
    if args.plot:
        print("Generando plots...")
        make_plots(table, sol, args.plot_output, args.verbose)

    # Resumen
    print("\n" + "-" * 60)
    print("RESUMEN:")
    print(f"  S: [{table['S'][0]:.4f}, {table['S'][-1]:.4f}]")
    print(f"  z: [{table['z'][-1]:.4f}, {table['z'][0]:.1f}]")
    print(f"  a: [{table['a'][0]:.4e}, {table['a'][-1]:.4f}]")
    print(f"  H(z=0) = {table['H'][-1]:.2f} km/s/Mpc")
    print("=" * 60)


if __name__ == '__main__':
    main()
