#!/usr/bin/env python3
"""Auxiliary Module: Baryogenesis Simulations and Visualizations.

Generates PNG plots for:
1. Sakharov conditions activation
2. CP violation profile
3. Baryon asymmetry generation
4. η_B integration
5. Connection to MCMC ontology
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Auxiliary imports
from mcmc.auxiliary.baryogenesis import (
    BaryogenesisParams,
    sakharov_conditions,
    cp_violation_mcmc,
    bl_violation_rate,
    departure_from_equilibrium,
    eta_B_of_S,
    integrate_eta_B,
)
from mcmc.core.ontology import S_0, S_GEOM

# Output directory
OUTDIR = Path("reports/figures/blocks")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_sakharov_conditions():
    """Plot Sakharov conditions activation near Big Bang."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    params = BaryogenesisParams()
    S_arr = np.linspace(0.5, 3.0, 200)

    # Get all Sakharov conditions for each S
    B_viol = []
    CP_viol = []
    out_eq = []
    combined = []

    for S in S_arr:
        cond = sakharov_conditions(S, params)
        # Use boolean to numerical conversion for plotting
        B_viol.append(1.0 if cond["B_violation"] else 0.0)
        CP_viol.append(1.0 if cond["CP_violation"] else 0.0)
        out_eq.append(1.0 if cond["out_of_equilibrium"] else 0.0)
        combined.append(1.0 if cond["all_satisfied"] else 0.0)

    # B violation
    ax = axes[0, 0]
    # Use bl_violation_rate for continuous plot
    B_rate = [bl_violation_rate(S, params) for S in S_arr]
    ax.plot(S_arr, B_rate, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label=f'S_GEOM = {S_GEOM}')
    ax.fill_between(S_arr, B_rate, alpha=0.3)
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('B-L Violation Rate')
    ax.set_title('Sakharov #1: Baryon Number Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CP violation
    ax = axes[0, 1]
    CP_rate = [cp_violation_mcmc(S, params) for S in S_arr]
    ax.plot(S_arr, CP_rate, 'purple', linewidth=2, label='CP violation')
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\epsilon_{CP}$')
    ax.set_title('Sakharov #2: CP Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Out of equilibrium
    ax = axes[1, 0]
    out_eq_rate = [departure_from_equilibrium(S, params) for S in S_arr]
    ax.plot(S_arr, out_eq_rate, 'orange', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.fill_between(S_arr, out_eq_rate, alpha=0.3, color='orange')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('Non-equilibrium Factor')
    ax.set_title('Sakharov #3: Out of Thermal Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined Sakharov factor
    ax = axes[1, 1]
    combined_rate = np.array(B_rate) * np.array(CP_rate) * np.array(out_eq_rate)
    # Normalize for visibility
    if np.max(combined_rate) > 0:
        combined_rate = combined_rate / np.max(combined_rate)
    ax.plot(S_arr, combined_rate, 'k-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.fill_between(S_arr, combined_rate, alpha=0.3, color='gray')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('Combined Factor (normalized)')
    ax.set_title('Combined Sakharov Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "auxiliary_01_sakharov.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'auxiliary_01_sakharov.png'}")


def plot_cp_violation():
    """Plot CP violation profile."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = BaryogenesisParams()

    # Full S range
    ax = axes[0]
    S_arr = np.linspace(0.1, 10.0, 200)
    epsilon_CP = [cp_violation_mcmc(S, params) for S in S_arr]

    ax.plot(S_arr, epsilon_CP, 'purple', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\epsilon_{CP}$')
    ax.set_title('CP Violation Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom near transition
    ax = axes[1]
    S_trans = np.linspace(0.8, 1.5, 200)
    epsilon_trans = [cp_violation_mcmc(S, params) for S in S_trans]

    ax.plot(S_trans, epsilon_trans, 'purple', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.fill_between(S_trans, epsilon_trans, alpha=0.3, color='purple')
    ax.set_xlabel('S')
    ax.set_ylabel(r'$\epsilon_{CP}$')
    ax.set_title('CP Violation near Big Bang')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Parameter dependence (varying epsilon_CP base value)
    ax = axes[2]
    epsilon_0_values = [1e-10, 5e-10, 1e-9, 5e-9]
    for eps0 in epsilon_0_values:
        params_var = BaryogenesisParams(epsilon_CP=eps0)
        epsilon_var = [cp_violation_mcmc(S, params_var) for S in S_trans]
        ax.plot(S_trans, epsilon_var, linewidth=1.5,
                label=f'$\\epsilon_0 = {eps0:.0e}$')

    ax.axvline(S_GEOM, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('S')
    ax.set_ylabel(r'$\epsilon_{CP}$')
    ax.set_title('CP Violation: Parameter Dependence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "auxiliary_02_cp_violation.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'auxiliary_02_cp_violation.png'}")


def plot_baryon_asymmetry():
    """Plot baryon asymmetry generation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    params = BaryogenesisParams()

    # Asymmetry rate
    ax = axes[0]
    S_arr = np.linspace(0.5, 3.0, 200)
    eta_rate = [eta_B_of_S(S, params) for S in S_arr]

    ax.plot(S_arr, eta_rate, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\eta_B(S)$')
    ax.set_title('Baryon Asymmetry at S')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative asymmetry
    ax = axes[1]
    S_fine = np.linspace(0.5, 5.0, 500)
    eta_fine = [eta_B_of_S(S, params) for S in S_fine]
    # Use trapezoid for integration
    try:
        eta_cumulative = np.array([np.trapezoid(eta_fine[:i+1], S_fine[:i+1])
                                   for i in range(len(S_fine))])
    except AttributeError:
        eta_cumulative = np.array([np.trapz(eta_fine[:i+1], S_fine[:i+1])
                                   for i in range(len(S_fine))])

    ax.plot(S_fine, eta_cumulative, 'g-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.axhline(eta_cumulative[-1], color='gray', linestyle=':',
               label=f'$\\eta_B = {eta_cumulative[-1]:.2e}$')
    ax.set_xlabel('S')
    ax.set_ylabel(r'$\int \eta_B dS$ (cumulative)')
    ax.set_title('Cumulative Baryon Asymmetry')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final value comparison
    ax = axes[2]
    eta_total, _, _ = integrate_eta_B(params)
    eta_observed = 6.1e-10  # Observed value from BBN

    bar_labels = ['MCMC\nPrediction', 'Observed\n(BBN)']
    bar_values = [eta_total, eta_observed]
    bar_colors = ['steelblue', 'green']

    ax.bar(bar_labels, bar_values, color=bar_colors, alpha=0.7)
    ax.set_ylabel(r'$\eta_B$')
    ax.set_title('Baryon Asymmetry Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add ratio
    ratio = eta_total / eta_observed if eta_observed > 0 else 0
    ax.text(0.5, 0.95, f'Ratio: {ratio:.2f}', transform=ax.transAxes,
            ha='center', va='top', fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTDIR / "auxiliary_03_baryon_asymmetry.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'auxiliary_03_baryon_asymmetry.png'}")


def plot_bl_violation():
    """Plot B-L violation rate and transition dynamics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    params = BaryogenesisParams()

    # B-L violation rate vs S
    ax = axes[0]
    S_arr = np.linspace(0.5, 3.0, 200)
    Gamma_BL = [bl_violation_rate(S, params) for S in S_arr]

    ax.plot(S_arr, Gamma_BL, 'orange', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\Gamma_{B-L}$ (B-L violation rate)')
    ax.set_title('B-L Violation Rate vs S')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Baryogenesis window
    ax = axes[1]
    CP_rate = np.array([cp_violation_mcmc(S, params) for S in S_arr])
    BL_rate = np.array([bl_violation_rate(S, params) for S in S_arr])
    out_eq = np.array([departure_from_equilibrium(S, params) for S in S_arr])

    # Effective baryogenesis efficiency
    efficiency = CP_rate * BL_rate * out_eq
    if np.max(efficiency) > 0:
        efficiency = efficiency / np.max(efficiency)

    ax.fill_between(S_arr, efficiency, alpha=0.3, color='purple',
                    label='Baryogenesis window')
    ax.plot(S_arr, efficiency, 'purple', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('Baryogenesis Efficiency (normalized)')
    ax.set_title('MCMC Baryogenesis Window')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "auxiliary_04_bl_violation.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'auxiliary_04_bl_violation.png'}")


def plot_mcmc_connection():
    """Plot connection between baryogenesis and MCMC ontology."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    params = BaryogenesisParams()

    # Baryogenesis in context of full S range
    ax = axes[0, 0]
    S_full = np.linspace(0.1, min(S_0, 100), 500)
    eta_full = [eta_B_of_S(S, params) for S in S_full]

    ax.semilogy(S_full, np.abs(eta_full) + 1e-20, 'b-', linewidth=1)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM (Big Bang)')
    ax.axvline(min(S_0, 100), color='g', linestyle='--', label=f'S_0 = {S_0:.1f} (Today)')

    # Mark key epochs
    ax.axvspan(0.5, 2.0, alpha=0.2, color='orange', label='Baryogenesis epoch')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$|\eta_B(S)|$')
    ax.set_title('Baryogenesis in MCMC Timeline')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Pre-geometric transition detail
    ax = axes[0, 1]
    S_pre = np.linspace(0.001, 2.0, 200)

    # Various pre-geometric thresholds
    S_thresholds = [0.001, 0.01, 0.1, 1.0, 1.001]
    threshold_names = ['Planck', 'GUT', 'Pre-3', 'EW', 'Big Bang']

    for S_th, name in zip(S_thresholds, threshold_names):
        ax.axvline(S_th, linestyle=':', alpha=0.7, label=name)

    eta_pre = [eta_B_of_S(S, params) for S in S_pre]
    ax.plot(S_pre, eta_pre, 'b-', linewidth=2)
    ax.set_xlabel('S')
    ax.set_ylabel(r'$\eta_B(S)$')
    ax.set_title('Pre-Geometric Phase Transitions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parameter sensitivity - varying transition width
    ax = axes[1, 0]
    S_test = np.linspace(0.5, 2.0, 100)

    # Vary delta_S_width (transition width)
    width_values = [0.05, 0.1, 0.2, 0.3]
    for width in width_values:
        params_var = BaryogenesisParams(delta_S_width=width)
        eta_var = [eta_B_of_S(S, params_var) for S in S_test]
        ax.plot(S_test, eta_var, linewidth=1.5, label=f'width = {width}')

    ax.axvline(S_GEOM, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('S')
    ax.set_ylabel(r'$\eta_B(S)$')
    ax.set_title('Sensitivity to Transition Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total η_B vs CP violation parameter
    ax = axes[1, 1]
    epsilon_CP_range = np.logspace(-10, -7, 20)
    eta_B_values = []

    for eps_CP in epsilon_CP_range:
        params_var = BaryogenesisParams(epsilon_CP=eps_CP)
        total, _, _ = integrate_eta_B(params_var)
        eta_B_values.append(total)

    ax.loglog(epsilon_CP_range, eta_B_values, 'o-', linewidth=1.5)
    ax.axhline(6.1e-10, color='g', linestyle='--', label='Observed (BBN)')
    ax.set_xlabel(r'$\epsilon_{CP}$ (CP violation amplitude)')
    ax.set_ylabel(r'$\eta_B$ (total)')
    ax.set_title('Total Baryon Asymmetry vs CP Violation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "auxiliary_05_mcmc_connection.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'auxiliary_05_mcmc_connection.png'}")


def main():
    """Run all auxiliary visualizations."""
    print("=" * 60)
    print("Auxiliary Module: Baryogenesis Simulations")
    print("=" * 60)

    plot_sakharov_conditions()
    plot_cp_violation()
    plot_baryon_asymmetry()
    plot_bl_violation()
    plot_mcmc_connection()

    print("\nAuxiliary visualizations complete!")


if __name__ == "__main__":
    main()
