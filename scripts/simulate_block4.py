#!/usr/bin/env python3
"""Block 4: Lattice-Gauge Simulations and Visualizations.

Generates PNG plots for:
1. Wilson coupling beta(S) transition
2. Theoretical plaquette expectations
3. Phase transition schematic
4. Confinement/deconfinement diagram
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Block 4 imports
from mcmc.blocks.block4.config import LatticeParams, WilsonParams
from mcmc.blocks.block4.wilson_action import beta_of_S
from mcmc.core.ontology import S_GEOM, S_0

# Output directory
OUTDIR = Path("reports/figures/blocks")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_beta_transition():
    """Plot Wilson coupling beta(S) transition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Full S range
    ax = axes[0]
    S_arr = np.linspace(0.01, 10.0, 200)
    beta_arr = np.array([beta_of_S(s) for s in S_arr])

    ax.plot(S_arr, beta_arr, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label=f'S_GEOM = {S_GEOM}')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\beta(S)$')
    ax.set_title('Wilson Coupling vs Entropic Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom near transition
    ax = axes[1]
    S_trans = np.linspace(0.5, 2.0, 200)
    beta_trans = np.array([beta_of_S(s) for s in S_trans])

    ax.plot(S_trans, beta_trans, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.fill_between(S_trans, beta_trans, alpha=0.3)
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\beta(S)$')
    ax.set_title('Beta Transition near Big Bang')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Beta derivative (susceptibility)
    ax = axes[2]
    d_beta = np.gradient(beta_trans, S_trans)
    ax.plot(S_trans, d_beta, 'g-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'd$\beta$/dS')
    ax.set_title('Beta Susceptibility (Phase Transition Signal)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block4_01_beta_transition.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block4_01_beta_transition.png'}")


def plot_plaquette_theory():
    """Plot theoretical plaquette expectations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Theoretical plaquette vs beta (analytic limits)
    ax = axes[0]
    beta_wc = np.linspace(2.0, 10.0, 100)
    # Weak coupling: <P> ~ 1 - 3/(4*beta) for SU(2)
    plaq_wc = 1 - 3.0 / (4 * beta_wc)

    beta_sc = np.linspace(0.1, 2.0, 50)
    # Strong coupling: <P> ~ beta/8 for SU(2)
    plaq_sc = beta_sc / 8

    ax.plot(beta_wc, plaq_wc, 'b-', linewidth=2, label='Weak coupling')
    ax.plot(beta_sc, plaq_sc, 'r-', linewidth=2, label='Strong coupling')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Plaquette expectation')
    ax.set_title('Plaquette: Analytic Limits (SU(2))')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)

    # Plaquette vs S (theoretical)
    ax = axes[1]
    S_arr = np.linspace(0.3, 5.0, 100)
    beta_arr = np.array([beta_of_S(s) for s in S_arr])

    # Use weak coupling formula
    plaq_S = 1 - 3.0 / (4 * np.maximum(beta_arr, 0.5))
    plaq_S = np.clip(plaq_S, 0, 1)

    ax.plot(S_arr, plaq_S, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.fill_between(S_arr, plaq_S, alpha=0.3)
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('Plaquette (theoretical)')
    ax.set_title('Plaquette vs S')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # String tension schematic
    ax = axes[2]
    # String tension sigma ~ 1/beta^2 in weak coupling
    sigma_wc = 1.0 / beta_wc ** 2
    ax.semilogy(beta_wc, sigma_wc, 'g-', linewidth=2, label=r'$\sigma \sim 1/\beta^2$')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'String tension $\sigma$ (schematic)')
    ax.set_title('Confinement String Tension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block4_02_plaquette_theory.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block4_02_plaquette_theory.png'}")


def plot_phase_diagram():
    """Plot lattice QCD phase diagram schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confinement/Deconfinement as function of S
    ax = axes[0]
    S_arr = np.linspace(0.1, 5.0, 200)

    # Order parameter (Polyakov loop expectation)
    # Confined: |P| ~ 0, Deconfined: |P| > 0
    # Transition near S_GEOM
    S_c = S_GEOM
    width = 0.2
    P_exp = 0.5 * (1 + np.tanh((S_arr - S_c) / width))

    ax.plot(S_arr, P_exp, 'purple', linewidth=2)
    ax.fill_between(S_arr, P_exp, alpha=0.3, color='purple')
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')

    # Label phases
    ax.text(0.3, 0.8, 'DECONFINED\n(QGP)', fontsize=12, ha='center',
            transform=ax.transAxes, color='purple')
    ax.text(0.8, 0.2, 'CONFINED\n(Hadrons)', fontsize=12, ha='center',
            transform=ax.transAxes, color='gray')

    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\langle|P|\rangle$ (Polyakov loop)')
    ax.set_title('Confinement-Deconfinement Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mass gap vs S
    ax = axes[1]
    # Mass gap: large in confined phase, small in deconfined
    mass_gap = 1.0 - 0.8 * P_exp

    ax.plot(S_arr, mass_gap, 'green', linewidth=2)
    ax.fill_between(S_arr, mass_gap, alpha=0.3, color='green')
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('Mass gap (normalized)')
    ax.set_title('Mass Gap vs S')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block4_03_phase_diagram.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block4_03_phase_diagram.png'}")


def plot_wilson_loops():
    """Plot Wilson loop schematic."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Wilson potential schematic
    ax = axes[0]
    R = np.linspace(0.1, 2.0, 100)

    # Confined: V(R) ~ sigma*R (linear)
    sigma = 1.0
    V_confined = sigma * R

    # Deconfined: V(R) ~ -alpha/R + const (Coulomb-like)
    alpha = 0.3
    V_deconfined = -alpha / R + 1.0

    ax.plot(R, V_confined, 'b-', linewidth=2, label='Confined (linear)')
    ax.plot(R, V_deconfined, 'r--', linewidth=2, label='Deconfined (Coulomb)')
    ax.set_xlabel('R (separation)')
    ax.set_ylabel('V(R) (potential)')
    ax.set_title('Quark-Antiquark Potential')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 3)

    # Wilson loop area law
    ax = axes[1]
    A = np.linspace(0.1, 5.0, 100)  # Area

    # Confined: W ~ exp(-sigma*A)
    W_confined = np.exp(-sigma * A)

    # Deconfined: W ~ exp(-P*A) with P ~ perimeter/area
    P = 2 * np.sqrt(A)
    W_deconfined = np.exp(-0.2 * P)

    ax.semilogy(A, W_confined, 'b-', linewidth=2, label='Area law (confined)')
    ax.semilogy(A, W_deconfined, 'r--', linewidth=2, label='Perimeter law (deconfined)')
    ax.set_xlabel('Loop Area')
    ax.set_ylabel('Wilson Loop W(A)')
    ax.set_title('Wilson Loop Behavior')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block4_04_wilson_loops.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block4_04_wilson_loops.png'}")


def plot_s_scan_schematic():
    """Plot S-scan for phase transition detection (schematic)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    S_arr = np.linspace(0.3, 4.0, 100)
    beta_arr = np.array([beta_of_S(s) for s in S_arr])

    # Plaquette vs S
    ax = axes[0, 0]
    plaq = 1 - 3.0 / (4 * np.maximum(beta_arr, 0.5))
    plaq = np.clip(plaq, 0, 1)
    ax.plot(S_arr, plaq, 'o-', linewidth=1.5, markersize=3)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S')
    ax.set_ylabel('Plaquette')
    ax.set_title('Plaquette S-Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Polyakov loop vs S
    ax = axes[0, 1]
    S_c = S_GEOM
    poly = 0.5 * (1 + np.tanh((S_arr - S_c) / 0.2))
    ax.plot(S_arr, poly, 's-', linewidth=1.5, markersize=3, color='purple')
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S')
    ax.set_ylabel('|Polyakov|')
    ax.set_title('Polyakov Loop S-Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Susceptibility
    ax = axes[1, 0]
    d_plaq = np.gradient(plaq, S_arr)
    ax.plot(S_arr, d_plaq, 'o-', linewidth=1.5, markersize=3, color='orange')
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S')
    ax.set_ylabel('d(Plaquette)/dS')
    ax.set_title('Plaquette Susceptibility')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined
    ax = axes[1, 1]
    ax.plot(S_arr, plaq, 'o-', label='Plaquette', linewidth=1.5, markersize=3)
    ax.plot(S_arr, poly, 's-', label='|Polyakov|', linewidth=1.5, markersize=3)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.set_xlabel('S')
    ax.set_ylabel('Observable')
    ax.set_title('Combined S-Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block4_05_s_scan.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block4_05_s_scan.png'}")


def main():
    """Run all Block 4 visualizations."""
    print("=" * 60)
    print("Block 4: Lattice-Gauge Simulations")
    print("=" * 60)

    plot_beta_transition()
    plot_plaquette_theory()
    plot_phase_diagram()
    plot_wilson_loops()
    plot_s_scan_schematic()

    print("\nBlock 4 visualizations complete!")


if __name__ == "__main__":
    main()
