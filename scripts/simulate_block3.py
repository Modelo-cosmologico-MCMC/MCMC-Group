#!/usr/bin/env python3
"""Block 3: N-body Cronos Simulations and Visualizations.

Generates PNG plots for:
1. Cronos timestep and lapse function N(S)
2. Halo density profiles (NFW, Burkert, Zhao-MCMC)
3. Rotation curves with MCV contribution
4. Mass function comparison
5. Subhalo count with MCV suppression
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Block 3 imports
from mcmc.blocks.block3.timestep_cronos import (
    CronosTimestep,
    lapse_function_default,
)
from mcmc.blocks.block3.profiles.nfw import NFWProfile, NFWParams, create_nfw_from_concentration
from mcmc.blocks.block3.profiles.burkert import BurkertProfile, BurkertParams
from mcmc.blocks.block3.profiles.zhao_mcmc import ZhaoMCMCProfile, ZhaoMCMCParams
from mcmc.core.ontology import S_0, S_GEOM

# Output directory
OUTDIR = Path("reports/figures/blocks")
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_cronos_timestep():
    """Plot Cronos timestep and lapse function."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # S range
    S_pre = np.linspace(0.01, 1.0, 100)
    S_post = np.linspace(1.001, S_0, 200)
    S_full = np.concatenate([S_pre, S_post])

    # Lapse function N(S)
    ax = axes[0]
    N_full = lapse_function_default(S_full)
    ax.plot(S_full, N_full, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label=f'S_GEOM = {S_GEOM}')
    ax.axvline(S_0, color='g', linestyle='--', label=f'S_0 = {S_0:.1f}')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('N(S) (lapse function)')
    ax.set_title('Entropic Lapse Function N(S)')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # Cronos timestep ratio
    ax = axes[1]
    cronos = CronosTimestep()
    dt_N = 0.01
    dt_C = np.array([cronos.dt_cronos(dt_N, s) for s in S_full])
    ratio = dt_C / dt_N
    ax.plot(S_full, ratio, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label='S_GEOM')
    ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('dt_C / dt_N')
    ax.set_title('Cronos Timestep Ratio')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)

    # Zoom near transition
    ax = axes[2]
    S_trans = np.linspace(0.5, 2.0, 200)
    N_trans = lapse_function_default(S_trans)
    ax.plot(S_trans, N_trans, 'b-', linewidth=2)
    ax.axvline(S_GEOM, color='r', linestyle='--', label=f'S_GEOM = {S_GEOM}')
    ax.fill_between(S_trans, N_trans, alpha=0.3)
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel('N(S)')
    ax.set_title('Lapse Function near Big Bang Transition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block3_01_cronos_timestep.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block3_01_cronos_timestep.png'}")


def plot_halo_profiles():
    """Plot halo density profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    r = np.logspace(-2, 0, 200)  # 0.01 to 1 Mpc

    # NFW profile
    ax = axes[0, 0]
    nfw = NFWProfile(NFWParams(rho_s=1e7, r_s=0.02))
    rho_nfw = nfw.density(r)
    ax.loglog(r * 1000, rho_nfw, 'b-', linewidth=2, label='NFW standard')

    # NFW with MCMC correction (stratified present)
    for S_halo, color in [(10.0, 'c'), (50.0, 'g'), (90.0, 'orange')]:
        nfw_s = NFWProfile(NFWParams(rho_s=1e7, r_s=0.02, S_halo=S_halo))
        S_local = S_halo * 0.8  # Denser region has lower local S
        rho_mcmc = nfw_s.density_mcmc(r, S_local)
        ax.loglog(r * 1000, rho_mcmc, color=color, linestyle='--',
                  linewidth=1.5, label=f'NFW S_halo={S_halo}')

    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\rho$ [M$_\odot$/Mpc$^3$]')
    ax.set_title('NFW Profile with MCMC Stratified Present')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Burkert profile
    ax = axes[0, 1]
    burkert = BurkertProfile(BurkertParams(rho_0=1e8, r_0=0.005))
    rho_burk = burkert.density(r)
    ax.loglog(r * 1000, rho_burk, 'r-', linewidth=2, label='Burkert (cored)')

    for S_halo, color in [(10.0, 'pink'), (50.0, 'salmon'), (90.0, 'darkred')]:
        burk_s = BurkertProfile(BurkertParams(rho_0=1e8, r_0=0.005, S_halo=S_halo))
        S_local = S_halo * 0.8
        rho_mcmc = burk_s.density_mcmc(r, S_local)
        ax.loglog(r * 1000, rho_mcmc, color=color, linestyle='--',
                  linewidth=1.5, label=f'Burkert S={S_halo}')

    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\rho$ [M$_\odot$/Mpc$^3$]')
    ax.set_title('Burkert Profile with MCMC Corrections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zhao-MCMC profile - S dependence
    ax = axes[1, 0]
    for S, color in [(5.0, 'purple'), (20.0, 'blue'), (50.0, 'green'),
                     (80.0, 'orange'), (95.0, 'red')]:
        zhao = ZhaoMCMCProfile(ZhaoMCMCParams(rho_s=1e7, r_s=0.02, S_halo=S))
        rho_zhao = zhao.density(r)
        ax.loglog(r * 1000, rho_zhao, color=color, linewidth=1.5, label=f'S={S}')

    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\rho$ [M$_\odot$/Mpc$^3$]')
    ax.set_title('Zhao-MCMC Profile: S-dependent Inner Slope')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Inner slope vs S
    ax = axes[1, 1]
    S_arr = np.linspace(1.0, 95.0, 100)
    gamma_inner = []
    for S in S_arr:
        zhao = ZhaoMCMCProfile(ZhaoMCMCParams(rho_s=1e7, r_s=0.02, S_halo=S))
        gamma_inner.append(zhao.gamma())

    ax.plot(S_arr, gamma_inner, 'b-', linewidth=2)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='NFW (gamma=1)')
    ax.axhline(0.0, color='r', linestyle='--', alpha=0.5, label='Core (gamma=0)')
    ax.axvline(S_GEOM, color='gray', linestyle=':', label='S_GEOM')
    ax.set_xlabel('S (entropic parameter)')
    ax.set_ylabel(r'$\gamma_{inner}$')
    ax.set_title('Zhao-MCMC Inner Slope vs S')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block3_02_halo_profiles.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block3_02_halo_profiles.png'}")


def plot_rotation_curves():
    """Plot rotation curves with MCV contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    r = np.linspace(0.001, 0.1, 100)  # Mpc

    # Standard rotation curve from NFW
    ax = axes[0]
    nfw = create_nfw_from_concentration(M_vir=1e12, c=10.0)
    v_halo = nfw.velocity(r)

    ax.plot(r * 1000, v_halo, 'k-', linewidth=2, label='NFW Halo')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.set_title('Standard NFW Rotation Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zhao-MCMC rotation curves at different S
    ax = axes[1]
    for S, color in [(10.0, 'blue'), (50.0, 'green'), (90.0, 'red')]:
        zhao = ZhaoMCMCProfile(ZhaoMCMCParams(rho_s=1e7, r_s=0.02, S_halo=S))
        v_mcmc = zhao.velocity(r)
        ax.plot(r * 1000, v_mcmc, color=color, linewidth=1.5, label=f'S={S}')

    ax.plot(r * 1000, v_halo, 'k--', linewidth=1, alpha=0.5, label='NFW')
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel('v [km/s]')
    ax.set_title('Zhao-MCMC Rotation Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity difference relative to NFW
    ax = axes[2]
    for S, color in [(10.0, 'blue'), (50.0, 'green'), (90.0, 'red')]:
        zhao = ZhaoMCMCProfile(ZhaoMCMCParams(rho_s=1e7, r_s=0.02, S_halo=S))
        v_mcmc = zhao.velocity(r)
        delta_v = (v_mcmc - v_halo) / (v_halo + 1e-10) * 100
        ax.plot(r * 1000, delta_v, color=color, linewidth=1.5, label=f'S={S}')

    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('r [kpc]')
    ax.set_ylabel(r'$\Delta v / v$ [%]')
    ax.set_title('MCMC Correction to Rotation Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block3_03_rotation_curves.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block3_03_rotation_curves.png'}")


def plot_mass_function():
    """Plot halo mass function with MCMC modifications."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    M = np.logspace(10, 15, 50)  # Solar masses

    # Standard Press-Schechter mass function (simplified)
    M_star = 1e13
    dn_dM_ps = M ** (-2) * np.exp(-M / M_star)
    dn_dM_ps = dn_dM_ps / np.max(dn_dM_ps)

    ax = axes[0]
    ax.loglog(M, M ** 2 * dn_dM_ps, 'k-', linewidth=2, label='Press-Schechter')

    # MCMC modification
    for S, color in [(50.0, 'blue'), (70.0, 'green'), (90.0, 'red')]:
        suppression = 1 - 0.3 * (S / 100) * np.exp(-M / 1e11)
        dn_dM_mcmc = dn_dM_ps * suppression
        ax.loglog(M, M ** 2 * dn_dM_mcmc, color=color, linestyle='--',
                  linewidth=1.5, label=f'MCMC S={S}')

    ax.set_xlabel(r'M [M$_\odot$]')
    ax.set_ylabel(r'M$^2$ dn/dM (arbitrary units)')
    ax.set_title('Halo Mass Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio to PS
    ax = axes[1]
    for S, color in [(50.0, 'blue'), (70.0, 'green'), (90.0, 'red')]:
        suppression = 1 - 0.3 * (S / 100) * np.exp(-M / 1e11)
        ax.semilogx(M, suppression, color=color, linewidth=1.5, label=f'S={S}')

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'M [M$_\odot$]')
    ax.set_ylabel('MCMC / Press-Schechter')
    ax.set_title('MCMC Correction to Mass Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.1)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block3_04_mass_function.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block3_04_mass_function.png'}")


def plot_subhalo_count():
    """Plot subhalo count with MCV suppression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    M_host = 1e12
    M_sub = np.logspace(6, 11, 50)

    # CDM subhalo mass function
    N_cdm = (M_sub / M_host) ** (-0.9)
    N_cdm = N_cdm / N_cdm[0] * 1000

    ax = axes[0]
    ax.loglog(M_sub, N_cdm, 'k-', linewidth=2, label='CDM')

    for S, color in [(30.0, 'blue'), (60.0, 'green'), (90.0, 'red')]:
        M_supp = 1e8 * (S / 50) ** 2
        suppression = 1 / (1 + (M_supp / M_sub) ** 2)
        N_mcmc = N_cdm * suppression
        ax.loglog(M_sub, N_mcmc, color=color, linestyle='--',
                  linewidth=1.5, label=f'MCMC S={S}')

    ax.set_xlabel(r'M$_{sub}$ [M$_\odot$]')
    ax.set_ylabel('N(>M)')
    ax.set_title(f'Subhalo Abundance (M_host = {M_host:.0e} M_sun)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Suppression factor
    ax = axes[1]
    for S, color in [(30.0, 'blue'), (60.0, 'green'), (90.0, 'red')]:
        M_supp = 1e8 * (S / 50) ** 2
        suppression = 1 / (1 + (M_supp / M_sub) ** 2)
        ax.semilogx(M_sub, suppression, color=color, linewidth=1.5, label=f'S={S}')

    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel(r'M$_{sub}$ [M$_\odot$]')
    ax.set_ylabel('N_MCMC / N_CDM')
    ax.set_title('MCV Suppression of Subhalos')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig(OUTDIR / "block3_05_subhalo_count.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTDIR / 'block3_05_subhalo_count.png'}")


def main():
    """Run all Block 3 visualizations."""
    print("=" * 60)
    print("Block 3: N-body Cronos Simulations")
    print("=" * 60)

    plot_cronos_timestep()
    plot_halo_profiles()
    plot_rotation_curves()
    plot_mass_function()
    plot_subhalo_count()

    print("\nBlock 3 visualizations complete!")


if __name__ == "__main__":
    main()
