#!/usr/bin/env python3
"""Generador de visualizaciones para la corrección ontológica MCMC 2025.

Genera PNGs para:
1. Rango S y épocas cosmológicas
2. Mapeo S(z)
3. Campo de Adrián y transiciones
4. Canales (rho_lat, Q_dual)
5. Gravedad modificada (mu, eta)
"""
from __future__ import annotations

import sys
sys.path.insert(0, '/home/user/MCMC-Group/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Crear directorio de salida
OUTPUT_DIR = Path('/home/user/MCMC-Group/reports/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# ============================================================================
# 1. Rango S y épocas cosmológicas
# ============================================================================
def plot_s_range_epochs():
    """Visualiza el rango S y las épocas cosmológicas."""
    from mcmc.core.ontology import THRESHOLDS, get_epoch_name

    fig, ax = plt.subplots(figsize=(14, 6))

    # Definir épocas con colores
    epochs = [
        (0.0, 0.001, 'pre-geom/primordial', '#1a1a2e'),
        (0.001, 0.01, 'pre-geom/trans-1', '#16213e'),
        (0.01, 0.1, 'pre-geom/trans-2', '#0f3460'),
        (0.1, 0.5, 'pre-geom/trans-3', '#1a508b'),
        (0.5, 1.001, 'pre-geom/pre-BB', '#1e5f74'),
        (1.001, 1.08, 'inflación', '#e94560'),
        (1.08, 2.5, 'edad oscura', '#533483'),
        (2.5, 47.5, 'formación estructuras', '#0f4c75'),
        (47.5, 65.0, 'pico estelar', '#3282b8'),
        (65.0, 95.07, 'energía oscura', '#bbe1fa'),
        (95.07, 100.0, 'presente/futuro', '#f0f5f9'),
    ]

    # Dibujar barras para cada época
    for i, (s_min, s_max, name, color) in enumerate(epochs):
        ax.barh(0, s_max - s_min, left=s_min, height=0.5, color=color,
                edgecolor='black', linewidth=0.5, alpha=0.8)

        # Etiquetas para épocas principales
        if s_max - s_min > 5:
            mid = (s_min + s_max) / 2
            ax.text(mid, 0, name, ha='center', va='center', fontsize=9,
                   color='white' if color in ['#1a1a2e', '#16213e', '#0f3460',
                                               '#533483', '#0f4c75'] else 'black')

    # Líneas verticales para transiciones clave
    transitions = [
        (1.001, 'Big Bang\nS=1.001', 'red'),
        (95.07, 'Presente\nS≈95.07', 'green'),
        (47.5, 'Pico estelar\nS=47.5', 'blue'),
    ]

    for s_val, label, color in transitions:
        ax.axvline(s_val, color=color, linestyle='--', linewidth=2, alpha=0.7)
        ax.text(s_val, 0.35, label, ha='center', va='bottom', fontsize=10,
               color=color, fontweight='bold')

    # Configurar ejes
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.5, 0.8)
    ax.set_xlabel('Variable Entropica S')
    ax.set_yticks([])
    ax.set_title('MCMC: Rango Ontologico S in [0, 100]\n'
                'Pre-geometrico: S in [0, 1.001) | Post-Big Bang: S in [1.001, 95.07]',
                fontweight='bold')

    # Leyenda
    ax.text(50, -0.35,
           'Calibracion: S_0 = 100 x (1 - Omega_b) = 95.07 con Omega_b = 0.0493',
           ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_s_range_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 01_s_range_epochs.png")


# ============================================================================
# 2. Mapeo S(z)
# ============================================================================
def plot_s_of_z_mapping():
    """Visualiza el mapeo S(z)."""
    from mcmc.ontology.s_map import EntropyMap

    s_map = EntropyMap()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: S(z) para z pequeños
    ax1 = axes[0, 0]
    z = np.linspace(0, 5, 200)
    S = s_map.S_of_z(z)
    ax1.plot(z, S, 'b-', linewidth=2)
    ax1.axhline(95.07, color='green', linestyle='--', label='S₀ ≈ 95.07 (hoy)')
    ax1.axhline(1.001, color='red', linestyle='--', label='S_GEOM = 1.001 (Big Bang)')
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('S(z)')
    ax1.set_title('Mapeo S(z) - Redshift bajo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: S(z) para z altos (log)
    ax2 = axes[0, 1]
    z_high = np.logspace(-2, 4, 200)
    S_high = s_map.S_of_z(z_high)
    ax2.semilogx(z_high, S_high, 'b-', linewidth=2)
    ax2.axhline(95.07, color='green', linestyle='--', label='S₀ ≈ 95.07')
    ax2.axhline(1.001, color='red', linestyle='--', label='S_GEOM = 1.001')
    ax2.set_xlabel('Redshift z (log)')
    ax2.set_ylabel('S(z)')
    ax2.set_title('Mapeo S(z) - Escala logarítmica')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: a(S) factor de escala
    ax3 = axes[1, 0]
    S_range = np.linspace(2, 95, 200)
    a = s_map.a_of_S(S_range)
    ax3.plot(S_range, a, 'g-', linewidth=2)
    ax3.axhline(1.0, color='black', linestyle=':', label='a = 1 (hoy)')
    ax3.set_xlabel('S')
    ax3.set_ylabel('a(S)')
    ax3.set_title('Factor de escala a(S)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: E(z)² = H²/H₀²
    ax4 = axes[1, 1]
    z = np.linspace(0, 5, 200)
    E2 = s_map.E_squared(z)
    ax4.plot(z, E2, 'purple', linewidth=2)
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('E(z)² = H(z)²/H₀²')
    ax4.set_title('Función de Hubble normalizada E(z)²')
    ax4.grid(True, alpha=0.3)

    # Añadir fórmula
    fig.text(0.5, 0.02,
             r'$S(z) = S_{geom} + \frac{S_0 - S_{geom}}{E(z)^2}$ donde $E(z) = H(z)/H_0 = \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$',
             ha='center', fontsize=12, style='italic')

    plt.suptitle('MCMC: Mapeo Entrópico S ↔ z', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '02_s_z_mapping.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 02_s_z_mapping.png")


# ============================================================================
# 3. Campo de Adrián
# ============================================================================
def plot_adrian_field():
    """Visualiza el Campo de Adrián y sus transiciones."""
    from mcmc.ontology.adrian_field import AdrianField, AdrianFieldParams

    field = AdrianField()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Transiciones (pre-geom y post-BB)
    ax1 = axes[0, 0]
    S_all = np.linspace(0, 100, 500)

    # Mostrar las transiciones como líneas verticales
    colors_pre = ['#1a1a2e', '#16213e', '#0f3460', '#1e5f74']
    colors_post = ['#e94560', '#533483', '#3282b8', '#bbe1fa']

    for i, trans in enumerate(field.transitions):
        if trans.S_n < 1.001:
            ax1.axvline(trans.S_n, color=colors_pre[i % len(colors_pre)],
                       linestyle='--', alpha=0.7, label=f'Pre-geom {trans.S_n:.3f}')
        else:
            ax1.axvline(trans.S_n, color=colors_post[i % len(colors_post)],
                       linestyle='-', linewidth=2, label=f'Post-BB {trans.S_n:.1f}')

    # Potencial V_eff para Φ = 0.1 (calculado punto a punto)
    V_eff = np.array([field.V_eff(0.1, s) for s in S_all])
    ax1.plot(S_all, V_eff / np.max(V_eff), 'k-', linewidth=2, label='V_eff(Φ=0.1)')
    ax1.set_xlabel('S')
    ax1.set_ylabel('V_eff (normalizado)')
    ax1.set_title('Transiciones Ontológicas')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Φ_ten(S)
    ax2 = axes[0, 1]
    Phi_ten = field.Phi_ten(S_all)
    ax2.plot(S_all, Phi_ten, 'b-', linewidth=2)
    ax2.axvline(48.0, color='red', linestyle='--', label='Centro (S=48)')
    ax2.set_xlabel('S')
    ax2.set_ylabel('Phi_ten(S)')
    ax2.set_title('Faz Tensorial Phi_ten')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Potencial V_eff(Φ) para diferentes S
    ax3 = axes[1, 0]
    Phi_range = np.linspace(-1, 1, 200)
    for S_val in [0.1, 1.001, 10.0, 50.0, 95.0]:
        V = np.array([field.V_eff(phi, S_val) for phi in Phi_range])
        ax3.plot(Phi_range, V, label=f'S={S_val}')
    ax3.set_xlabel('Phi_Ad')
    ax3.set_ylabel('V_eff(Phi; S)')
    ax3.set_title('Potencial efectivo V_eff para distintos S')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Theta_lambda suavizado
    ax4 = axes[1, 1]
    x = np.linspace(-5, 5, 200)
    theta = field.Theta_lambda(x)
    ax4.plot(x, theta, 'g-', linewidth=2)
    ax4.axhline(0.5, color='gray', linestyle='--')
    ax4.axvline(0, color='gray', linestyle='--')
    ax4.set_xlabel('x = (S - S_n) / lambda')
    ax4.set_ylabel('Theta_lambda(x)')
    ax4.set_title('Escalon suavizado Theta_lambda(x) = 0.5[1 + tanh(x/lambda)]')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MCMC: Campo de Adrian Phi_Ad', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '03_adrian_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 03_adrian_field.png")


# ============================================================================
# 4. Canales (rho_lat, Q_dual)
# ============================================================================
def plot_channels():
    """Visualiza los canales rho_lat y Q_dual."""
    from mcmc.channels.rho_lat import LatentChannel, LatentChannelParams
    from mcmc.channels.q_dual import QDualParams, eta_lat_of_S, eta_id_of_S, Q_dual_simple

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: κ_lat(S) - coeficiente de decaimiento
    ax1 = axes[0, 0]
    channel = LatentChannel()
    S_arr = np.linspace(0, 100, 500)
    kappa = channel.kappa_lat(S_arr)
    ax1.plot(S_arr, kappa, 'r-', linewidth=2)
    ax1.axvline(1.001, color='gray', linestyle='--', label='S_BB = 1.001')
    ax1.set_xlabel('S')
    ax1.set_ylabel('κ_lat(S)')
    ax1.set_title('Coeficiente de decaimiento entrópico κ_lat(S)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: ρ_lat(S) - densidad latente
    ax2 = axes[0, 1]
    S_post = np.linspace(1.001, 95, 200)
    rho_lat = channel.rho_lat_array(S_post)
    ax2.semilogy(S_post, rho_lat, 'b-', linewidth=2)
    ax2.set_xlabel('S')
    ax2.set_ylabel('ρ_lat(S) / ρ_c')
    ax2.set_title('Densidad latente ρ_lat(S)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: η_lat(S) y η_id(S)
    ax3 = axes[1, 0]
    q_params = QDualParams()
    S_arr = np.linspace(0, 100, 500)
    eta_lat = eta_lat_of_S(S_arr, q_params)
    eta_id = eta_id_of_S(S_arr, q_params)
    ax3.plot(S_arr, eta_lat, 'b-', linewidth=2, label='η_lat(S)')
    ax3.plot(S_arr, eta_id, 'r-', linewidth=2, label='η_id(S)')
    ax3.axvline(q_params.S_star, color='gray', linestyle='--', label=f'S_★ = {q_params.S_star}')
    ax3.set_xlabel('S')
    ax3.set_ylabel('η(S)')
    ax3.set_title('Fracciones de acople η_lat + η_id = 1')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Q_dual simplificado
    ax4 = axes[1, 1]
    Q_result = Q_dual_simple(S_arr, q_params)
    # Q_dual_simple puede retornar tuple (Q_id, Q_lat) o array
    if isinstance(Q_result, tuple):
        Q_id, Q_lat = Q_result
        ax4.plot(S_arr, Q_id, 'b-', linewidth=2, label='Q_id')
        ax4.plot(S_arr, Q_lat, 'r-', linewidth=2, label='Q_lat')
        ax4.legend()
    else:
        ax4.plot(S_arr, Q_result, 'purple', linewidth=2)
    ax4.axhline(0, color='gray', linestyle='-')
    ax4.axvline(q_params.S_star, color='gray', linestyle='--')
    ax4.set_xlabel('S')
    ax4.set_ylabel('Q_dual(S)')
    ax4.set_title('Termino de intercambio Q_dual')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MCMC: Canales ρ_lat y Q_dual', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '04_channels.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 04_channels.png")


# ============================================================================
# 5. Gravedad modificada (μ, η)
# ============================================================================
def plot_modified_gravity():
    """Visualiza los parámetros de gravedad modificada μ y η."""
    from mcmc.growth.mu_eta import (
        MuEtaParams, MuEtaFromS, mu_CPL, eta_CPL,
        Sigma_lensing, compare_modified_gravity
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: μ(a) y η(a) CPL
    ax1 = axes[0, 0]
    a = np.linspace(0.1, 1.0, 100)

    # GR
    params_GR = MuEtaParams(mu_0=1.0, mu_a=0.0, eta_0=1.0, eta_a=0.0)
    mu_GR = mu_CPL(a, params_GR)

    # Pequeña desviación
    params_dev = MuEtaParams(mu_0=1.05, mu_a=0.1, eta_0=0.95, eta_a=-0.05)
    mu_dev = mu_CPL(a, params_dev)
    eta_dev = eta_CPL(a, params_dev)

    ax1.plot(a, mu_GR, 'k--', linewidth=2, label='μ = 1 (GR)')
    ax1.plot(a, mu_dev, 'b-', linewidth=2, label=f'μ(a), μ₀={params_dev.mu_0}')
    ax1.plot(a, eta_dev, 'r-', linewidth=2, label=f'η(a), η₀={params_dev.eta_0}')
    ax1.set_xlabel('Factor de escala a')
    ax1.set_ylabel('μ(a), η(a)')
    ax1.set_title('Parametrización CPL de μ y η')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: μ(S) y η(S) desde el mapa entrópico
    ax2 = axes[0, 1]
    mu_eta_S = MuEtaFromS(alpha_mu=0.01, alpha_eta=0.01)
    S_arr = np.linspace(1.001, 95, 200)
    mu_S = mu_eta_S.mu_of_S(S_arr)
    eta_S = mu_eta_S.eta_of_S(S_arr)

    ax2.plot(S_arr, mu_S, 'b-', linewidth=2, label='μ(S)')
    ax2.plot(S_arr, eta_S, 'r-', linewidth=2, label='η(S)')
    ax2.axhline(1.0, color='gray', linestyle='--', label='GR (μ=η=1)')
    ax2.set_xlabel('S')
    ax2.set_ylabel('μ(S), η(S)')
    ax2.set_title('Gravedad modificada desde mapa entrópico')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Σ para lensing
    ax3 = axes[1, 0]
    Sigma = Sigma_lensing(mu_dev, eta_dev)
    Sigma_GR = np.ones_like(a)
    ax3.plot(a, Sigma_GR, 'k--', linewidth=2, label='Σ = 1 (GR)')
    ax3.plot(a, Sigma, 'g-', linewidth=2, label='Σ = μ(1+η)/2')
    ax3.set_xlabel('Factor de escala a')
    ax3.set_ylabel('Σ(a)')
    ax3.set_title('Parámetro de lensing Σ')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Comparación crecimiento D(a)
    ax4 = axes[1, 1]
    try:
        result = compare_modified_gravity(params_dev, n_points=100)
        ax4.plot(result.a, result.D_GR, 'k--', linewidth=2, label='D(a) GR')
        ax4.plot(result.a, result.D_mod, 'b-', linewidth=2, label='D(a) modificado')
        ax4.set_xlabel('Factor de escala a')
        ax4.set_ylabel('D(a)')
        ax4.set_title('Factor de crecimiento D(a)')
        ax4.legend()
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MCMC: Gravedad Modificada μ(a) y η(a)', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '05_modified_gravity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 05_modified_gravity.png")


# ============================================================================
# 6. Métrica Dual Relativa
# ============================================================================
def plot_dual_metric():
    """Visualiza la Métrica Dual Relativa."""
    from mcmc.ontology.dual_metric import DualRelativeMetric, create_LCDM_metric, create_MCMC_metric

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    S_arr = np.linspace(2, 95, 200)

    # Métricas
    metric_lcdm = create_LCDM_metric()
    metric_mcmc = create_MCMC_metric(phi_ten_amplitude=0.02)

    # Panel 1: Lapse N(S)
    ax1 = axes[0, 0]
    N_lcdm = metric_lcdm.N_of_S(S_arr)
    N_mcmc = metric_mcmc.N_of_S(S_arr)
    ax1.plot(S_arr, N_lcdm, 'k--', linewidth=2, label='N(S) ΛCDM')
    ax1.plot(S_arr, N_mcmc, 'b-', linewidth=2, label='N(S) MCMC')
    ax1.set_xlabel('S')
    ax1.set_ylabel('N(S)')
    ax1.set_title('Lapse entrópico N(S) = exp[Φ_ten(S)]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: g_tt(S)
    ax2 = axes[0, 1]
    g_tt_lcdm = metric_lcdm.g_tt(S_arr)
    g_tt_mcmc = metric_mcmc.g_tt(S_arr)
    ax2.plot(S_arr, g_tt_lcdm, 'k--', linewidth=2, label='g_tt ΛCDM')
    ax2.plot(S_arr, g_tt_mcmc, 'b-', linewidth=2, label='g_tt MCMC')
    ax2.set_xlabel('S')
    ax2.set_ylabel('g_tt(S)')
    ax2.set_title('Componente temporal g_tt = -N²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: g_rr(S)
    ax3 = axes[1, 0]
    g_rr_lcdm = metric_lcdm.g_rr(S_arr)
    g_rr_mcmc = metric_mcmc.g_rr(S_arr)
    ax3.plot(S_arr, g_rr_lcdm, 'k--', linewidth=2, label='g_rr ΛCDM')
    ax3.plot(S_arr, g_rr_mcmc, 'b-', linewidth=2, label='g_rr MCMC')
    ax3.set_xlabel('S')
    ax3.set_ylabel('g_rr(S)')
    ax3.set_title('Componente espacial g_rr = a²(S)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Desviación de FRW
    ax4 = axes[1, 1]
    dev = np.array([metric_mcmc.deviation_from_FRW(s) for s in S_arr])
    ax4.plot(S_arr, dev, 'r-', linewidth=2)
    ax4.axhline(0, color='gray', linestyle='--')
    ax4.set_xlabel('S')
    ax4.set_ylabel('|N - 1|')
    ax4.set_title('Desviación de métrica FRW estándar')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('MCMC: Métrica Dual Relativa g_μν(S)', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / '06_dual_metric.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generado: 06_dual_metric.png")


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generando visualizaciones MCMC - Corrección Ontológica 2025")
    print("=" * 60)
    print()

    plot_s_range_epochs()
    plot_s_of_z_mapping()
    plot_adrian_field()
    plot_channels()
    plot_modified_gravity()
    plot_dual_metric()

    print()
    print("=" * 60)
    print(f"Todos los gráficos guardados en: {OUTPUT_DIR}")
    print("=" * 60)
