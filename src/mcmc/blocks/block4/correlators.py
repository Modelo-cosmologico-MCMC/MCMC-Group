"""Correlator calculations for lattice gauge theory.

Computes gluon, Polyakov loop, and meson correlators
for extracting physical observables like mass gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.blocks.block4.wilson_action import (
    LatticeConfiguration,
    plaquette,
    wilson_loop,
)


@dataclass
class CorrelatorData:
    """Container for correlator data.

    Attributes:
        t: Time separations
        C: Correlator values
        C_err: Correlator errors (from bootstrap)
        connected: Whether correlator is connected
    """
    t: np.ndarray
    C: np.ndarray
    C_err: np.ndarray | None = None
    connected: bool = True


def gluon_correlator(
    configs: list[LatticeConfiguration],
    t_max: int | None = None,
) -> CorrelatorData:
    """Compute gluon (plaquette) correlator.

    C(t) = <P(0) P(t)> - <P>^2

    where P is the average spatial plaquette at time slice t.

    Args:
        configs: List of gauge configurations
        t_max: Maximum time separation (default: Nt/2)

    Returns:
        CorrelatorData with time-separated correlator
    """
    if not configs:
        raise ValueError("No configurations provided")

    lattice = configs[0].lattice
    Nt = lattice.Nt
    if t_max is None:
        t_max = Nt // 2

    N = configs[0].N
    n_configs = len(configs)

    # Compute spatial plaquette at each time slice
    def spatial_plaquette_slice(config: LatticeConfiguration, t: int) -> float:
        """Average spatial plaquette at time t."""
        total = 0.0
        count = 0
        for x in range(lattice.Nx):
            for y in range(lattice.Ny):
                for z in range(lattice.Nz):
                    site = (x, y, z, t)
                    # Spatial plaquettes: xy, xz, yz
                    for mu in range(3):
                        for nu in range(mu + 1, 3):
                            P = plaquette(config, site, mu, nu)
                            total += np.real(P) / N
                            count += 1
        return total / count

    # Collect plaquette data
    P_all = np.zeros((n_configs, Nt))
    for ic, config in enumerate(configs):
        for t in range(Nt):
            P_all[ic, t] = spatial_plaquette_slice(config, t)

    # Compute correlator C(dt)
    t_arr = np.arange(t_max + 1)
    C = np.zeros(t_max + 1)
    C_var = np.zeros(t_max + 1)

    P_mean = np.mean(P_all)

    for dt in range(t_max + 1):
        C_samples = []
        for ic in range(n_configs):
            for t0 in range(Nt):
                t1 = (t0 + dt) % Nt
                c = P_all[ic, t0] * P_all[ic, t1] - P_mean ** 2
                C_samples.append(c)
        C[dt] = np.mean(C_samples)
        C_var[dt] = np.var(C_samples) / len(C_samples)

    C_err = np.sqrt(C_var)

    return CorrelatorData(t=t_arr, C=C, C_err=C_err, connected=True)


def polyakov_loop(config: LatticeConfiguration) -> np.ndarray:
    """Compute Polyakov loop at each spatial site.

    L(x) = Tr[prod_t U_t(x, t)]

    The Polyakov loop is the order parameter for confinement.

    Args:
        config: Gauge configuration

    Returns:
        Array of Polyakov loops [Nx, Ny, Nz]
    """
    lattice = config.lattice
    N = config.N
    L = np.zeros((lattice.Nx, lattice.Ny, lattice.Nz), dtype=complex)

    for x in range(lattice.Nx):
        for y in range(lattice.Ny):
            for z in range(lattice.Nz):
                # Product of temporal links
                W = np.eye(N, dtype=complex)
                for t in range(lattice.Nt):
                    site = (x, y, z, t)
                    W = W @ config.get_link(site, 3)  # mu=3 is temporal
                L[x, y, z] = np.trace(W) / N

    return L


def polyakov_correlator(
    configs: list[LatticeConfiguration],
    r_max: int | None = None,
) -> CorrelatorData:
    """Compute Polyakov loop correlator.

    C(r) = <L(0) L^*(r)> - |<L>|^2

    Extracts the static quark potential V(r).

    Args:
        configs: List of gauge configurations
        r_max: Maximum spatial separation

    Returns:
        CorrelatorData with spatial correlator
    """
    if not configs:
        raise ValueError("No configurations provided")

    lattice = configs[0].lattice
    if r_max is None:
        r_max = lattice.Nx // 2

    len(configs)

    # Compute Polyakov loops for all configs
    L_all = []
    for config in configs:
        L_all.append(polyakov_loop(config))

    # Compute correlator
    r_arr = np.arange(r_max + 1)
    C = np.zeros(r_max + 1)
    C_var = np.zeros(r_max + 1)

    L_mean = np.mean([np.mean(L) for L in L_all])

    for r in range(r_max + 1):
        C_samples = []
        for L in L_all:
            for x in range(lattice.Nx):
                for y in range(lattice.Ny):
                    for z in range(lattice.Nz):
                        # Correlator in x-direction
                        x2 = (x + r) % lattice.Nx
                        c = np.real(L[x, y, z] * np.conj(L[x2, y, z])) - np.abs(L_mean) ** 2
                        C_samples.append(c)
        C[r] = np.mean(C_samples)
        C_var[r] = np.var(C_samples) / len(C_samples)

    C_err = np.sqrt(C_var)

    return CorrelatorData(t=r_arr, C=C, C_err=C_err, connected=True)


def meson_correlator(
    configs: list[LatticeConfiguration],
    t_max: int | None = None,
    operator: str = "local",
) -> CorrelatorData:
    """Compute meson-like correlator (Wilson loop correlator).

    For pure gauge theory, we use Wilson loops as meson interpolators:
    C(t) = <W(R, t)>

    where R is a fixed spatial extent.

    Args:
        configs: List of gauge configurations
        t_max: Maximum time separation
        operator: Meson operator type ("local" or "smeared")

    Returns:
        CorrelatorData with meson correlator
    """
    if not configs:
        raise ValueError("No configurations provided")

    lattice = configs[0].lattice
    Nt = lattice.Nt
    if t_max is None:
        t_max = Nt // 2

    len(configs)
    N = configs[0].N
    R = 3  # Fixed spatial extent for "meson"

    # Compute Wilson loops W(R, t) for each config
    t_arr = np.arange(1, t_max + 1)
    C = np.zeros(t_max)
    C_var = np.zeros(t_max)

    for it, T in enumerate(t_arr):
        W_samples = []
        for config in configs:
            # Average over spatial positions
            W_total = 0.0
            count = 0
            for x in range(lattice.Nx):
                for y in range(lattice.Ny):
                    for z in range(lattice.Nz):
                        for t in range(Nt):
                            site = (x, y, z, t)
                            W = wilson_loop(config, site, R, T)
                            W_total += np.real(W) / N
                            count += 1
            W_samples.append(W_total / count)

        C[it] = np.mean(W_samples)
        C_var[it] = np.var(W_samples) / len(W_samples)

    C_err = np.sqrt(C_var)

    return CorrelatorData(t=t_arr, C=C, C_err=C_err, connected=False)


def wilson_potential(
    configs: list[LatticeConfiguration],
    R_max: int | None = None,
    T: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract static quark potential from Wilson loops.

    V(R) = -ln(W(R, T+1) / W(R, T)) / a

    Args:
        configs: List of gauge configurations
        R_max: Maximum spatial separation
        T: Temporal extent of Wilson loops

    Returns:
        Tuple of (R_array, V_array, V_error)
    """
    if not configs:
        raise ValueError("No configurations provided")

    lattice = configs[0].lattice
    if R_max is None:
        R_max = lattice.Nx // 2

    N = configs[0].N
    n_configs = len(configs)

    R_arr = np.arange(1, R_max + 1)
    V = np.zeros(R_max)
    V_err = np.zeros(R_max)

    for ir, R in enumerate(R_arr):
        # Wilson loops at T and T+1
        W_T = []
        W_T1 = []

        for config in configs:
            # Average over spatial positions
            w_t = 0.0
            w_t1 = 0.0
            count = 0
            for x in range(lattice.Nx):
                for y in range(lattice.Ny):
                    for z in range(lattice.Nz):
                        for t in range(lattice.Nt):
                            site = (x, y, z, t)
                            w_t += np.real(wilson_loop(config, site, R, T)) / N
                            w_t1 += np.real(wilson_loop(config, site, R, T + 1)) / N
                            count += 1
            W_T.append(w_t / count)
            W_T1.append(w_t1 / count)

        W_T = np.array(W_T)
        W_T1 = np.array(W_T1)

        # Ratio estimator for potential
        ratio = W_T1 / np.maximum(W_T, 1e-10)
        ratio = np.maximum(ratio, 1e-10)  # Avoid log of zero/negative

        V_samples = -np.log(ratio)
        V[ir] = np.mean(V_samples)
        V_err[ir] = np.std(V_samples) / np.sqrt(n_configs)

    return R_arr, V, V_err


def effective_mass(correlator: CorrelatorData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute effective mass from correlator.

    m_eff(t) = ln(C(t) / C(t+1))

    Args:
        correlator: Correlator data

    Returns:
        Tuple of (t_array, m_eff, m_eff_error)
    """
    t = correlator.t[:-1]
    C = np.maximum(np.abs(correlator.C), 1e-30)  # Avoid log of zero

    # Effective mass
    ratio = C[:-1] / C[1:]
    ratio = np.maximum(ratio, 1e-10)
    m_eff = np.log(ratio)

    # Error propagation (if available)
    if correlator.C_err is not None:
        # Simplified error: assume uncorrelated
        rel_err = correlator.C_err / np.maximum(np.abs(correlator.C), 1e-30)
        m_err = np.sqrt(rel_err[:-1] ** 2 + rel_err[1:] ** 2)
    else:
        m_err = np.zeros_like(m_eff)

    return t, m_eff, m_err
