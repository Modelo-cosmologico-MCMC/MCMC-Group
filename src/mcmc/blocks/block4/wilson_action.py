"""Wilson gauge action with MCMC S-dependent coupling.

MCMC Ontology: The gauge coupling beta(S) depends on the entropic
coordinate, allowing study of:
1. Pre-geometric strong coupling regime (S << S_GEOM)
2. Transition to geometric phase at S_GEOM
3. Present-day QCD coupling at S_0
"""
from __future__ import annotations

import numpy as np

from mcmc.core.ontology import S_GEOM
from mcmc.blocks.block4.config import LatticeParams, WilsonParams


def beta_of_S(
    S: float,
    params: WilsonParams | None = None,
) -> float:
    """MCMC coupling beta(S).

    The coupling evolves with S:
    - Pre-geometric (S < S_GEOM): Strong coupling, beta ~ 0.1
    - Transition (S ~ S_GEOM): Rapid increase
    - Geometric (S > S_GEOM): Asymptotic freedom, beta ~ 6

    Args:
        S: Entropic coordinate
        params: Wilson parameters

    Returns:
        Gauge coupling beta
    """
    if params is None:
        params = WilsonParams()

    if not params.use_mcmc_beta:
        return params.beta_0

    # Transition function
    S_trans = S_GEOM
    width = 0.1  # Width of transition

    # Smooth interpolation between regimes
    x = (S - S_trans) / width
    transition = 0.5 * (1 + np.tanh(x))

    beta = params.beta_pre_geom + (params.beta_0 - params.beta_pre_geom) * transition

    return float(beta)


def su2_generator(index: int) -> np.ndarray:
    """SU(2) Pauli matrices (generators).

    Args:
        index: Generator index (0, 1, 2)

    Returns:
        2x2 complex matrix
    """
    if index == 0:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif index == 1:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    else:
        return np.array([[1, 0], [0, -1]], dtype=complex)


def su3_generator(index: int) -> np.ndarray:
    """SU(3) Gell-Mann matrices (generators).

    Args:
        index: Generator index (0-7)

    Returns:
        3x3 complex matrix
    """
    generators = [
        # lambda_1
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
        # lambda_2
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
        # lambda_3
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
        # lambda_4
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
        # lambda_5
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
        # lambda_6
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
        # lambda_7
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
        # lambda_8
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3),
    ]
    return generators[index]


def random_su_matrix(
    N: int = 3,
    epsilon: float = 0.1,
) -> np.ndarray:
    """Generate random SU(N) matrix near identity.

    Args:
        N: Group dimension (2 or 3)
        epsilon: Step size (0 = identity, 1 = Haar random)

    Returns:
        NxN complex unitary matrix with det = 1
    """
    if N == 2:
        # SU(2) parametrization
        alpha = np.random.randn(3) * epsilon
        U = np.eye(2, dtype=complex)
        for i in range(3):
            U = U @ (np.cos(alpha[i]) * np.eye(2) +
                    1j * np.sin(alpha[i]) * su2_generator(i) / 2)
    else:
        # SU(3) parametrization
        alpha = np.random.randn(8) * epsilon
        A = sum(alpha[i] * su3_generator(i) for i in range(8)) / 2
        U = np.eye(3, dtype=complex) + 1j * A
        # Gram-Schmidt orthogonalization
        U, _ = np.linalg.qr(U)
        # Fix determinant
        U = U / np.linalg.det(U) ** (1 / N)

    return U


class LatticeConfiguration:
    """Configuration of gauge links on the lattice.

    U[x, mu] is an SU(N) matrix at site x in direction mu.
    """

    def __init__(
        self,
        lattice: LatticeParams,
        cold_start: bool = True,
    ):
        """Initialize lattice configuration.

        Args:
            lattice: Lattice parameters
            cold_start: Start from identity (True) or random (False)
        """
        self.lattice = lattice
        self.N = 3 if lattice.gauge_group == "SU3" else 2

        # Link array: [Nx, Ny, Nz, Nt, 4, N, N]
        shape = (lattice.Nx, lattice.Ny, lattice.Nz, lattice.Nt, 4, self.N, self.N)

        if cold_start:
            # Cold start: all links = identity
            self.U = np.zeros(shape, dtype=complex)
            for i in range(self.N):
                self.U[..., i, i] = 1.0
        else:
            # Hot start: random links
            self.U = np.zeros(shape, dtype=complex)
            for x in range(lattice.Nx):
                for y in range(lattice.Ny):
                    for z in range(lattice.Nz):
                        for t in range(lattice.Nt):
                            for mu in range(4):
                                self.U[x, y, z, t, mu] = random_su_matrix(self.N, 1.0)

    def get_link(self, site: tuple, mu: int) -> np.ndarray:
        """Get link U_mu(x).

        Args:
            site: Lattice site (x, y, z, t)
            mu: Direction (0=x, 1=y, 2=z, 3=t)

        Returns:
            SU(N) matrix
        """
        x, y, z, t = site
        return self.U[x, y, z, t, mu]

    def set_link(self, site: tuple, mu: int, U: np.ndarray) -> None:
        """Set link U_mu(x).

        Args:
            site: Lattice site
            mu: Direction
            U: New SU(N) matrix
        """
        x, y, z, t = site
        self.U[x, y, z, t, mu] = U

    def shift_site(self, site: tuple, mu: int, n: int = 1) -> tuple:
        """Shift site in direction mu with periodic BC.

        Args:
            site: Starting site
            mu: Direction
            n: Number of steps

        Returns:
            Shifted site
        """
        x, y, z, t = site
        L = [self.lattice.Nx, self.lattice.Ny, self.lattice.Nz, self.lattice.Nt]
        shifted = list(site)
        shifted[mu] = (shifted[mu] + n) % L[mu]
        return tuple(shifted)


def staple(
    config: LatticeConfiguration,
    site: tuple,
    mu: int,
) -> np.ndarray:
    """Compute staple sum for link U_mu(x).

    The staple is the sum of products of links forming the
    boundary of plaquettes containing U_mu(x):

        sum_nu [ U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
               + U_nu^dag(x+mu-nu) U_mu^dag(x-nu) U_nu(x-nu) ]

    Args:
        config: Lattice configuration
        site: Lattice site
        mu: Link direction

    Returns:
        Staple sum (SU(N) matrix)
    """
    N = config.N
    staple_sum = np.zeros((N, N), dtype=complex)

    for nu in range(4):
        if nu == mu:
            continue

        # Forward staple
        site_mu = config.shift_site(site, mu)
        site_nu = config.shift_site(site, nu)

        U_nu_xmu = config.get_link(site_mu, nu)
        U_mu_xnu = config.get_link(site_nu, mu)
        U_nu_x = config.get_link(site, nu)

        staple_sum += U_nu_xmu @ U_mu_xnu.conj().T @ U_nu_x.conj().T

        # Backward staple
        site_mu_minus_nu = config.shift_site(site_mu, nu, -1)
        site_minus_nu = config.shift_site(site, nu, -1)

        U_nu_xmu_mnu = config.get_link(site_mu_minus_nu, nu)
        U_mu_mnu = config.get_link(site_minus_nu, mu)
        U_nu_mnu = config.get_link(site_minus_nu, nu)

        staple_sum += U_nu_xmu_mnu.conj().T @ U_mu_mnu.conj().T @ U_nu_mnu

    return staple_sum


def plaquette(
    config: LatticeConfiguration,
    site: tuple,
    mu: int,
    nu: int,
) -> complex:
    """Compute plaquette P_munu(x).

    P_munu(x) = Tr[U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)]

    Args:
        config: Lattice configuration
        site: Lattice site
        mu, nu: Plaquette directions

    Returns:
        Trace of plaquette
    """
    site_mu = config.shift_site(site, mu)
    site_nu = config.shift_site(site, nu)

    U_mu = config.get_link(site, mu)
    U_nu_xmu = config.get_link(site_mu, nu)
    U_mu_xnu = config.get_link(site_nu, mu)
    U_nu = config.get_link(site, nu)

    P = U_mu @ U_nu_xmu @ U_mu_xnu.conj().T @ U_nu.conj().T
    return np.trace(P)


def wilson_loop(
    config: LatticeConfiguration,
    site: tuple,
    R: int,
    T: int,
    mu: int = 0,
    nu: int = 3,
) -> complex:
    """Compute Wilson loop W(R, T).

    Product of links around R×T rectangle in mu-nu plane.

    Args:
        config: Lattice configuration
        site: Starting site
        R: Spatial extent
        T: Temporal extent
        mu: Spatial direction
        nu: Temporal direction

    Returns:
        Trace of Wilson loop
    """
    N = config.N
    W = np.eye(N, dtype=complex)
    current = site

    # Bottom edge (mu direction)
    for _ in range(R):
        W = W @ config.get_link(current, mu)
        current = config.shift_site(current, mu)

    # Right edge (nu direction)
    for _ in range(T):
        W = W @ config.get_link(current, nu)
        current = config.shift_site(current, nu)

    # Top edge (mu direction, backward)
    for _ in range(R):
        current = config.shift_site(current, mu, -1)
        W = W @ config.get_link(current, mu).conj().T

    # Left edge (nu direction, backward)
    for _ in range(T):
        current = config.shift_site(current, nu, -1)
        W = W @ config.get_link(current, nu).conj().T

    return np.trace(W)


class WilsonAction:
    """Wilson gauge action with MCMC S-dependent coupling."""

    def __init__(
        self,
        lattice: LatticeParams,
        wilson: WilsonParams,
    ):
        """Initialize Wilson action.

        Args:
            lattice: Lattice parameters
            wilson: Wilson action parameters
        """
        self.lattice = lattice
        self.wilson = wilson
        self.N = 3 if lattice.gauge_group == "SU3" else 2

    def beta(self, S: float | None = None) -> float:
        """Get coupling at given S.

        Args:
            S: Entropic coordinate (default: wilson.S_current)

        Returns:
            Gauge coupling beta
        """
        if S is None:
            S = self.wilson.S_current
        return beta_of_S(S, self.wilson)

    def local_action(
        self,
        config: LatticeConfiguration,
        site: tuple,
        mu: int,
    ) -> float:
        """Local action contribution from link U_mu(x).

        S_local = beta * (1 - Re Tr(U * staple^dag) / N)

        Args:
            config: Lattice configuration
            site: Lattice site
            mu: Link direction

        Returns:
            Local action value
        """
        U = config.get_link(site, mu)
        S = staple(config, site, mu)

        # Action contribution
        action = np.real(np.trace(U @ S.conj().T)) / self.N
        return self.beta() * (1 - action / 6)  # 6 = number of staples

    def total_action(self, config: LatticeConfiguration) -> float:
        """Total Wilson action.

        S = beta * sum_P [1 - Re Tr(P) / N]

        Args:
            config: Lattice configuration

        Returns:
            Total action
        """
        total = 0.0
        for x in range(self.lattice.Nx):
            for y in range(self.lattice.Ny):
                for z in range(self.lattice.Nz):
                    for t in range(self.lattice.Nt):
                        site = (x, y, z, t)
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                P = plaquette(config, site, mu, nu)
                                total += 1 - np.real(P) / self.N

        return self.beta() * total

    def average_plaquette(self, config: LatticeConfiguration) -> float:
        """Average plaquette value.

        <P> = (1/6V) sum_P Re Tr(P) / N

        Args:
            config: Lattice configuration

        Returns:
            Average plaquette
        """
        total = 0.0
        count = 0
        for x in range(self.lattice.Nx):
            for y in range(self.lattice.Ny):
                for z in range(self.lattice.Nz):
                    for t in range(self.lattice.Nt):
                        site = (x, y, z, t)
                        for mu in range(4):
                            for nu in range(mu + 1, 4):
                                P = plaquette(config, site, mu, nu)
                                total += np.real(P) / self.N
                                count += 1

        return total / count
