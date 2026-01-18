from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.hz_from_background import BackgroundHz
from mcmc.observables.distances import distance_modulus
from mcmc.observables.bao_distances import dv_over_rd


@dataclass(frozen=True)
class Block1ModelConfig:
    # Nuisance (para observables)
    rd: float = 147.0
    M: float = -19.3
    # H0 del Bloque I
    H0: float = 67.4
    # El resto de shapes se toma del defaults (BackgroundParams()).
    # Si quieres exponerlos en YAML, lo hacemos en PR-05b.


def build_block1_model(cfg: Block1ModelConfig):
    grid, S = create_default_grid()
    p = BackgroundParams(H0=cfg.H0)

    sol = solve_background(
        S, p,
        S1=grid.seals.S1, S2=grid.seals.S2, S3=grid.seals.S3, S4=grid.seals.S4,
    )

    Hz = BackgroundHz.from_solution(sol)

    def H_model(z):
        return Hz(np.asarray(z, float))

    def mu_model(z):
        z = np.asarray(z, float)
        return distance_modulus(z, H_model(z), M=cfg.M)

    def dvrd_model(z):
        z = np.asarray(z, float)
        return dv_over_rd(z, H_model(z), rd=cfg.rd)

    return {"H(z)": H_model, "mu(z)": mu_model, "DVrd(z)": dvrd_model}
