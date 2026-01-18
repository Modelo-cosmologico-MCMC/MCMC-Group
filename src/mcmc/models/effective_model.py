from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mcmc.core.friedmann_effective import EffectiveParams, H_of_z
from mcmc.channels.rho_id_refined import RhoIDRefinedParams
from mcmc.observables.distances import distance_modulus
from mcmc.observables.bao_distances import dv_over_rd


@dataclass(frozen=True)
class EffectiveModelConfig:
    H0: float
    rho_b0: float
    rho0: float
    z_trans: float
    eps: float
    rd: float
    M: float


def build_effective_model(cfg: EffectiveModelConfig):
    rid = RhoIDRefinedParams(rho0=cfg.rho0, z_trans=cfg.z_trans, eps=cfg.eps)
    p = EffectiveParams(H0=cfg.H0, rho_b0=cfg.rho_b0, rho_id=rid)

    def H_model(z):
        return H_of_z(np.asarray(z, float), p)

    def mu_model(z):
        return distance_modulus(np.asarray(z, float), H_model(np.asarray(z, float)), M=cfg.M)

    def dvrd_model(z):
        return dv_over_rd(np.asarray(z, float), H_model(np.asarray(z, float)), rd=cfg.rd)

    return {"H(z)": H_model, "mu(z)": mu_model, "DVrd(z)": dvrd_model}
