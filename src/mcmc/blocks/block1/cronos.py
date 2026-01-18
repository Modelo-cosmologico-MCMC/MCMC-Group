from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CronosParams:
    # C(S): rígido temprano, inercial tardío (MVP suave)
    C_early: float = 2.2
    C_mid: float = 1.7
    C_late: float = 1.05

    # transiciones alrededor de sellos
    S1: float = 0.010
    S2: float = 0.100
    S3: float = 1.000
    S4: float = 1.001

    # T(S) base y picos en sellos
    T0: float = 1.0
    T_peak_amp: float = 0.15
    T_peak_w: float = 0.010

    # Phi_ten envolvente + bultos
    phi_env_amp: float = 0.7
    phi_env_tau: float = 0.6
    phi_bump_amp: float = 0.8
    phi_bump_w: float = 0.020


def _gauss(x: float, mu: float, sig: float) -> float:
    sig = max(sig, 1e-12)
    u = (x - mu) / sig
    return math.exp(-0.5 * u * u)


class CronosModel:
    def __init__(self, p: CronosParams) -> None:
        self.p = p

    def C(self, S: float) -> float:
        # interpolación por tramos suave: early -> mid -> late
        if S < self.p.S2:
            return self.p.C_early
        if S < self.p.S3:
            return self.p.C_mid
        return self.p.C_late

    def phi_ten(self, S: float) -> float:
        # envolvente decreciente + bultos en sellos (S1,S2,S3)
        env = self.p.phi_env_amp * math.exp(-S / max(self.p.phi_env_tau, 1e-12))
        bumps = self.p.phi_bump_amp * (
            _gauss(S, self.p.S1, self.p.phi_bump_w)
            + _gauss(S, self.p.S2, self.p.phi_bump_w)
            + _gauss(S, self.p.S3, self.p.phi_bump_w)
        )
        return env + bumps

    def N(self, S: float) -> float:
        return math.exp(self.phi_ten(S))

    def T(self, S: float) -> float:
        # base + picos en sellos (cronificación intensificada cerca de colapsos)
        peaks = self.p.T_peak_amp * (
            _gauss(S, self.p.S1, self.p.T_peak_w)
            + _gauss(S, self.p.S2, self.p.T_peak_w)
            + _gauss(S, self.p.S3, self.p.T_peak_w)
        )
        return self.p.T0 * (1.0 + peaks)
