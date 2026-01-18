from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Criteria:
    chi2_min: float
    k: int
    n: int

    @property
    def aic(self) -> float:
        return self.chi2_min + 2.0 * self.k

    @property
    def bic(self) -> float:
        return self.chi2_min + self.k * np.log(max(self.n, 1))


def compare_models(m1: Criteria, m2: Criteria) -> dict:
    return {
        "delta_aic": m1.aic - m2.aic,
        "delta_bic": m1.bic - m2.bic,
        "m1": m1,
        "m2": m2,
    }
