import numpy as np
from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.hz_from_background import BackgroundHz


def test_background_hz_adapter_finite_and_monotonic():
    grid, S = create_default_grid()
    sol = solve_background(
        S,
        BackgroundParams(),
        S1=grid.seals.S1, S2=grid.seals.S2, S3=grid.seals.S3, S4=grid.seals.S4,
    )

    Hz = BackgroundHz.from_solution(sol)

    z = np.array([0.0, 0.5, 1.0, 2.0])
    H = Hz(z)

    assert np.isfinite(H).all()
    assert (H > 0).all()
