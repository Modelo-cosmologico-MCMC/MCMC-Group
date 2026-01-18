import numpy as np
from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.checks import assert_background_ok


def test_background_solution_ok():
    grid, S = create_default_grid()
    sol = solve_background(
        S,
        BackgroundParams(),
        S1=grid.seals.S1,
        S2=grid.seals.S2,
        S3=grid.seals.S3,
        S4=grid.seals.S4,
    )
    assert_background_ok(sol)
    assert np.isfinite(sol["H"]).all()
    assert np.isfinite(sol["a"]).all()
    assert np.isfinite(sol["z"]).all()
