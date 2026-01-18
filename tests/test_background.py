import numpy as np
from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background
from mcmc.core.checks import assert_background_ok


def test_background_solution_ok():
    grid, S = create_default_grid()
    sol = solve_background(S, BackgroundParams())
    assert_background_ok(sol)
    assert np.isfinite(sol["H"]).all()
