def test_imports():
    import mcmc
    from mcmc.core.s_grid import create_default_grid
    from mcmc.core.background import solve_background, BackgroundParams
    assert mcmc.__version__
    assert create_default_grid
    assert solve_background
    assert BackgroundParams
