import numpy as np
import pytest
from mcmc.core.cronos import CronosParams, dt_dS, t_of_S


def test_dt_dS_shape():
    S = np.linspace(0.01, 1.0, 100)
    p = CronosParams()
    result = dt_dS(S, p)

    assert result.shape == S.shape
    assert np.all(np.isfinite(result))


def test_dt_dS_positive():
    S = np.linspace(0.01, 1.0, 100)
    p = CronosParams()
    result = dt_dS(S, p)

    assert np.all(result > 0)


def test_dt_dS_monotonic():
    S = np.linspace(0.01, 1.0, 100)
    p = CronosParams()
    result = dt_dS(S, p)

    # dt/dS should increase with S (tanh behavior)
    assert np.all(np.diff(result) >= 0)


def test_t_of_S_monotonic():
    S = np.linspace(0.01, 1.0, 100)
    p = CronosParams()
    t = t_of_S(S, p)

    assert t[0] == 0.0
    assert np.all(np.diff(t) > 0)


def test_t_of_S_invalid_grid():
    S = np.array([1.0, 0.5, 0.1])  # decreasing
    p = CronosParams()

    with pytest.raises(ValueError):
        t_of_S(S, p)
