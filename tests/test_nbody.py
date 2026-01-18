import numpy as np
import pytest
from mcmc.nbody.kdk import State, kdk_step, integrate
from mcmc.nbody.crono_step import CronosStepParams, delta_t_cronos, global_timestep
from mcmc.nbody.poisson import PoissonParams, direct_acceleration, estimate_density


def test_state_creation():
    x = np.random.randn(10, 3)
    v = np.random.randn(10, 3)
    state = State(x=x, v=v)

    assert state.N == 10
    assert state.x.shape == (10, 3)
    assert state.v.shape == (10, 3)


def test_state_kinetic_energy():
    x = np.zeros((2, 3))
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    state = State(x=x, v=v)

    assert state.kinetic_energy() == pytest.approx(1.0)


def test_kdk_step_preserves_shape():
    x = np.random.randn(5, 3)
    v = np.random.randn(5, 3)
    state = State(x=x, v=v)

    def acc_fn(x):
        return -0.1 * x  # harmonic potential

    new_state = kdk_step(state, 0.01, acc_fn)

    assert new_state.x.shape == state.x.shape
    assert new_state.v.shape == state.v.shape


def test_integrate_returns_states():
    x = np.random.randn(3, 3)
    v = np.random.randn(3, 3)
    state = State(x=x, v=v)

    def acc_fn(x):
        return -0.1 * x

    states = integrate(state, acc_fn, 0.01, 10, save_every=2)

    assert len(states) == 6  # initial + 5 saved


def test_delta_t_cronos_shape():
    acc_norm = np.array([1.0, 2.0, 3.0])
    rho = np.array([0.5, 1.0, 1.5])
    p = CronosStepParams()

    dt = delta_t_cronos(acc_norm, 1.0, rho, p)

    assert dt.shape == acc_norm.shape
    assert np.all(dt > 0)


def test_global_timestep():
    dt_particles = np.array([0.1, 0.05, 0.2])

    dt = global_timestep(dt_particles, safety=0.5)

    assert dt == pytest.approx(0.025)


def test_direct_acceleration_shape():
    x = np.random.randn(4, 3)
    p = PoissonParams()

    acc = direct_acceleration(x, None, p)

    assert acc.shape == x.shape


def test_estimate_density_positive():
    x = np.random.randn(5, 3)

    rho = estimate_density(x, None, h=0.5)

    assert rho.shape == (5,)
    assert np.all(rho >= 0)
