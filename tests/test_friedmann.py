import numpy as np
import pytest
from mcmc.core.friedmann import (
    FriedmannParams, E2_of_z, E2_of_z_S, H_of_z, w_eff_of_z_S, q_of_z_S
)
from mcmc.channels.rho_id_parametric import RhoIDParams
from mcmc.channels.rho_lat_parametric import RhoLatParams


def test_E2_of_z_positive():
    z = np.linspace(0, 2, 50)
    p = FriedmannParams()
    e2 = E2_of_z(z, p)

    assert np.all(e2 > 0)
    assert np.all(np.isfinite(e2))


def test_E2_of_z_today():
    z = np.array([0.0])
    p = FriedmannParams()
    e2 = E2_of_z(z, p)

    # E(z=0) = 1 para LCDM plano
    assert e2[0] == pytest.approx(1.0, rel=1e-10)


def test_H_of_z_today():
    z = np.array([0.0])
    p = FriedmannParams(H0=70.0)
    H = H_of_z(z, p)

    assert H[0] == pytest.approx(70.0, rel=1e-10)


def test_E2_of_z_S_without_corrections():
    z = np.linspace(0, 2, 50)
    S = np.linspace(0.01, 1.0, 50)
    p = FriedmannParams(
        rho_id=RhoIDParams(rho0=0.0),
        rho_lat=RhoLatParams(enabled=False)
    )

    e2_base = E2_of_z(z, p)
    e2_full = E2_of_z_S(z, S, p)

    np.testing.assert_allclose(e2_base, e2_full, rtol=1e-10)


def test_w_eff_shape():
    z = np.linspace(0.01, 2, 50)
    S = np.linspace(0.01, 1.0, 50)
    p = FriedmannParams()

    w = w_eff_of_z_S(z, S, p)
    assert w.shape == z.shape
    assert np.all(np.isfinite(w))


def test_q_shape():
    z = np.linspace(0.01, 2, 50)
    S = np.linspace(0.01, 1.0, 50)
    p = FriedmannParams()

    q = q_of_z_S(z, S, p)
    assert q.shape == z.shape
    assert np.all(np.isfinite(q))
