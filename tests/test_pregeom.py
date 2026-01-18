import pytest
from mcmc.pregeom.s0_state import PreGeomParams, InitialConditions, compute_initial_conditions


def test_pregeom_default():
    p = PreGeomParams()
    ic = compute_initial_conditions(p)

    assert ic.Mp_pre + ic.Ep_pre == pytest.approx(1.0)
    assert ic.Ep_pre == pytest.approx(p.eps)
    assert ic.S_start == p.S_start


def test_pregeom_custom_eps():
    p = PreGeomParams(eps=0.05)
    ic = compute_initial_conditions(p)

    assert ic.Ep_pre == pytest.approx(0.05)
    assert ic.Mp_pre == pytest.approx(0.95)


def test_pregeom_invalid_eps():
    with pytest.raises(ValueError):
        compute_initial_conditions(PreGeomParams(eps=0.0))

    with pytest.raises(ValueError):
        compute_initial_conditions(PreGeomParams(eps=0.6))


def test_initial_conditions_to_dict():
    ic = InitialConditions(Mp_pre=0.99, Ep_pre=0.01, phi_pre=0.0, k_pre=1.0, S_start=0.01)
    d = ic.to_dict()

    assert d["Mp_pre"] == 0.99
    assert d["Ep_pre"] == 0.01
    assert "S_start" in d
