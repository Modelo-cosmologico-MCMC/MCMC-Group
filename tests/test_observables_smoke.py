import numpy as np
from mcmc.observables.distances import distance_modulus


def test_distance_modulus_smoke():
    z = np.linspace(0.01, 1.0, 20)

    # H(z) demo: constante (no fisico, pero valido para smoke test)
    H = np.full_like(z, 70.0)

    mu = distance_modulus(z, H)
    assert mu.shape == z.shape
    assert np.isfinite(mu).all()
