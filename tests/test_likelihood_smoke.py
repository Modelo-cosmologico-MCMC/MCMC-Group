import numpy as np
from mcmc.observables.likelihoods import loglike_total


def test_loglike_total_smoke():
    datasets = {
        "hz": {"z": np.array([0.1, 0.5]), "H": np.array([70.0, 90.0]), "sigma": np.array([5.0, 5.0])},
    }

    def H_model(z):
        z = np.asarray(z, float)
        return 70.0 + 40.0 * z

    model = {"H(z)": H_model, "mu(z)": lambda z: z * 0.0, "DVrd(z)": lambda z: z * 0.0}

    ll = loglike_total(datasets, model)
    assert np.isfinite(ll)
