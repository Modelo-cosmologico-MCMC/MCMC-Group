from __future__ import annotations

import numpy as np
import emcee


def run_emcee(logprob_fn, x0: np.ndarray, *, nwalkers: int = 32, nsteps: int = 2000, seed: int = 42):
    """
    Ejecuta emcee con inicializacion gaussiana alrededor de x0.
    """
    rng = np.random.default_rng(seed)
    ndim = len(x0)
    p0 = x0 + 1e-3 * rng.standard_normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob_fn)
    sampler.run_mcmc(p0, nsteps, progress=False)
    return sampler
