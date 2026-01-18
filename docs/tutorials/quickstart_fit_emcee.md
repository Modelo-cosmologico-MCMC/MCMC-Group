# Quickstart: Ajuste Bayesiano con emcee

Este tutorial muestra como ejecutar un ajuste bayesiano del modelo MCMC
usando el sampler emcee.

## Requisitos

```bash
pip install emcee matplotlib corner
```

## Ejecucion desde Linea de Comandos

```bash
# Ajuste rapido (para pruebas)
python scripts/run_fit_emcee.py --model mcmc_refined --n-steps 1000 --n-burnin 200

# Ajuste completo
python scripts/run_fit_emcee.py --model mcmc_refined --n-walkers 64 --n-steps 10000 --n-burnin 2000
```

## Uso Programatico

```python
import numpy as np
from src.mcmc.observables.distances import DistanceCalculator
from src.mcmc.observables.bao import get_combined_bao_data
from src.mcmc.observables.hz import get_cosmic_chronometers_data
from src.mcmc.observables.sne import get_pantheon_binned_data
from src.mcmc.observables.likelihoods import CombinedLikelihood, LikelihoodConfig
from src.mcmc.channels.rho_id_parametric import RhoIdParametricParams, H_of_z_with_rho_id
from src.mcmc.inference.emcee_fit import Parameter, MCMCConfig, MCMCFitter

# 1. Cargar datos
bao_data = get_combined_bao_data()
Hz_data = get_cosmic_chronometers_data()
sne_data = get_pantheon_binned_data()

# 2. Crear likelihood
config = LikelihoodConfig()
likelihood = CombinedLikelihood(
    config=config,
    bao_data=bao_data,
    Hz_data=Hz_data,
    sne_data=sne_data
)

# 3. Definir modelo
Omega_m0 = 0.3

def model_builder(params):
    H0, Omega_id0, z_trans, epsilon, gamma = params
    rho_params = RhoIdParametricParams(
        Omega_id0=Omega_id0, z_trans=z_trans,
        epsilon=epsilon, gamma=gamma
    )
    def H_func(z):
        return H_of_z_with_rho_id(np.atleast_1d(z), H0, rho_params, Omega_m0)[0]
    dist_calc = DistanceCalculator(H_func=H_func, H0=H0)
    return dist_calc, H_func

def log_likelihood(params):
    try:
        dist_calc, H_func = model_builder(params)
        return likelihood.log_likelihood(dist_calc, H_func)
    except:
        return -np.inf

# 4. Definir parametros
parameters = [
    Parameter('H0', r'$H_0$', 67.4, 60.0, 80.0),
    Parameter('Omega_id0', r'$\Omega_{id,0}$', 0.7, 0.5, 0.9),
    Parameter('z_trans', r'$z_{trans}$', 0.5, 0.1, 2.0),
    Parameter('epsilon', r'$\epsilon$', 0.01, -0.1, 0.2),
    Parameter('gamma', r'$\gamma$', 0.0, -1.0, 1.0),
]

# 5. Ejecutar MCMC
mcmc_config = MCMCConfig(n_walkers=32, n_steps=2000, n_burnin=500)
fitter = MCMCFitter(parameters, log_likelihood, mcmc_config)
result = fitter.run()

# 6. Ver resultados
print(result.summary())
```

## Analizar Resultados

```python
# Corner plot
from src.mcmc.inference.postprocess import make_corner_plot

fig = make_corner_plot(
    result.samples,
    [p.latex for p in parameters],
    output_file='corner_mcmc.png'
)

# Estadisticas
print(f"H0 = {result.get_param_summary('H0')}")
```

## Comparar con Lambda-CDM

```python
from src.mcmc.observables.info_criteria import compute_all_criteria, compare_models

# Calcular criterios para MCMC
mcmc_criteria = compute_all_criteria(
    chi2=-2*np.max(result.log_prob),
    n_params=5,
    n_data=likelihood.n_bao + likelihood.n_Hz + likelihood.n_sne
)

# Calcular criterios para Lambda-CDM (con su propio ajuste)
lcdm_criteria = compute_all_criteria(chi2=..., n_params=2, n_data=...)

# Comparar
comparison = compare_models(mcmc_criteria, lcdm_criteria)
print(f"Delta BIC = {comparison.delta_BIC:.2f}")
print(comparison.interpretation)
```

## Siguiente Paso

Ver [como reproducir plots de fondo](reproduce_background_plots.md).
