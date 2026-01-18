# MCMC - Modelo Cosmologico de Multiples Colapsos

Implementacion computacional (MVP) para:
1. Integracion del fondo cosmologico en la variable entropica discreta S
2. Calculo de observables basicos (distancias, H(z), BAO demo, SNe demo)
3. Ajuste bayesiano con `emcee`
4. Tests reproducibles en GitHub Actions

## Instalacion

```bash
git clone https://github.com/Modelo-cosmologico-MCMC/MCMC-Group.git
cd MCMC-Group
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

## Ejecucion rapida

### Tabla de fondo

```bash
python scripts/run_background_table.py --plot --out background_table.csv
```

### Ajuste MCMC (emcee) con datos demo

```bash
python scripts/run_fit_emcee.py --nsteps 500 --nwalkers 32
```

## Uso programatico

```python
from mcmc.core.s_grid import create_default_grid
from mcmc.core.background import BackgroundParams, solve_background

grid, S = create_default_grid()
params = BackgroundParams(H0=67.4)
sol = solve_background(S, params)

print(sol["a"][-1])  # a(S4)=1
print(sol["H"][-1])  # H(z=0)=H0
```

## Tests

```bash
pytest -q
```

## Nota sobre el acoplo completo MCMC

Este repositorio es un baseline estable (Release v0.1). El siguiente paso es acoplar:

* `rho_id(z)` / `rho_lat(S)` al Friedmann modificado,
* y, posteriormente, CLASS/CAMB y Cronos N-body.
