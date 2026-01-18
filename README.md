# MCMC - Modelo Cosmologico de Multiples Colapsos

[![Tests](https://github.com/Modelo-cosmologico-MCMC/MCMC-Group/actions/workflows/ci.yml/badge.svg)](https://github.com/Modelo-cosmologico-MCMC/MCMC-Group/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom-green.svg)](LICENSE)

## Descripcion

El **MCMC** (Modelo Cosmologico de Multiples Colapsos) es un modelo cosmologico alternativo donde la evolucion no se parametriza primariamente por el tiempo *t* sino por una **variable entropica discreta S**.

El modelo introduce:
- **Canal determinado** (rho_m): materia efectiva
- **Canal indeterminado** (rho_id): energia cuantica virtual emergente
- **Canal latente** (rho_lat): energia "sellada" aun no liberada
- **Campo de Adrian**: campo mediador escalar/tensional

Este repositorio implementa las herramientas computacionales para:
1. Integracion del fondo cosmologico en la variable S
2. Calculo de observables (distancias, BAO, H(z), SNe)
3. Ajuste bayesiano con emcee
4. Comparacion con Lambda-CDM mediante AIC/BIC

## Quickstart

### Instalacion

```bash
# Clonar el repositorio
git clone https://github.com/Modelo-cosmologico-MCMC/MCMC-Group.git
cd MCMC-Group

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar en modo desarrollo
pip install -e .

# O instalar dependencias directamente
pip install -r requirements.txt
```

### Ejecucion Rapida

```bash
# Generar tabla de fondo cosmologico
python scripts/run_background_table.py --verbose --plot

# Ejecutar ajuste MCMC (requiere emcee)
python scripts/run_fit_emcee.py --model mcmc_refined --n-steps 2000

# Correr tests
pytest tests/ -v
```

### Uso Programatico

```python
from src.mcmc.core.s_grid import create_default_grid
from src.mcmc.core.background import solve_background, BackgroundParams

# Crear rejilla entropica
grid, S = create_default_grid()

# Resolver ecuaciones de fondo
params = BackgroundParams(H0=67.4)
sol = solve_background(S, params, grid.seals)

# Acceder a cantidades
print(f"a(S4) = {sol.a[-1]}")  # Factor de escala hoy
print(f"H(z=0) = {sol.H[-1]} km/s/Mpc")  # Hubble hoy
```

## Estructura del Proyecto

```
mcmc/
  src/mcmc/
    core/           # Motor de fondo (rejilla S, integracion)
      s_grid.py     # Rejilla entropica y sellos
      background.py # Ecuaciones de fondo
      mapping.py    # Mapeo S <-> z
      checks.py     # Validacion ontologica

    channels/       # Canales oscuros
      rho_id_parametric.py   # rho_id(z) parametrico (Nivel A)
      rho_id_ontological.py  # rho_id(S) por balances (Nivel B)
      rho_lat.py    # Canal latente
      eos.py        # Ecuacion de estado

    observables/    # Calculos cosmologicos
      distances.py  # d_C, d_A, d_L, mu
      bao.py        # Observables BAO
      hz.py         # H(z) cosmic chronometers
      sne.py        # Supernovas Ia
      likelihoods.py    # Likelihood combinado
      info_criteria.py  # AIC/BIC

    inference/      # Ajuste bayesiano
      emcee_fit.py  # Sampler MCMC
      postprocess.py # Analisis de cadenas

    config/         # Configuracion
      defaults.yaml
      priors.yaml

  tests/            # Tests unitarios
  scripts/          # Scripts de ejecucion
  docs/             # Documentacion
  data/             # Datos (external, processed, benchmarks)
```

## Modelos Disponibles

### Nivel A: Modelo Parametrico Refinado (Release v0.1)

Parametrizacion directa en z para validacion rapida:

```
rho_id(z) = rho_0 * (1+z)^3           para z > z_trans
rho_id(z) = rho_0 * [1 + eps*(z_trans - z)]   para z <= z_trans
```

**Parametros**: H0, Omega_id0, z_trans, epsilon, gamma

### Nivel B: Modelo Ontologico Completo (Release v0.4)

Balances en la variable entropica S:
```
d(rho_id)/dS = -gamma * rho_id + delta(S)
```

## Comparacion con Lambda-CDM

El modelo se valida mediante:

1. **Observables**: BAO, H(z), SNe Ia
2. **Likelihood global**: chi2_total = chi2_BAO + chi2_Hz + chi2_SNe
3. **Criterios de informacion**: AIC, BIC, AICc

```python
from src.mcmc.observables.info_criteria import compare_models

# Comparar MCMC vs Lambda-CDM
comparison = compare_models(mcmc_criteria, lcdm_criteria)
print(comparison.interpretation)
```

## Roadmap

- [x] **v0.1**: Fondo + Ajuste emcee (BAO/H(z)/SNe)
- [ ] **v0.2**: Parches CLASS/CAMB reproducibles
- [ ] **v0.3**: Cronos N-body (Delta_t modificado)
- [ ] **v0.4**: Ontologia completa (rho_lat, w_DE(z), CMB)

## Requisitos

- Python 3.9+
- numpy, scipy
- emcee (para MCMC)
- matplotlib (para plots)
- PyYAML (para configuracion)

Opcionales:
- corner (para corner plots)
- h5py (para guardar cadenas)

## Tests

```bash
# Todos los tests
pytest tests/ -v

# Tests especificos
pytest tests/test_background_monotonicity.py -v
pytest tests/test_likelihood_smoke.py -v

# Con cobertura
pytest tests/ --cov=src/mcmc --cov-report=html
```

## Documentacion

La documentacion completa esta en `docs/`:

- `docs/theory/`: Ecuaciones y fundamentos
- `docs/tutorials/`: Guias paso a paso

## Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Add nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## Referencias

- Tratado MCMC Maestro (documentacion interna)
- Documento de Simulaciones Observacionales
- Apartado Computacional: integracion y validacion

## Licencia

Ver archivo [LICENSE](LICENSE) para los terminos de uso.

## Contacto

Para preguntas o colaboraciones, abrir un Issue en este repositorio.
