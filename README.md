# MCMC - Modelo Cosmologico de Multiples Colapsos

Implementacion computacional completa del modelo cosmologico MCMC con arquitectura de bloques ontologicos y ajuste bayesiano.

## Arquitectura

El modelo MCMC se organiza en tres bloques ontologicos:

| Bloque | Rango S | Descripcion |
|--------|---------|-------------|
| **Bloque 0** | [0.001, 0.009] | Estado pre-geometrico, colapso primordial |
| **Bloque I** | [0.010, 1.001] | Nucleo ontologico, Ley de Cronos (C, T, Phi_ten, N) |
| **Bloque II** | [0.010, 1.001] | Cosmologia efectiva, Friedmann normalizado |

**Sellos ontologicos:**
- S1 = 0.010 (inicio post-geometrico)
- S2 = 0.100 (transicion temprana)
- S3 = 1.000 (transicion tardia)
- S4 = 1.001 (hoy, normalizacion)

## Instalacion

```bash
git clone https://github.com/Modelo-cosmologico-MCMC/MCMC-Group.git
cd MCMC-Group
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

## Ejecucion Rapida

### CLI Unificado

```bash
# Pipeline ontologico (Bloque 0 -> 1 -> 2)
mcmc run --config configs/run_base.yaml

# Pipeline de inferencia (evaluar log-likelihood)
mcmc fit --config configs/run_base.yaml

# Pipeline de inferencia (ajuste MCMC con emcee)
# Modificar mode: "fit" en el config
mcmc fit --config configs/run_fit.yaml
```

### Scripts Wrapper

```bash
# Wrapper unificado
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode evaluate
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode ontological

# Tabla de fondo clasica
python scripts/run_background_table.py --plot --out background_table.csv

# Ajuste MCMC directo
python scripts/run_fit_emcee.py --nsteps 500 --nwalkers 32
```

### Generacion de Graficas

Despues de ejecutar el pipeline, genera graficas automaticamente:

```bash
# Especificar directorio de salida
python scripts/plot_run.py --outdir outputs/run_base

# Auto-detectar el run mas reciente
python scripts/plot_run.py --latest

# Personalizar rango de z
python scripts/plot_run.py --latest --zmax 3.0
```

Graficas generadas en `outputs/<run_id>/plots/`:
- `Hz.png`, `mu.png`, `bao_dvrd.png` - Curvas del modelo + datos
- `*_residuals.png` - Residuos (modelo - datos)
- `trace_*.png`, `posterior_*.png` - Diagnosticos de cadena (si existe chain.npy)

## Configuracion

El archivo `configs/run_base.yaml` define todos los parametros:

```yaml
run:
  run_id: "run_base"
  outdir: "outputs"
  mode: "evaluate"     # "fit" o "evaluate"
  seed: 42
  nwalkers: 32
  nsteps: 500

model:
  backend: "effective" # effective | block1 | unified

data:
  hz: "data/demo/hz.csv"
  sne: "data/demo/sne.csv"
  bao: "data/demo/bao.csv"

effective:
  H0: 67.4
  rho_b0: 0.30
  rd: 147.0
  M: -19.3
  rho_id:
    rho0: 0.70
    z_trans: 1.0
    eps: 0.05
```

## Backends de Modelo

| Backend | Descripcion | Ajuste MCMC |
|---------|-------------|-------------|
| `effective` | Bloque II parametrico (rho_id) | Si |
| `block1` | Bloque I ontologico directo | Evaluacion |
| `unified` | Bloque I + puente de densidad | Evaluacion |

## Uso Programatico

### Pipeline Ontologico

```python
from mcmc.pipeline import load_config, run_pipeline

cfg = load_config("configs/run_base.yaml")
out = run_pipeline(cfg)

# Bloque 0
print(f"Mp_pre final: {out.block0.Mp_pre[-1]:.6f}")

# Bloque 1
print(f"a(S4): {out.block1.a[-1]:.10f}")
print(f"z(S4): {out.block1.z[-1]:.10f}")

# Bloque 2
print(f"H_eff(z=0): {out.block2.H_eff[-1]:.4f} km/s/Mpc")
```

### Pipeline de Inferencia

```python
from mcmc.pipeline import run_from_config

out = run_from_config("configs/run_base.yaml")
print(f"Log-likelihood: {out.loglike:.8f}")
print(f"Output: {out.outdir}")
```

### Modelo Unificado

```python
from mcmc.models import build_model_from_config

cfg = {
    "model": {"backend": "effective"},
    "effective": {"H0": 67.4, "rho_b0": 0.30, "rd": 147.0, "M": -19.3,
                  "rho_id": {"rho0": 0.70, "z_trans": 1.0, "eps": 0.05}}
}
model = build_model_from_config(cfg)

import numpy as np
z = np.linspace(0, 2, 100)
H = model.H_of_z(z)
mu = model.mu_of_z(z)
```

## Datasets

### Datos Demo (CI)

```
data/demo/hz.csv   - H(z) simulado
data/demo/sne.csv  - SNe Ia simulado
data/demo/bao.csv  - BAO simulado
```

### Datos Reales

Los datos reales se guardan en `data/real/` (excluido de git).
Ver `docs/implementation/real_datasets_workflow.md` para instrucciones.

Formatos soportados:
- CSV con columnas: z, value, error
- Matrices de covarianza opcionales

## Tests

```bash
# Todos los tests
pytest -q

# Tests especificos
pytest tests/test_pr06_pipeline.py -v
pytest tests/test_pr07_pipeline_smoke.py -v
```

## Estructura del Proyecto

```
MCMC-Group/
├── src/mcmc/
│   ├── blocks/
│   │   ├── block0/       # Pre-geometrico
│   │   ├── block1/       # Cronos
│   │   └── block2/       # Efectivo
│   ├── core/             # S-grid, background, Friedmann
│   ├── channels/         # rho_id, rho_lat
│   ├── observables/      # Distancias, chi2, likelihoods
│   ├── models/           # API unificada de modelos
│   ├── pipeline/         # Config, run, inference
│   ├── inference/        # emcee, postprocess
│   └── data/             # IO, registry
├── configs/              # Configuraciones YAML
├── data/
│   ├── demo/             # Datos para CI
│   └── real/             # Datos reales (no en git)
├── scripts/              # Scripts de ejecucion
├── tests/                # Tests pytest
└── docs/
    ├── theory/           # Documentacion teorica
    └── implementation/   # Documentacion tecnica
```

## Documentacion

- `docs/theory/MCMC_ECUACIONES_BASE.txt` - Contrato de ecuaciones
- `docs/theory/overview.md` - Vision general
- `docs/theory/block_*.md` - Documentacion por bloque
- `docs/implementation/` - Guias tecnicas

## Invariantes del Modelo

1. **Dualidad de masas:** Mp(S) + Ep(S) = 1.0 para todo S
2. **Normalizacion:** a(S4) = 1, z(S4) = 0, H(S4) = H0
3. **Friedmann normalizado:** H(z=0) = H0 exactamente

## Licencia

MIT

## Citacion

Si utilizas este codigo, por favor cita:
```
@software{mcmc_group,
  title = {MCMC - Modelo Cosmologico de Multiples Colapsos},
  url = {https://github.com/Modelo-cosmologico-MCMC/MCMC-Group}
}
```
