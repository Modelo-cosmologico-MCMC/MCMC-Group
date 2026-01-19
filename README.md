# MCMC - Modelo Cosmologico de Multiples Colapsos

Implementacion computacional del modelo cosmologico MCMC con arquitectura por bloques ontologicos y soporte de ajuste bayesiano (emcee) para el backend efectivo.

## Arquitectura Ontologica

El MCMC se organiza en dos regimenes principales definidos sobre la variable discreta entropica **S**:

### Regimenes

| Regimen | Rango S | Descripcion |
|---------|--------:|-------------|
| **Pre-BB** | [0, 1.001] | Regimen primordial: 4 colapsos, fijacion de constantes |
| **Post-BB** | S > 1.001 | Cosmologia observable: expansion, estructuras, observables |

### Umbrales Ontologicos (segun Manuscrito Maestro Zenodo)

| Umbral | S aprox. | Transicion fisica |
|--------|----------|-------------------|
| Planck | 0.009 | Escala cuantico-gravitatoria |
| GUT | 0.099 | Unificacion de fuerzas |
| EW | 0.999 | Ruptura de simetria electrodebil |
| **S_BB** | **1.001** | **Confinamiento QCD / Big Bang observable** |

> **CRITICO:** S_BB = 1.001 es el **Big Bang observable** (4to latido/umbral), **NO** "hoy".
> La evolucion del universo observable ocurre para **S > 1.001**.

### Bloques de Implementacion

| Bloque | Rango S | Descripcion |
|--------|--------:|-------------|
| Bloque 0 | [0.001, 0.009] | Estado pregeometrico, campo tensional |
| Bloque I | [0.010, 1.001] | Ley de Cronos: C(S), T(S), Phi_ten(S), N(S) |
| Bloque II | Post-BB | Cosmologia efectiva: H(z), mu(z), BAO |

## Instalacion

```bash
git clone https://github.com/Modelo-cosmologico-MCMC/MCMC-Group.git
cd MCMC-Group
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Ejecucion Rapida

### CLI Unificado

```bash
# Pipeline ontologico (pre-BB: Bloque 0 -> 1)
mcmc run --config configs/run_base.yaml

# Pipeline de inferencia post-BB (evaluate o fit segun run.mode)
mcmc fit --config configs/run_base.yaml
```

### Wrapper Unificado

```bash
# Modo evaluate (log-likelihood sin MCMC)
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode evaluate

# Modo fit (emcee, solo backend effective)
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode fit

# Modo ontologico (Bloque 0 -> 1)
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode ontological
```

### Generacion de Graficas

Despues de ejecutar el pipeline:

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

Archivo base: `configs/run_base.yaml`

```yaml
run:
  run_id: "run_base"
  outdir: "outputs"
  mode: "evaluate"   # "evaluate" o "fit"
  seed: 42
  nwalkers: 32
  nsteps: 500

ontology:
  S_BB: 1.001        # Big Bang observable (4to umbral)
  thresholds: [0.009, 0.099, 0.999, 1.001]

model:
  backend: "effective"  # effective | block1 | unified

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

### Backends disponibles

| Backend | Descripcion | MCMC (emcee) |
|---------|-------------|--------------|
| `effective` | Cosmologia efectiva post-BB (rho_id parametrico) | Si (`mode=fit`) |
| `block1` | Bloque I ontologico (pre-BB) | Solo evaluacion |
| `unified` | Bloque I + puente a post-BB | Solo evaluacion |

Nota: Los observables H(z), mu(z), BAO se calculan en regimen **post-BB**.

## Uso Programatico

### Construir modelo desde config

```python
import yaml
from mcmc.models.builder import build_model_from_config
import numpy as np

cfg = yaml.safe_load(open("configs/run_base.yaml", encoding="utf-8"))
model = build_model_from_config(cfg)

z = np.linspace(0.01, 2, 200)
H = model["H(z)"](z)      # H(z) post-BB
mu = model["mu(z)"](z)    # Modulo de distancia
dvrd = model["DVrd(z)"](z)  # BAO observable
```

### Ejecutar pipeline desde Python

```python
from mcmc.pipeline import run_from_config

out = run_from_config("configs/run_base.yaml")
print(f"Log-likelihood: {out.loglike:.8f}")
print(f"Output: {out.outdir}")
```

## Datasets

### Datos Demo (CI)

```
data/demo/hz.csv   - H(z) simulado (columnas: z, H, sigma)
data/demo/sne.csv  - SNe Ia simulado (columnas: z, mu, sigma)
data/demo/bao.csv  - BAO simulado (columnas: z, dv_rd, sigma)
```

### Datos Reales

Los datos reales se guardan en `data/real/` (excluido de git).
Ver `docs/implementation/real_datasets_workflow.md` para instrucciones.

### Formatos de covarianza (opcional)

- `.npy` con matriz NxN
- `.npz` con clave `cov` o `cov_inv`

## Tests

```bash
# Linting
ruff check src tests scripts

# Tests
pytest -q

# Validacion completa
mcmc fit --config configs/run_base.yaml
python scripts/plot_run.py --latest
```

## Estructura del Proyecto

```
MCMC-Group/
├── src/mcmc/
│   ├── blocks/          # Bloques ontologicos (block0, block1, block2)
│   ├── core/            # S-grid, ontology, Cronos, Friedmann
│   ├── channels/        # rho_id, rho_lat
│   ├── observables/     # Distancias, chi2, likelihoods (post-BB)
│   ├── models/          # API unificada (effective, block1, unified)
│   ├── pipeline/        # Config, run, inference
│   ├── inference/       # emcee, postprocess
│   └── data/            # IO, registry
├── configs/             # Configuraciones YAML
├── data/
│   ├── demo/            # Datos para CI
│   └── real/            # Datos reales (no versionado)
├── scripts/             # Scripts de ejecucion y plotting
├── tests/               # Tests pytest
└── docs/
    ├── theory/          # Documentacion teorica + contrato ecuaciones
    └── implementation/  # Guias tecnicas
```

## Contrato Teorico

El archivo `docs/theory/MCMC_ECUACIONES_BASE.txt` define el contrato de ecuaciones fundamentales.
Cualquier cambio en ecuaciones debe reflejarse en este documento.

Referencia: [MCMC Maestro en Zenodo](https://zenodo.org/records/15556310)

## Invariantes del Modelo

1. **Dualidad de masas:** Mp(S) + Ep(S) = 1.0 para todo S (pre-BB)
2. **Big Bang observable:** S_BB = 1.001 marca la transicion al universo observable
3. **Observables post-BB:** H(z), mu(z), BAO definidos para z >= 0 (S > S_BB)
4. **Friedmann normalizado:** H(z=0) = H0 exactamente (en post-BB)

## Licencia

MIT

## Citacion

```bibtex
@software{mcmc_group,
  title = {MCMC - Modelo Cosmologico de Multiples Colapsos},
  url = {https://github.com/Modelo-cosmologico-MCMC/MCMC-Group}
}
```
