# MCMC - Modelo Cosmologico de Multiples Colapsos

Implementacion computacional del modelo cosmologico MCMC con arquitectura por bloques ontologicos y soporte de ajuste bayesiano (emcee) para el backend efectivo.

## Arquitectura

El MCMC se organiza en bloques ontologicos definidos sobre la variable discreta entropica **S**:

| Bloque | Rango S | Descripcion |
|--------|--------:|-------------|
| Bloque 0 | [0.001, 0.009] | Estado pregeometrico y colapso primordial |
| Bloque I | [0.010, 1.001] | Nucleo ontologico: Ley de Cronos C(S), T(S), Phi_ten(S), N(S) |
| Bloque II | [0.010, 1.001] | Cosmologia efectiva: Friedmann normalizado y observables |

### Sellos ontologicos
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
pip install -e ".[dev]"
```

## Ejecucion Rapida

### CLI Unificado

```bash
# Pipeline ontologico (Bloque 0 -> 1 -> 2)
mcmc run --config configs/run_base.yaml

# Pipeline de inferencia (evaluate o fit segun run.mode en config)
mcmc fit --config configs/run_base.yaml
```

### Wrapper Unificado

```bash
# Modo evaluate (log-likelihood sin MCMC)
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode evaluate

# Modo fit (emcee, solo backend effective)
python scripts/wrappers/run_pipeline.py --config configs/run_base.yaml --mode fit

# Modo ontologico (Bloque 0 -> 1 -> 2)
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

model:
  backend: "effective"  # effective | block1 | unified

data:
  hz: "data/demo/hz.csv"
  hz_cov: null
  sne: "data/demo/sne.csv"
  sne_cov: null
  bao: "data/demo/bao.csv"
  bao_cov: null

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
| `effective` | Bloque II parametrizado (rho_id + nuisance rd, M) | Si (`mode=fit`) |
| `block1` | Bloque I ontologico directo | Solo evaluacion |
| `unified` | Bloque I + puente densidades | Solo evaluacion |

Nota: si `backend != effective` y `mode=fit`, el sistema ejecuta evaluacion y notifica en summary.txt.

## Uso Programatico

### Construir modelo desde config

```python
import yaml
from mcmc.models.builder import build_model_from_config
import numpy as np

cfg = yaml.safe_load(open("configs/run_base.yaml", encoding="utf-8"))
model = build_model_from_config(cfg)

z = np.linspace(0.01, 2, 200)
H = model["H(z)"](z)
mu = model["mu(z)"](z)
dvrd = model["DVrd(z)"](z)
```

### Ejecutar pipeline desde Python

```python
from mcmc.pipeline import run_from_config

out = run_from_config("configs/run_base.yaml")
print(f"Log-likelihood: {out.loglike:.8f}")
print(f"Output: {out.outdir}")
```

### Pipeline ontologico

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
│   ├── core/            # S-grid, background, Friedmann
│   ├── channels/        # rho_id, rho_lat
│   ├── observables/     # Distancias, chi2, likelihoods
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

## Invariantes del Modelo

1. **Dualidad de masas:** Mp(S) + Ep(S) = 1.0 para todo S
2. **Normalizacion:** a(S4) = 1, z(S4) = 0, H(S4) = H0
3. **Friedmann normalizado:** H(z=0) = H0 exactamente

## Licencia

MIT

## Citacion

```bibtex
@software{mcmc_group,
  title = {MCMC - Modelo Cosmologico de Multiples Colapsos},
  url = {https://github.com/Modelo-cosmologico-MCMC/MCMC-Group}
}
```
