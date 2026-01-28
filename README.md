# MCMC - Modelo Cosmologico de Multiples Colapsos

Implementacion computacional del modelo cosmologico MCMC con arquitectura por bloques ontologicos y soporte de ajuste bayesiano (emcee) para el backend efectivo.

## Correccion Ontologica 2025

**IMPORTANTE:** El parametro entropico S tiene rango **[0, 100]**, NO [1.001, 1.0015].

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| S_MIN | 0.0 | Estado primordial (maxima superposicion) |
| S_GEOM | 1.001 | Big Bang - transicion pre-geometrica |
| S_0 | 95.07 | Presente cosmologico |
| S_MAX | 100.0 | Limite asintotico de Sitter |

**Calibracion:** `S_0 = 100 * (1 - Omega_b) = 95.07` con `Omega_b = 0.0493`

## Arquitectura Ontologica

El MCMC se organiza en dos regimenes principales definidos sobre la variable discreta entropica **S**:

### Regimenes

| Regimen | Rango S | Descripcion |
|---------|---------|-------------|
| **Pre-Geometrico** | [0, 1.001) | No existe espacio-tiempo clasico |
| **Post-Big Bang** | [1.001, 95.07] | Cosmologia observable |

### Transiciones Pre-Geometricas (S < 1.001)

| Umbral | S | Descripcion |
|--------|---|-------------|
| S_PRE_0 | 0.001 | Primera singularidad pre-geom (Planck) |
| S_PRE_1 | 0.01 | Segunda transicion pre-geom (GUT) |
| S_PRE_2 | 0.1 | Tercera transicion pre-geom |
| S_PRE_3 | 1.000 | Cuarta transicion pre-geom (EW) |
| **S_GEOM** | **1.001** | **Big Bang observable** |

### Epocas Cosmologicas Post-Big Bang (S >= 1.001)

| Umbral | S | z aprox. | Descripcion |
|--------|---|----------|-------------|
| S_GEOM | 1.001 | ∞ | Big Bang |
| S_RECOMB | 1.08 | 1100 | Recombinacion |
| S_GALAXY | 2.5 | 10 | Primeras galaxias |
| S_STAR_PEAK | 47.5 | 2 | Pico formacion estelar |
| S_Z1 | 65.0 | 1 | Referencia SNe Ia |
| S_Z05 | 84.2 | 0.5 | Era de energia oscura |
| **S_0** | **95.07** | **0** | **Presente cosmologico** |

> **CRITICO:** S_BB = S_GEOM = 1.001 es el **Big Bang observable**, **NO** "hoy".
> El presente cosmologico es S_0 = 95.07.

### Mapeo S(z)

La ecuacion maestra basada en termodinamica de Bekenstein-Hawking:

```
S(z) = S_geom + (S_0 - S_geom) / E(z)^2
```

donde:

```
E(z) = H(z)/H_0 = sqrt[Omega_m * (1+z)^3 + Omega_Lambda]
```

**Propiedades:**
- S(z=0) = S_0 ~ 95.07 (hoy)
- S(z->infinito) -> S_GEOM = 1.001 (Big Bang)
- dS/dz < 0 (monotona decreciente)

### Presente Estratificado

El modelo incluye la nocion de "presente estratificado":

```
S_local(x) = S_global * sqrt(1 - 2GM/rc^2)
```

Las islas tensoriales (agujeros negros, cumulos) experimentan S_local < S_global.

### Correspondencia con LCDM

| MCMC | LCDM | Valor |
|------|------|-------|
| Masa determinada | Omega_b | 4.93% |
| MCV (Masa Cuantica Virtual) | Omega_DM | 26.6% |
| ECV (Espacio Cuantico Virtual) | Omega_Lambda | 68.5% |

### Bloques de Implementacion

| Bloque | Rango S | Descripcion |
|--------|---------|-------------|
| Bloque 0 | [0, 0.001] | Estado pregeometrico, campo tensorial |
| Bloque I | [0.001, 1.001] | Ley de Cronos: C(S), T(S), Phi_ten(S), N(S) |
| Bloque II | [1.001, 95.07] | Cosmologia efectiva: H(z), mu(z), BAO |

## Modulos Principales

### Core (`src/mcmc/core/`)

- **ontology.py** - Constantes ontologicas, umbrales, regimenes
- **s_grid.py** - Grids entropicos pre-BB y post-BB

### Ontology (`src/mcmc/ontology/`)

- **s_map.py** - Mapeo entropico S <-> z <-> t <-> a
- **adrian_field.py** - Campo de Adrian Phi_Ad (regulador tensional)
- **dual_metric.py** - Metrica Dual Relativa g_uv(S)

### Channels (`src/mcmc/channels/`)

- **rho_lat.py** - Canal latente rho_lat(S) (Masa Cuantica Virtual)
- **q_dual.py** - Termino de intercambio Q_dual entre canales
- **lambda_rel.py** - Lambda relativista dinamica

### Growth (`src/mcmc/growth/`)

- **linear_growth.py** - Factor de crecimiento D(a), f(a)
- **f_sigma8.py** - Observable f*sigma8(z)
- **mu_eta.py** - Gravedad modificada mu(a), eta(a)

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

### Visualizaciones Ontologicas

```bash
# Generar graficas de la correccion ontologica
python reports/generate_ontology_plots.py
```

Graficas generadas en `reports/figures/`:
- `01_s_range_epochs.png` - Rango S y epocas cosmologicas
- `02_s_z_mapping.png` - Mapeo entropico S(z)
- `03_adrian_field.png` - Campo de Adrian y transiciones
- `04_channels.png` - Canales rho_lat y Q_dual
- `05_modified_gravity.png` - Gravedad modificada mu, eta
- `06_dual_metric.png` - Metrica Dual Relativa

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
  S_GEOM: 1.001      # Big Bang observable
  S_0: 95.07         # Presente cosmologico
  thresholds: [0.001, 0.01, 0.1, 0.5, 1.001]

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

### Mapeo Entropico

```python
from mcmc.ontology.s_map import EntropyMap
import numpy as np

# Crear mapa entropico
s_map = EntropyMap()

# Mapeo z -> S
z = np.array([0.0, 0.5, 1.0, 2.0])
S = s_map.S_of_z(z)
print(f"S(z=0) = {S[0]:.2f}")  # ~95.07

# Mapeo inverso S -> z
z_recovered = s_map.z_of_S(S)

# Factor de escala
a = s_map.a_of_S(S)
```

### Campo de Adrian

```python
from mcmc.ontology.adrian_field import AdrianField

# Campo con transiciones canonicas
field = AdrianField()

# Faz tensorial
S = 50.0
phi_ten = field.Phi_ten(S)

# Potencial efectivo
V = field.V_eff(phi=0.1, S=50.0)
```

### Canales

```python
from mcmc.channels.rho_lat import LatentChannel
from mcmc.channels.q_dual import QDualParams, eta_lat_of_S

# Canal latente
channel = LatentChannel()
S_arr = np.linspace(1.001, 95, 100)
rho_lat = channel.rho_lat_array(S_arr)

# Fracciones de acople
params = QDualParams()
eta_lat = eta_lat_of_S(S_arr, params)
```

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

### Cobertura de Tests

```
243 passed, 1 skipped
```

Tests por modulo:
- `test_ontology.py` - Mapa entropico, Campo de Adrian, Metrica Dual
- `test_channels_new.py` - Lambda_rel, Q_dual, canales acoplados
- `test_growth.py` - Crecimiento lineal, f*sigma8, mu/eta

## Estructura del Proyecto

```
MCMC-Group/
├── src/mcmc/
│   ├── blocks/          # Bloques ontologicos (block0, block1, block2)
│   ├── core/            # S-grid, ontology, Cronos, Friedmann
│   ├── ontology/        # s_map, adrian_field, dual_metric
│   ├── channels/        # rho_id, rho_lat, q_dual, lambda_rel
│   ├── growth/          # linear_growth, f_sigma8, mu_eta
│   ├── observables/     # Distancias, chi2, likelihoods (post-BB)
│   ├── models/          # API unificada (effective, block1, unified)
│   ├── pipeline/        # Config, run, inference
│   ├── inference/       # emcee, postprocess
│   └── data/            # IO, registry
├── configs/             # Configuraciones YAML
├── data/
│   ├── demo/            # Datos para CI
│   └── real/            # Datos reales (no versionado)
├── reports/
│   ├── figures/         # Visualizaciones PNG
│   └── *.md             # Informes
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

1. **Rango ontologico:** S in [0, 100] con S_0 ~ 95.07 (presente)
2. **Big Bang observable:** S_GEOM = 1.001 marca la transicion al universo observable
3. **Pre-geometrico:** S in [0, 1.001) - transiciones canonicas preservadas
4. **Observables post-BB:** H(z), mu(z), BAO definidos para z >= 0 (S >= S_GEOM)
5. **Friedmann normalizado:** H(z=0) = H0 exactamente (en post-BB)
6. **Correspondencia LCDM:** Omega_b -> masa determinada, Omega_DM -> MCV, Omega_Lambda -> Ep

## Licencia

Leer licencia.

## Citacion

```bibtex
@software{mcmc_group,
  title = {MCMC - Modelo Cosmologico de Multiples Colapsos},
  url = {https://github.com/Modelo-cosmologico-MCMC/MCMC-Group}
}
```
