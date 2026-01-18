# Bloque I: Nucleo S-Grid y Ley de Cronos

## Descripcion

El Bloque I define el nucleo del modelo MCMC:
- Rejilla entropica discreta S
- Sellos ontologicos (S1, S2, S3, S4)
- Ley de Cronos: relacion t(S)
- Integracion del fondo: a(S), H(S), z(S)

## Variable Entropica S

La variable entropica S reemplaza al tiempo cosmico t como parametro
de evolucion. Los sellos ontologicos marcan transiciones criticas:

| Sello | Valor | Significado |
|-------|-------|-------------|
| S1 | 0.010 | Inicio post-geometrico |
| S2 | 0.100 | Transicion temprana |
| S3 | 1.000 | Transicion tardia |
| S4 | 1.001 | Hoy (normalizacion) |

## Ley de Cronos

La Ley de Cronos define como fluye el tiempo en funcion de S:

```
dt/dS = k_alpha * tanh(S / lambda_c)
```

Parametros:
- `lambda_c`: Escala de activacion
- `k_alpha`: Normalizacion

## Ecuaciones de Fondo

```
d ln a / dS = C(S)
dt_rel / dS = T(S) * N(S)
```

Donde N(S) = exp(Phi_ten(S)) es el factor de lapse tensional.

## Normalizacion

En S4:
- a(S4) = 1
- t_rel(S4) = 0
- H(S4) = H0

## Implementacion

- `src/mcmc/core/s_grid.py`: Rejilla y sellos
- `src/mcmc/core/cronos.py`: Ley de Cronos
- `src/mcmc/core/background.py`: Integracion del fondo
