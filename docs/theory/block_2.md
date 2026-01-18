# Bloque II: Cosmologia Efectiva

## Descripcion

El Bloque II implementa el Friedmann modificado con los canales oscuros:
- Canal indeterminado rho_id(z)
- Canal latente rho_lat(S)
- Ecuacion de estado efectiva w_eff(z)

## Friedmann Modificado

```
E(z)^2 = (H/H0)^2 = Omega_r * (1+z)^4 + Omega_m * (1+z)^3
                  + Omega_k * (1+z)^2 + Omega_DE
                  + rho_id(z) + rho_lat(S)
```

## Canal Indeterminado rho_id

Parametrizacion (Nivel A):

```
rho_id(z) = rho0 * (1+z)^3           para z > z_trans
rho_id(z) = rho0 * [1 + eps*(z_trans - z)]  para z <= z_trans
```

Parametros:
- `rho0`: Amplitud normalizada
- `z_trans`: Redshift de transicion
- `eps`: Pendiente post-transicion

## Canal Latente rho_lat

```
rho_lat(S) = amp / (1 + exp(-(S - S0) / width))
```

Parametros:
- `amp`: Amplitud
- `S0`: Centro de activacion
- `width`: Ancho de transicion

## Ecuacion de Estado Efectiva

```
w_eff = -1 - (2/3) * (1/E^2) * dE^2/d(ln a)
```

## Parametro de Deceleracion

```
q = -1 - d(ln H)/d(ln a)
```

## Implementacion

- `src/mcmc/core/friedmann.py`: Friedmann modificado
- `src/mcmc/channels/rho_id_parametric.py`: Canal rho_id
- `src/mcmc/channels/rho_lat_parametric.py`: Canal rho_lat
