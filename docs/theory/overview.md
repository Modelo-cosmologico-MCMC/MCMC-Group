# MCMC - Overview Teorico

## Modelo Cosmologico de Multiples Colapsos

El MCMC parametriza la evolucion cosmica mediante una **variable entropica discreta S** en lugar del tiempo cosmico t.

### Ecuaciones de Fondo

```
d ln a / dS = C(S)
dt_rel / dS = T(S) * N(S)
```

Donde:
- `a(S)`: factor de escala
- `C(S)`: funcion de expansion
- `T(S)`: cronificacion
- `N(S) = exp(Phi_ten(S))`: lapse tensional

### Sellos Ontologicos

Los sellos criticos son:
- S1 = 0.010 (inicio post-geometrico)
- S2 = 0.100 (transicion temprana)
- S3 = 1.000 (transicion tardia)
- S4 = 1.001 (hoy, normalizacion)

### Normalizacion

En S4:
- a(S4) = 1
- t_rel(S4) = 0
- H(S4) = H0

### Canal Indeterminado (rho_id)

Parametrizacion MVP (Nivel A):
```
rho_id(z) ~ rho0*(1+z)^3        para z > z_trans
rho_id(z) ~ rho0*[1 + eps*(z_trans - z)]  para z <= z_trans
```
