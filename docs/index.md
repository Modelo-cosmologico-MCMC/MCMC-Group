# MCMC - Documentacion

Bienvenido a la documentacion del **Modelo Cosmologico de Multiples Colapsos**.

## Contenido

### Teoria

- [Ontologia del modelo](theory/ontology.md)
- [Ecuaciones de fondo](theory/equations_background.md)
- [Canales de energia oscura](theory/rho_channels.md)
- [Perturbaciones con CLASS/CAMB](theory/perturbations_class_camb.md)
- [N-body Cronos](theory/nbody_cronos.md)

### Tutoriales

- [Quickstart: ajuste con emcee](tutorials/quickstart_fit_emcee.md)
- [Reproducir plots de fondo](tutorials/reproduce_background_plots.md)
- [Compilar CLASS parcheado](tutorials/class_patch_build.md)
- [Compilar CAMB parcheado](tutorials/camb_patch_build.md)

## Instalacion Rapida

```bash
pip install -e .
```

## Ejecutar Tests

```bash
pytest tests/ -v
```

## Siguiente Paso

Comienza con el [tutorial de quickstart](tutorials/quickstart_fit_emcee.md).
