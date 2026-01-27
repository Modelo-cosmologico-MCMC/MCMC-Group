"""Identificador de halos para simulaciones N-body Cronos.

Implementa algoritmos estándar de identificación de halos:
    - Friends-of-Friends (FoF): Enlace por proximidad
    - Spherical Overdensity (SO): Contornos de sobredensidad

La masa de halo se define típicamente como M_200c (masa dentro
de r_200, donde ρ̄ = 200 ρ_crit).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Halo:
    """Representa un halo identificado.

    Attributes:
        id: Identificador único
        center: Centro de masa (3,)
        mass: Masa total en unidades de simulación
        r_200: Radio donde ρ̄ = 200 ρ_crit
        v_max: Velocidad circular máxima
        r_max: Radio de V_max
        n_particles: Número de partículas
        particle_ids: Índices de partículas miembro
        S_loc: Entropía local estimada (opcional)
    """
    id: int
    center: np.ndarray
    mass: float
    r_200: float = 0.0
    v_max: float = 0.0
    r_max: float = 0.0
    n_particles: int = 0
    particle_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    S_loc: float = 1.001


@dataclass
class FoFParams:
    """Parámetros para Friends-of-Friends.

    Attributes:
        b: Linking length en unidades del espaciado medio
        min_particles: Mínimo de partículas para formar halo
    """
    b: float = 0.2
    min_particles: int = 20


@dataclass
class SOParams:
    """Parámetros para Spherical Overdensity.

    Attributes:
        delta: Sobredensidad de referencia (200 para M_200c)
        rho_crit: Densidad crítica en unidades de simulación
        min_particles: Mínimo de partículas
    """
    delta: float = 200.0
    rho_crit: float = 1.0
    min_particles: int = 20


def _mean_separation(N: int, L_box: float) -> float:
    """Espaciado medio entre partículas.

    l̄ = L_box / N^{1/3}
    """
    return L_box / (N ** (1.0 / 3.0))


def fof_groups(
    x: np.ndarray,
    L_box: float,
    params: FoFParams
) -> list[np.ndarray]:
    """Encuentra grupos FoF.

    Algoritmo Friends-of-Friends: dos partículas pertenecen
    al mismo grupo si están separadas por menos de b·l̄.

    Args:
        x: Posiciones de partículas (N, 3)
        L_box: Tamaño de la caja
        params: Parámetros FoF

    Returns:
        Lista de arrays con índices de partículas por grupo
    """
    N = x.shape[0]
    l_mean = _mean_separation(N, L_box)
    linking_length = params.b * l_mean

    # Construir árbol KD
    tree = cKDTree(x, boxsize=L_box)

    # Union-Find para agrupar
    parent = np.arange(N)

    def find(i):
        if parent[i] != i:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Encontrar pares dentro del linking length
    pairs = tree.query_pairs(linking_length, output_type='ndarray')
    for i, j in pairs:
        union(i, j)

    # Agrupar por componente
    groups_dict: dict[int, list[int]] = {}
    for i in range(N):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    # Filtrar por tamaño mínimo
    groups = [
        np.array(g) for g in groups_dict.values()
        if len(g) >= params.min_particles
    ]

    return groups


def spherical_overdensity(
    x: np.ndarray,
    m: np.ndarray | None,
    center: np.ndarray,
    params: SOParams,
    r_max: float = 100.0
) -> tuple[float, float]:
    """Calcula masa y radio de sobredensidad esférica.

    Encuentra r_Δ tal que M(<r_Δ) / V(<r_Δ) = Δ · ρ_crit

    Args:
        x: Posiciones de partículas (N, 3)
        m: Masas de partículas (N,) o None
        center: Centro del halo
        params: Parámetros SO
        r_max: Radio máximo de búsqueda

    Returns:
        (M_delta, r_delta): Masa y radio de sobredensidad
    """
    N = x.shape[0]
    if m is None:
        m = np.ones(N)

    # Distancias al centro
    dx = x - center
    r = np.linalg.norm(dx, axis=1)

    # Ordenar por radio
    idx_sorted = np.argsort(r)
    r_sorted = r[idx_sorted]
    m_sorted = m[idx_sorted]

    # Masa acumulada
    M_cumsum = np.cumsum(m_sorted)

    # Densidad media encerrada
    V = (4.0 / 3.0) * np.pi * r_sorted**3
    V_safe = np.maximum(V, 1e-30)
    rho_mean = M_cumsum / V_safe

    # Encontrar donde ρ̄ cruza Δ·ρ_crit
    threshold = params.delta * params.rho_crit
    mask = rho_mean >= threshold

    if not np.any(mask):
        # No alcanza la sobredensidad
        return 0.0, 0.0

    # Último punto donde se cumple
    idx_last = np.where(mask)[0][-1]

    return float(M_cumsum[idx_last]), float(r_sorted[idx_last])


def compute_halo_properties(
    x: np.ndarray,
    v: np.ndarray,
    m: np.ndarray | None,
    particle_ids: np.ndarray,
    halo_id: int,
    so_params: SOParams
) -> Halo:
    """Calcula propiedades de un halo dado sus partículas.

    Args:
        x: Posiciones de todas las partículas
        v: Velocidades de todas las partículas
        m: Masas de todas las partículas
        particle_ids: Índices de partículas del halo
        halo_id: ID del halo
        so_params: Parámetros para cálculo SO

    Returns:
        Objeto Halo con propiedades calculadas
    """
    x_halo = x[particle_ids]
    _ = v[particle_ids]  # Reserved for future velocity analysis

    if m is not None:
        m_halo = m[particle_ids]
    else:
        m_halo = np.ones(len(particle_ids))

    # Centro de masa
    total_mass = np.sum(m_halo)
    center = np.sum(x_halo * m_halo[:, np.newaxis], axis=0) / total_mass

    # Masa y radio SO
    M_200, r_200 = spherical_overdensity(x_halo, m_halo, center, so_params)

    # Velocidad circular máxima (aproximación simple)
    dx = x_halo - center
    r = np.linalg.norm(dx, axis=1)
    idx_sorted = np.argsort(r)
    r_sorted = r[idx_sorted]
    m_sorted = m_halo[idx_sorted]

    M_cumsum = np.cumsum(m_sorted)
    r_safe = np.maximum(r_sorted, 1e-10)

    # V_circ = sqrt(G·M(<r)/r), con G=1 en unidades de simulación
    V_circ = np.sqrt(M_cumsum / r_safe)

    idx_max = np.argmax(V_circ)
    v_max = float(V_circ[idx_max])
    r_max = float(r_sorted[idx_max])

    return Halo(
        id=halo_id,
        center=center,
        mass=total_mass,
        r_200=r_200,
        v_max=v_max,
        r_max=r_max,
        n_particles=len(particle_ids),
        particle_ids=particle_ids,
    )


def find_halos(
    x: np.ndarray,
    v: np.ndarray,
    m: np.ndarray | None,
    L_box: float,
    fof_params: FoFParams | None = None,
    so_params: SOParams | None = None
) -> list[Halo]:
    """Pipeline completo de identificación de halos.

    1. FoF para identificar grupos
    2. SO para calcular M_200, r_200
    3. Calcular V_max, r_max

    Args:
        x: Posiciones (N, 3)
        v: Velocidades (N, 3)
        m: Masas (N,) o None
        L_box: Tamaño de la caja
        fof_params: Parámetros FoF
        so_params: Parámetros SO

    Returns:
        Lista de halos identificados
    """
    if fof_params is None:
        fof_params = FoFParams()
    if so_params is None:
        so_params = SOParams()

    # Encontrar grupos FoF
    groups = fof_groups(x, L_box, fof_params)

    # Calcular propiedades de cada halo
    halos = []
    for i, particle_ids in enumerate(groups):
        halo = compute_halo_properties(
            x, v, m, particle_ids, halo_id=i, so_params=so_params
        )
        halos.append(halo)

    # Ordenar por masa descendente
    halos.sort(key=lambda h: h.mass, reverse=True)

    # Reasignar IDs
    for i, halo in enumerate(halos):
        halo.id = i

    return halos


def estimate_Sloc(
    halo: Halo,
    x: np.ndarray,
    rho_background: float,
    S_BB: float = 1.001
) -> float:
    """Estima la entropía local de un halo.

    Aproximación basada en la densidad media del halo
    respecto al fondo:

    S_loc ≈ S_BB · (1 + α · log(ρ_halo / ρ_bg))

    donde α es un factor de calibración.

    Args:
        halo: Halo a analizar
        x: Posiciones de todas las partículas
        rho_background: Densidad de fondo
        S_BB: Entropía del Big Bang

    Returns:
        Estimación de S_loc
    """
    if halo.r_200 <= 0:
        return S_BB

    # Densidad media del halo
    V_halo = (4.0 / 3.0) * np.pi * halo.r_200**3
    rho_halo = halo.mass / max(V_halo, 1e-30)

    # Factor de sobredensidad
    delta = rho_halo / max(rho_background, 1e-30)

    # Mapeo fenomenológico (calibrar con simulaciones)
    alpha = 0.001  # Factor pequeño para mantener S_loc ≈ S_BB
    S_loc = S_BB * (1.0 + alpha * np.log(max(delta, 1.0)))

    return float(S_loc)
