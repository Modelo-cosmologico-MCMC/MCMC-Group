"""
Tests para el módulo de rejilla entrópica S.
"""

import pytest
import numpy as np

from src.mcmc.core.s_grid import (
    Seals,
    SGrid,
    create_default_grid,
    DEFAULT_DS,
    DEFAULT_S_MIN,
    DEFAULT_S_MAX,
)


class TestSeals:
    """Tests para la clase Seals."""

    def test_default_values(self):
        """Verifica valores por defecto de los sellos."""
        seals = Seals()
        assert seals.S1 == 0.010
        assert seals.S2 == 0.100
        assert seals.S3 == 1.000
        assert seals.S4 == 1.001

    def test_as_tuple(self):
        """Verifica conversión a tupla."""
        seals = Seals()
        t = seals.as_tuple()
        assert t == (0.010, 0.100, 1.000, 1.001)

    def test_as_array(self):
        """Verifica conversión a array."""
        seals = Seals()
        arr = seals.as_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 4
        np.testing.assert_array_equal(arr, [0.010, 0.100, 1.000, 1.001])

    def test_get_all(self):
        """Verifica diccionario de sellos."""
        seals = Seals()
        d = seals.get_all()
        assert d == {"S1": 0.010, "S2": 0.100, "S3": 1.000, "S4": 1.001}


class TestSGrid:
    """Tests para la clase SGrid."""

    def test_default_initialization(self):
        """Verifica inicialización por defecto."""
        grid = SGrid()
        assert grid.S_min == DEFAULT_S_MIN
        assert grid.S_max == DEFAULT_S_MAX
        assert grid.dS == DEFAULT_DS

    def test_build_grid(self):
        """Verifica construcción de la rejilla."""
        grid = SGrid()
        S = grid.build()

        # Verificar tipo
        assert isinstance(S, np.ndarray)

        # Verificar rango
        assert S[0] == grid.S_min
        assert np.isclose(S[-1], grid.S_max, atol=grid.dS)

        # Verificar monotonía
        assert np.all(np.diff(S) > 0)

    def test_build_with_forced_seals(self):
        """Verifica que los sellos están exactamente en la rejilla."""
        grid = SGrid()
        S = grid.build_with_forced_seals()

        # Verificar que todos los sellos están
        for seal_val in grid.seals.as_array():
            assert np.any(np.isclose(S, seal_val, atol=1e-12))

    def test_assert_seals_on_grid_pass(self):
        """Verifica que pasa cuando los sellos están en la rejilla."""
        grid = SGrid()
        S = grid.build_with_forced_seals()

        # No debe lanzar excepción
        grid.assert_seals_on_grid(S)

    def test_assert_seals_on_grid_fail(self):
        """Verifica que falla cuando falta un sello."""
        grid = SGrid()
        # Rejilla artificial sin el sello S2
        S = np.array([0.010, 0.050, 0.150, 1.000, 1.001])

        with pytest.raises(ValueError, match="Sello S2"):
            grid.assert_seals_on_grid(S)

    def test_get_seal_indices(self):
        """Verifica obtención de índices de sellos."""
        grid = SGrid()
        S = grid.build_with_forced_seals()
        indices = grid.get_seal_indices(S)

        assert "S1" in indices
        assert "S4" in indices

        # Verificar que los índices son correctos
        assert np.isclose(S[indices["S1"]], grid.seals.S1, atol=1e-12)
        assert np.isclose(S[indices["S4"]], grid.seals.S4, atol=1e-12)

    def test_n_nodes(self):
        """Verifica número aproximado de nodos."""
        grid = SGrid()
        n = grid.n_nodes
        # Debe ser aproximadamente (S_max - S_min) / dS + 1
        expected = int((grid.S_max - grid.S_min) / grid.dS) + 1
        assert n == expected

    def test_get_regime_mask_early(self):
        """Verifica máscara del régimen temprano."""
        grid = SGrid()
        S = grid.build_with_forced_seals()
        mask = grid.get_regime_mask(S, 'early')

        # S en régimen early debe estar entre S1 y S2
        S_early = S[mask]
        assert np.all(S_early >= grid.seals.S1)
        assert np.all(S_early < grid.seals.S2)

    def test_get_regime_mask_invalid(self):
        """Verifica error con régimen inválido."""
        grid = SGrid()
        S = grid.build()

        with pytest.raises(ValueError, match="Régimen desconocido"):
            grid.get_regime_mask(S, 'invalid_regime')

    def test_validation_S_min_S1_mismatch(self):
        """Verifica que S_min debe coincidir con S1."""
        with pytest.raises(ValueError, match="S_min"):
            SGrid(S_min=0.001)  # No coincide con S1=0.010

    def test_validation_negative_dS(self):
        """Verifica que dS debe ser positivo."""
        with pytest.raises(ValueError, match="dS"):
            SGrid(dS=-0.001)


class TestCreateDefaultGrid:
    """Tests para la función create_default_grid."""

    def test_returns_tuple(self):
        """Verifica que retorna tupla (SGrid, array)."""
        result = create_default_grid()
        assert isinstance(result, tuple)
        assert len(result) == 2

        grid, S = result
        assert isinstance(grid, SGrid)
        assert isinstance(S, np.ndarray)

    def test_seals_validated(self):
        """Verifica que los sellos están validados."""
        grid, S = create_default_grid()

        # No debe lanzar excepción
        grid.assert_seals_on_grid(S)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
