"""
Core test suite for tl-tensor-examples.

Run with: pytest tests/
"""

import pytest


class TestJonesComputation:
    """Test Jones polynomial computation."""

    def test_trefoil(self):
        """Test trefoil knot Jones polynomial."""
        from tl_examples import compute_jones

        jones = compute_jones([1, 1, 1])
        # Sort by exponent for consistent comparison
        jones_sorted = sorted(jones, key=lambda x: x[1])
        expected = [(-1, -16), (1, -12), (1, -4)]
        assert jones_sorted == expected, f"Trefoil Jones mismatch: {jones_sorted}"

    def test_figure_eight(self):
        """Test figure-8 knot Jones polynomial."""
        from tl_examples import compute_jones

        jones = compute_jones([1, -2, 1, -2])
        jones_sorted = sorted(jones, key=lambda x: x[1])
        expected = [(1, -8), (-1, -4), (1, 0), (-1, 4), (1, 8)]
        assert jones_sorted == expected, f"Figure-8 Jones mismatch: {jones_sorted}"

    def test_unknot(self):
        """Test unknot has Jones polynomial = 1."""
        from tl_examples import compute_jones

        # The commutator [1, 2, -1, -2] is the unknot
        jones = compute_jones([1, 2, -1, -2])
        assert jones == [(1, 0)], f"Unknot should have Jones=1, got {jones}"

    def test_torus_knot_t25(self):
        """Test T(2,5) torus knot."""
        from tl_examples import compute_jones, torus_braid

        braid = torus_braid(2, 5)
        jones = compute_jones(braid)
        # T(2,5) has 5 terms
        assert len(jones) == 5, f"T(2,5) should have 5 terms, got {len(jones)}"


class TestBraidUtilities:
    """Test braid utility functions."""

    def test_n_strands(self):
        """Test strand counting."""
        from tl_examples import n_strands

        assert n_strands([1, 1, 1]) == 2
        assert n_strands([1, 2, 3]) == 4
        assert n_strands([1, -2, 3, -4]) == 5

    def test_writhe(self):
        """Test writhe computation."""
        from tl_examples import writhe

        assert writhe([1, 1, 1]) == 3
        assert writhe([1, -1]) == 0
        assert writhe([1, -2, 1, -2]) == 0

    def test_num_components(self):
        """Test component counting."""
        from tl_examples import num_components

        # Trefoil is a knot (1 component)
        assert num_components([1, 1, 1]) == 1
        # [1, -1] is a 2-component unlink
        assert num_components([1, -1]) == 2
        # Commutator is unknot (1 component)
        assert num_components([1, 2, -1, -2]) == 1

    def test_torus_braid(self):
        """Test torus braid generation."""
        from tl_examples import torus_braid

        assert torus_braid(2, 3) == [1, 1, 1]
        assert torus_braid(3, 2) == [1, 2, 1, 2]
        assert len(torus_braid(4, 5)) == 15  # (4-1) * 5 = 15

    def test_random_braid_reproducible(self):
        """Test random braid with seed is reproducible."""
        from tl_examples import random_braid

        b1 = random_braid(num_strands=4, length=20, seed=42)
        b2 = random_braid(num_strands=4, length=20, seed=42)
        assert b1 == b2


class TestKnotIdentification:
    """Test knot identification."""

    def test_identify_trefoil(self):
        """Test identifying trefoil."""
        from tl_examples import identify_braid

        matches = identify_braid([1, 1, 1])
        assert "K3a1" in matches, f"Trefoil not identified: {matches}"

    def test_identify_unknot(self):
        """Test identifying unknot."""
        from tl_examples import identify_braid

        matches = identify_braid([1, 2, -1, -2])
        assert "Unknot" in matches, f"Unknot not identified: {matches}"

    def test_identify_figure_eight(self):
        """Test identifying figure-8."""
        from tl_examples import identify_braid

        matches = identify_braid([1, -2, 1, -2])
        # Figure-8 is K4a1
        assert any("4a1" in m or "4_1" in m for m in matches), f"Figure-8 not identified: {matches}"


class TestLaTeX:
    """Test LaTeX formatting."""

    def test_braid_to_latex(self):
        """Test braid LaTeX formatting."""
        from tl_examples import braid_to_latex

        latex = braid_to_latex([1, 1, 1])
        assert "sigma" in latex.lower() or "\\sigma" in latex

    def test_braid_to_latex_compact(self):
        """Test compact braid LaTeX formatting."""
        from tl_examples import braid_to_latex

        latex = braid_to_latex([1, 1, 1], compact=True)
        assert "3" in latex  # Should have exponent 3

    def test_jones_to_latex(self):
        """Test Jones polynomial LaTeX formatting."""
        from tl_examples import jones_to_latex

        jones = [(-1, -16), (1, -12), (1, -4)]
        latex = jones_to_latex(jones, variable="t")
        assert "t" in latex


class TestConverters:
    """Test knot notation converters (require SnaPPy)."""

    @pytest.fixture(autouse=True)
    def check_snappy(self):
        """Skip if SnaPPy not available."""
        try:
            import snappy
        except ImportError:
            pytest.skip("SnaPPy not available")

    def test_from_name(self):
        """Test converting named knot to braid."""
        from tl_examples import from_name

        braid = from_name("3_1")
        assert isinstance(braid, list)
        assert all(isinstance(g, int) for g in braid)

    def test_from_name_torus(self):
        """Test converting torus knot notation."""
        from tl_examples import from_name

        braid = from_name("T(2,3)")
        assert len(braid) == 3  # T(2,3) has 3 crossings

    def test_to_braid_auto_detect(self):
        """Test universal converter auto-detection."""
        from tl_examples import to_braid

        # Already a braid
        assert to_braid([1, 1, 1]) == [1, 1, 1]
        # Named knot
        braid = to_braid("4_1")
        assert isinstance(braid, list)


class TestHybridMode:
    """Test hybrid computation mode."""

    def test_hybrid_returns_dict(self):
        """Test hybrid mode returns correct structure."""
        from tl_examples import compute_jones_hybrid

        result = compute_jones_hybrid([1, 1, 1])
        assert "result" in result
        assert "method" in result
        assert "n_strands" in result

    def test_hybrid_wide_braid_uses_tl_tensor(self):
        """Test wide braids use tl-tensor."""
        from tl_examples import compute_jones_hybrid

        # 7-strand braid should use tl-tensor
        braid = [1, 2, 3, 4, 5, 6] * 2
        result = compute_jones_hybrid(braid)
        assert result["method"] == "tl-tensor"
        assert result["n_strands"] == 7

    def test_hybrid_disable_sage_env(self):
        """Test disabling sage environment."""
        from tl_examples import compute_jones_hybrid

        result = compute_jones_hybrid([1, 1, 1], use_sage_env=False)
        # Should fall back to tl-tensor since SnaPPy needs Sage for jones
        assert result["method"] == "tl-tensor"


# Integration test
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test full pipeline: braid -> Jones -> identify."""
        from tl_examples import compute_jones, identify_braid, torus_braid

        # Generate T(2,5) braid
        braid = torus_braid(2, 5)

        # Compute Jones
        jones = compute_jones(braid)
        assert len(jones) > 0

        # Identify
        matches = identify_braid(braid)
        assert len(matches) > 0
        assert matches != ["Unknown"]
