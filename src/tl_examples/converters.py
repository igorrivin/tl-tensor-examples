"""
Knot notation converters using SnaPPy.

Provides conversion from various knot notations to braid words:
- PD code (Planar Diagram)
- DT code (Dowker-Thistlethwaite)
- Named knots (Rolfsen, Hoste-Thistlethwaite tables)
- Torus knots T(p,q)

All converters require SnaPPy but do NOT require Sage.
"""

from typing import Union, Optional

# SnaPPy is required for converters
try:
    import snappy

    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False


def _check_snappy():
    """Raise if SnaPPy is not available."""
    if not HAS_SNAPPY:
        raise ImportError(
            "SnaPPy is required for knot notation conversion. "
            "Install via: conda install -c conda-forge snappy (ARM) or pip install snappy (x86)"
        )


def from_pd_code(pd_code: list[tuple[int, ...]]) -> list[int]:
    """
    Convert a PD (Planar Diagram) code to a braid word.

    PD code represents a knot diagram as a list of 4-tuples, one per crossing.
    Each tuple contains the indices of the four strands meeting at that crossing,
    listed counterclockwise starting from the incoming understrand.

    Args:
        pd_code: List of 4-tuples representing crossings
                 e.g., [(7, 4, 0, 5), (3, 0, 4, 1), (1, 7, 2, 6), (5, 3, 6, 2)]

    Returns:
        Braid word as list of integers

    Example:
        >>> from_pd_code([(5, 2, 0, 3), (3, 0, 4, 1), (1, 4, 2, 5)])  # Trefoil
        [-1, -1, -1]

    Note:
        PD codes can be obtained from SnapPy, KnotInfo, or computed from diagrams.
    """
    _check_snappy()
    link = snappy.Link(pd_code)
    return list(link.braid_word())


def from_dt_code(dt_code: Union[str, list[tuple[int, ...]]]) -> list[int]:
    """
    Convert a DT (Dowker-Thistlethwaite) code to a braid word.

    DT code is a compact encoding of knot diagrams. There are several formats:
    - Numeric tuples: [(4, 6, 2)] or [(4, 6, 8, 2)]
    - String format: "DT: [(4,6,2)]"

    Args:
        dt_code: DT code as string or list of tuples

    Returns:
        Braid word as list of integers

    Example:
        >>> from_dt_code("DT: [(4,6,2)]")  # Trefoil
        [1, 1, 1]
        >>> from_dt_code([(4, 6, 8, 2)])  # Figure-8 (may need string format)
        [1, -2, 1, -2]

    Note:
        The string format "DT: ..." is more reliable for complex knots.
    """
    _check_snappy()
    if isinstance(dt_code, str):
        if not dt_code.startswith("DT:"):
            dt_code = f"DT: {dt_code}"
    else:
        # Convert list of tuples to string format
        dt_code = f"DT: {dt_code}"
    link = snappy.Link(dt_code)
    return list(link.braid_word())


def from_name(name: str) -> list[int]:
    """
    Convert a named knot to a braid word.

    Supports various naming conventions:
    - Rolfsen notation: "3_1", "4_1", "5_2", etc.
    - Hoste-Thistlethwaite: "K3a1", "K11n42", "L10a123"
    - Torus knots: "T(p,q)" e.g., "T(3,2)" for trefoil
    - Links: "L6a1", etc.

    Args:
        name: Knot name string

    Returns:
        Braid word as list of integers

    Example:
        >>> from_name("3_1")  # Trefoil
        [-1, -1, -1]
        >>> from_name("4_1")  # Figure-8
        [1, -2, 1, -2]
        >>> from_name("T(2,7)")  # T(2,7) torus knot
        [1, 1, 1, 1, 1, 1, 1]
        >>> from_name("K5a2")  # 5-crossing alternating #2
        [1, 1, 1, 1, 1]

    Note:
        This is the most convenient method for standard knots.
        Use SnapPy's database for knots up to ~19 crossings.
    """
    _check_snappy()
    link = snappy.Link(name)
    return list(link.braid_word())


def from_gauss_code(gauss_code: list[list[int]]) -> list[int]:
    """
    Convert a Gauss code to a braid word.

    Gauss code represents a knot by listing crossings in order as you
    traverse the knot. Each crossing appears twice with opposite signs
    indicating over/under.

    Args:
        gauss_code: List of lists (one per component) of signed crossing labels
                    e.g., [[1, -2, 3, -1, 2, -3]] for trefoil

    Returns:
        Braid word as list of integers

    Note:
        Gauss code support in SnapPy may be limited. Use PD code for
        better compatibility.
    """
    _check_snappy()
    # SnapPy's Gauss code handling is more limited than PD
    # Try to construct, but may fail for some codes
    link = snappy.Link(gauss_code)
    return list(link.braid_word())


def to_braid(
    knot: Union[str, list[int], list[tuple[int, ...]], list[list[int]]],
    notation: Optional[str] = None,
) -> list[int]:
    """
    Universal converter: any knot notation to braid word.

    Attempts to automatically detect the notation type, or uses the
    specified notation parameter.

    Args:
        knot: Knot specification in any supported format:
              - Braid word: [1, 1, 1]
              - Named knot: "3_1", "K4a1", "T(2,5)"
              - PD code: [(5, 2, 0, 3), (3, 0, 4, 1), (1, 4, 2, 5)]
              - DT code: "DT: [(4,6,2)]" or [(4, 6, 2)]
              - Gauss code: [[1, -2, 3, -1, 2, -3]]
        notation: Explicit notation type: "braid", "name", "pd", "dt", "gauss"
                  If None, attempts auto-detection.

    Returns:
        Braid word as list of integers

    Example:
        >>> to_braid([1, 1, 1])  # Already a braid
        [1, 1, 1]
        >>> to_braid("4_1")  # Named knot
        [1, -2, 1, -2]
        >>> to_braid("T(3,5)")  # Torus knot
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        >>> to_braid([(5, 2, 0, 3), (3, 0, 4, 1), (1, 4, 2, 5)])  # PD code
        [-1, -1, -1]
    """
    # Explicit notation
    if notation == "braid":
        return list(knot)
    if notation == "name":
        return from_name(knot)
    if notation == "pd":
        return from_pd_code(knot)
    if notation == "dt":
        return from_dt_code(knot)
    if notation == "gauss":
        return from_gauss_code(knot)

    # Auto-detection
    if isinstance(knot, str):
        # String input: named knot or DT string
        if knot.startswith("DT:"):
            return from_dt_code(knot)
        return from_name(knot)

    if isinstance(knot, list):
        if not knot:
            return []

        first = knot[0]

        # List of integers -> braid word
        if isinstance(first, int):
            return list(knot)

        # List of tuples -> PD code or DT code
        if isinstance(first, tuple):
            # PD code has 4-tuples, DT code typically has variable-length tuples
            if len(first) == 4:
                return from_pd_code(knot)
            else:
                return from_dt_code(knot)

        # List of lists -> Gauss code or nested structure
        if isinstance(first, list):
            return from_gauss_code(knot)

    raise ValueError(f"Cannot determine notation type for: {knot}")


def get_link_info(knot: Union[str, list[int], list[tuple[int, ...]]]) -> dict:
    """
    Get information about a knot from SnapPy.

    Args:
        knot: Knot in any supported notation

    Returns:
        dict with:
        - 'name': Knot name if identified
        - 'braid_word': Braid word representation
        - 'n_crossings': Number of crossings
        - 'n_components': Number of link components
        - 'pd_code': PD code representation
        - 'dt_code': DT code representation

    Example:
        >>> info = get_link_info("4_1")
        >>> info['n_crossings']
        4
    """
    _check_snappy()

    if isinstance(knot, str):
        link = snappy.Link(knot)
    elif isinstance(knot, list) and knot and isinstance(knot[0], int):
        link = snappy.Link(braid_closure=knot)
    else:
        braid = to_braid(knot)
        link = snappy.Link(braid_closure=braid)

    return {
        "name": str(link),
        "braid_word": list(link.braid_word()),
        "n_crossings": link.crossings,
        "n_components": len(link.link_components),
        "pd_code": link.PD_code(),
        "dt_code": link.DT_code(),
    }
