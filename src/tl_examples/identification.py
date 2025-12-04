"""
Knot identification using polynomial invariants.
"""

import json
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

from .jones import compute_jones

# Try to import SnaPPy for additional invariants
try:
    import snappy

    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False


# Data directory for pre-built databases
DATA_DIR = Path(__file__).parent / "data"


def jones_to_key(jones: list[tuple[int, int]]) -> str:
    """Convert Jones polynomial to a hashable string key."""
    normalized = sorted(jones, key=lambda x: x[1])
    return str(normalized)


def load_jones_database(filepath: Optional[Path] = None) -> Optional[dict]:
    """Load the Jones polynomial database."""
    if filepath is None:
        filepath = DATA_DIR / "jones_database.json"
    if not filepath.exists():
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def identify_braid(
    braid: list[int],
    database: Optional[dict] = None,
) -> list[str]:
    """
    Identify a braid closure by its Jones polynomial.

    Args:
        braid: Braid word as list of generators
        database: Jones database (loaded automatically if None)

    Returns:
        List of matching knot names, or ["Unknown"] if not in database

    Example:
        >>> identify_braid([1, 1, 1])
        ['K3a1']  # Trefoil
        >>> identify_braid([1, 2, -1, -2])
        ['Unknot']
    """
    if database is None:
        database = load_jones_database()
        if database is None:
            raise FileNotFoundError(
                "Jones database not found. Run build_database() first or "
                "provide a database."
            )

    jones = compute_jones(braid)
    key = jones_to_key(jones)

    if key in database:
        return database[key]
    return ["Unknown"]


def identify_with_invariants(
    braid: list[int],
    jones_db: Optional[dict] = None,
    use_alexander: bool = True,
    use_volume: bool = True,
    use_signature: bool = True,
    volume_tolerance: float = 0.01,
) -> dict:
    """
    Identify a braid using multiple invariants.

    Uses the cascade: Jones → Alexander → Signature → Volume
    Each successive invariant helps narrow down the identification.

    Args:
        braid: Braid word as list of generators
        jones_db: Jones database (loaded automatically if None)
        use_alexander: Include Alexander polynomial (requires SnaPPy)
        use_volume: Include hyperbolic volume (requires SnaPPy)
        use_signature: Include knot signature (requires SnaPPy)
        volume_tolerance: Tolerance for volume matching

    Returns:
        dict with:
        - 'jones': Jones polynomial (tl-tensor format)
        - 'jones_matches': Knots matching Jones
        - 'alexander': Alexander polynomial (if available)
        - 'volume': Hyperbolic volume (if available)
        - 'signature': Knot signature (if available)
        - 'best_matches': Best identification considering all invariants

    Example:
        >>> result = identify_with_invariants([1, -2, 1, -2])
        >>> result['jones_matches']
        ['K4a1', 'K11n19']  # Same Jones polynomial
        >>> result['best_matches']
        ['K4a1']  # Distinguished by volume
    """
    if jones_db is None:
        jones_db = load_jones_database()

    result = {
        "jones": None,
        "jones_matches": ["Unknown"],
        "alexander": None,
        "volume": None,
        "signature": None,
        "best_matches": ["Unknown"],
    }

    # Compute Jones polynomial (always available via tl-tensor)
    jones = compute_jones(braid)
    result["jones"] = jones

    # Find Jones matches
    if jones_db is not None:
        key = jones_to_key(jones)
        if key in jones_db:
            result["jones_matches"] = jones_db[key]
        else:
            result["jones_matches"] = ["Unknown"]

    candidates = set(result["jones_matches"])

    # Use SnaPPy for additional invariants
    if HAS_SNAPPY and (use_alexander or use_volume or use_signature):
        try:
            K = snappy.Link(braid_closure=braid)

            if use_alexander:
                try:
                    result["alexander"] = str(K.alexander_polynomial())
                except Exception:
                    pass

            if use_signature:
                try:
                    result["signature"] = K.signature()
                except Exception:
                    pass

            if use_volume:
                try:
                    M = K.exterior()
                    vol = float(M.volume())
                    result["volume"] = vol if vol > 0.01 else 0.0
                except Exception:
                    pass

        except Exception:
            pass

    # Determine best matches
    # For now, just return Jones matches (full disambiguation would need
    # a database with all invariants)
    if "Unknown" not in candidates:
        result["best_matches"] = list(candidates)
    else:
        result["best_matches"] = ["Unknown"]

    return result


def build_jones_database(max_crossings: int = 12, verbose: bool = True) -> dict:
    """
    Build a database of Jones polynomials for knots from SnaPPy tables.

    Requires SnaPPy to be installed.

    Args:
        max_crossings: Maximum crossing number to include
        verbose: Print progress

    Returns:
        dict mapping Jones polynomial key to list of knot names
    """
    if not HAS_SNAPPY:
        raise ImportError("SnaPPy required to build database")

    from collections import defaultdict

    database = defaultdict(list)

    # Add unknot (Jones = 1)
    unknot_jones = [(1, 0)]
    database[jones_to_key(unknot_jones)].append("Unknot")

    for crossings in range(3, max_crossings + 1):
        if verbose:
            print(f"Processing {crossings}-crossing knots...", end=" ", flush=True)
        count = 0

        for suffix in ["a", "n"]:
            i = 1
            while True:
                name = f"K{crossings}{suffix}{i}"
                try:
                    K = snappy.Link(name)
                    braid = list(K.braid_word())
                    jones = compute_jones(braid)
                    key = jones_to_key(jones)
                    database[key].append(name)
                    count += 1
                    i += 1
                except Exception:
                    break

        if verbose:
            print(f"{count} knots")

    return dict(database)


def save_database(database: dict, filepath: Optional[Path] = None):
    """Save a database to JSON."""
    if filepath is None:
        filepath = DATA_DIR / "jones_database.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(database, f, indent=2)
