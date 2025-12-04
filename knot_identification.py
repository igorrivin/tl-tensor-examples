#!/usr/bin/env python3
"""
Knot identification via Jones polynomial matching.

Builds a database of Jones polynomials for known knots using SnaPPy's
knot tables and tl-tensor, then uses it to identify random braids.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import snappy
except ImportError:
    print("Error: SnaPPy is required. Install with: conda install -c conda-forge snappy")
    exit(1)

from tl_tensor import TLTensorNetwork, CotengraOptimizer


# Cache files for databases
CACHE_FILE = Path(__file__).parent / "jones_database.json"
FULL_CACHE_FILE = Path(__file__).parent / "knot_database.json"


def writhe(braid: list[int]) -> int:
    """Compute the writhe (sum of crossing signs) of a braid."""
    return sum(1 if g > 0 else -1 for g in braid)


def normalize_jones(jones: list[tuple[int, int]], w: int) -> list[tuple[int, int]]:
    """
    Normalize Jones polynomial by writhe.

    The unnormalized Jones polynomial V'(L) is related to the normalized V(L) by:
    V(L) = (-A^3)^{-w} V'(L)  where w is the writhe and A^4 = t

    In x-coordinates (where t = x^4), this is a shift by -3w in exponent.
    """
    shift = -3 * w
    return [(coeff, exp + shift) for coeff, exp in jones]


def compute_jones(braid: list[int], max_repeats: int = 32, normalize: bool = False) -> tuple:
    """
    Compute Jones polynomial for a braid closure.

    Args:
        braid: Braid word as list of generators
        max_repeats: Optimization repeats for cotengra
        normalize: If True, normalize by writhe to get standard Jones polynomial

    Returns:
        Jones polynomial as list of (coefficient, exponent) tuples
    """
    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=['greedy'])
    info = network.contract_info(optimize=opt)
    contracted = network.contract(optimize=info.path)
    jones = contracted.tensors[0].terms[0][1]

    if normalize:
        jones = normalize_jones(jones, writhe(braid))

    return jones


def jones_to_key(jones: list[tuple[int, int]]) -> str:
    """Convert Jones polynomial to a hashable string key."""
    # Normalize: sort by exponent
    normalized = sorted(jones, key=lambda x: x[1])
    return str(normalized)


def compute_volume(braid: list[int]) -> float:
    """
    Compute the hyperbolic volume of a braid closure.

    Returns 0.0 for non-hyperbolic knots (e.g., torus knots).
    """
    try:
        # Build link from braid closure
        K = snappy.Link(braid_closure=braid)
        M = K.exterior()
        vol = float(M.volume())
        return vol if vol > 0.01 else 0.0  # Threshold for numerical noise
    except:
        return 0.0


def compute_volume_from_name(name: str) -> float:
    """Compute hyperbolic volume for a knot by name."""
    try:
        K = snappy.Link(name)
        M = K.exterior()
        vol = float(M.volume())
        return vol if vol > 0.01 else 0.0
    except:
        return 0.0


def build_jones_database(max_crossings: int = 12, verbose: bool = True) -> dict:
    """
    Build a database of Jones polynomials for knots from SnaPPy tables.

    Returns dict mapping Jones polynomial (as string) to list of knot names.
    """
    database = defaultdict(list)

    # Also track the unknot
    unknot_jones = compute_jones([1, -1])  # Trivial braid gives unknot
    database[jones_to_key(unknot_jones)].append("Unknot")

    for crossings in range(3, max_crossings + 1):
        if verbose:
            print(f"Processing {crossings}-crossing knots...", end=" ", flush=True)
        count = 0

        for suffix in ['a', 'n']:  # alternating and non-alternating
            i = 1
            while True:
                name = f"K{crossings}{suffix}{i}"
                try:
                    K = snappy.Link(name)
                    braid = list(K.braid_word())

                    # Compute Jones polynomial
                    jones = compute_jones(braid)
                    key = jones_to_key(jones)

                    database[key].append(name)
                    count += 1
                    i += 1
                except:
                    break

        if verbose:
            print(f"{count} knots")

    return dict(database)


def save_database(database: dict, filepath: Path = CACHE_FILE):
    """Save the Jones database to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(database, f, indent=2)
    print(f"Saved database to {filepath}")


def load_database(filepath: Path = CACHE_FILE) -> dict:
    """Load the Jones database from a JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def get_database(max_crossings: int = 12, rebuild: bool = False, verbose: bool = True) -> dict:
    """Get the Jones database, building it if necessary."""
    if not rebuild:
        db = load_database()
        if db is not None:
            if verbose:
                print(f"Loaded database with {len(db)} distinct Jones polynomials")
            return db

    if verbose:
        print(f"Building Jones polynomial database (up to {max_crossings} crossings)...")
    db = build_jones_database(max_crossings, verbose)
    save_database(db)
    return db


def identify_knot(jones: list[tuple[int, int]], database: dict) -> list[str]:
    """
    Identify a knot by its Jones polynomial.

    Returns list of possible knot names, or ["Unknown"] if not in database.
    """
    key = jones_to_key(jones)
    if key in database:
        return database[key]
    return ["Unknown"]


def identify_braid(braid: list[int], database: dict) -> list[str]:
    """Identify a braid closure by computing its Jones polynomial."""
    jones = compute_jones(braid)
    return identify_knot(jones, database)


# === Full database with volumes ===

def build_full_database(max_crossings: int = 12, verbose: bool = True) -> dict:
    """
    Build a comprehensive database with Jones polynomials AND hyperbolic volumes.

    Returns dict with structure:
    {
        "knots": {name: {"jones": key, "volume": float, "crossings": int}, ...},
        "by_jones": {jones_key: [names], ...},
        "by_volume": {volume_bucket: [names], ...}
    }
    """
    knots = {}
    by_jones = defaultdict(list)

    # Add unknot
    unknot_jones = compute_jones([1, -1])
    unknot_key = jones_to_key(unknot_jones)
    knots["Unknot"] = {"jones": unknot_key, "volume": 0.0, "crossings": 0}
    by_jones[unknot_key].append("Unknot")

    for crossings in range(3, max_crossings + 1):
        if verbose:
            print(f"Processing {crossings}-crossing knots...", end=" ", flush=True)
        count = 0

        for suffix in ['a', 'n']:
            i = 1
            while True:
                name = f"K{crossings}{suffix}{i}"
                try:
                    K = snappy.Link(name)
                    braid = list(K.braid_word())

                    # Compute Jones polynomial
                    jones = compute_jones(braid)
                    key = jones_to_key(jones)

                    # Compute volume
                    vol = compute_volume_from_name(name)

                    knots[name] = {
                        "jones": key,
                        "volume": vol,
                        "crossings": crossings
                    }
                    by_jones[key].append(name)
                    count += 1
                    i += 1
                except:
                    break

        if verbose:
            print(f"{count} knots")

    return {
        "knots": knots,
        "by_jones": dict(by_jones)
    }


def save_full_database(database: dict, filepath: Path = FULL_CACHE_FILE):
    """Save the full database to JSON."""
    with open(filepath, 'w') as f:
        json.dump(database, f, indent=2)
    print(f"Saved full database to {filepath}")


def load_full_database(filepath: Path = FULL_CACHE_FILE) -> dict:
    """Load the full database from JSON."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def get_full_database(max_crossings: int = 12, rebuild: bool = False, verbose: bool = True) -> dict:
    """Get the full database, building if necessary."""
    if not rebuild:
        db = load_full_database()
        if db is not None:
            if verbose:
                print(f"Loaded full database with {len(db['knots'])} knots")
            return db

    if verbose:
        print(f"Building full database (up to {max_crossings} crossings)...")
    db = build_full_database(max_crossings, verbose)
    save_full_database(db)
    return db


def identify_braid_with_volume(braid: list[int], database: dict,
                                volume_tolerance: float = 0.01) -> list[str]:
    """
    Identify a braid using both Jones polynomial and hyperbolic volume.

    Returns list of matching knot names, sorted by volume match quality.
    """
    jones = compute_jones(braid)
    key = jones_to_key(jones)

    # First, find Jones matches
    if key not in database["by_jones"]:
        return ["Unknown"]

    candidates = database["by_jones"][key]

    if len(candidates) == 1:
        return candidates

    # Multiple candidates - use volume to distinguish
    braid_vol = compute_volume(braid)

    # Score candidates by volume match
    scored = []
    for name in candidates:
        knot_vol = database["knots"][name]["volume"]
        vol_diff = abs(braid_vol - knot_vol)
        scored.append((name, vol_diff))

    # Sort by volume difference
    scored.sort(key=lambda x: x[1])

    # Return best matches (within tolerance)
    best_diff = scored[0][1]
    matches = [name for name, diff in scored if diff <= best_diff + volume_tolerance]

    return matches


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Knot identification via Jones polynomials')
    parser.add_argument('--build', action='store_true', help='Build/rebuild the Jones-only database')
    parser.add_argument('--build-full', action='store_true', help='Build full database with volumes')
    parser.add_argument('--max-crossings', type=int, default=12, help='Max crossings for database')
    parser.add_argument('--braid', type=str, help='Identify a braid (comma-separated generators)')
    parser.add_argument('--with-volume', action='store_true', help='Use volume for disambiguation')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    args = parser.parse_args()

    if args.build:
        db = get_database(args.max_crossings, rebuild=True)
    elif args.build_full:
        db = get_full_database(args.max_crossings, rebuild=True)
    elif args.with_volume or args.braid and FULL_CACHE_FILE.exists():
        db = get_full_database(args.max_crossings)
    else:
        db = get_database(args.max_crossings)

    if args.stats:
        if "knots" in db:
            # Full database
            print(f"\nFull database statistics:")
            print(f"  Total knots: {len(db['knots'])}")
            print(f"  Distinct Jones polynomials: {len(db['by_jones'])}")

            shared = [(k, v) for k, v in db['by_jones'].items() if len(v) > 1]
            print(f"  Jones polynomials shared by multiple knots: {len(shared)}")

            # Show examples with volumes
            if shared:
                print("\n  Examples of shared Jones (with volumes):")
                for key, names in sorted(shared, key=lambda x: -len(x[1]))[:3]:
                    print(f"    {names}")
                    for n in names[:3]:
                        vol = db['knots'][n]['volume']
                        cross = db['knots'][n]['crossings']
                        print(f"      {n}: {cross} crossings, vol={vol:.4f}")
        else:
            # Jones-only database
            print(f"\nDatabase statistics:")
            print(f"  Distinct Jones polynomials: {len(db)}")
            shared = [(k, v) for k, v in db.items() if len(v) > 1]
            print(f"  Jones polynomials shared by multiple knots: {len(shared)}")
            if shared:
                print("\n  Examples of shared Jones polynomials:")
                for key, names in sorted(shared, key=lambda x: -len(x[1]))[:5]:
                    print(f"    {names}")

    if args.braid:
        word = [int(x.strip()) for x in args.braid.split(',')]

        if args.with_volume and "knots" in db:
            matches = identify_braid_with_volume(word, db)
            vol = compute_volume(word)
            print(f"\nBraid {word}")
            print(f"  Volume: {vol:.6f}")
            print(f"  Identified as: {matches}")
        else:
            if "by_jones" in db:
                matches = identify_knot(compute_jones(word), db["by_jones"])
            else:
                matches = identify_braid(word, db)
            print(f"\nBraid {word} identified as: {matches}")
