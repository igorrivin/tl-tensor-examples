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


# Cache file for the Jones polynomial database
CACHE_FILE = Path(__file__).parent / "jones_database.json"


def compute_jones(braid: list[int], max_repeats: int = 32) -> tuple:
    """Compute Jones polynomial for a braid closure."""
    network = TLTensorNetwork.from_word(braid)
    opt = CotengraOptimizer(max_repeats=max_repeats, methods=['greedy'])
    info = network.contract_info(optimize=opt)
    contracted = network.contract(optimize=info.path)
    return contracted.tensors[0].terms[0][1]


def jones_to_key(jones: list[tuple[int, int]]) -> str:
    """Convert Jones polynomial to a hashable string key."""
    # Normalize: sort by exponent
    normalized = sorted(jones, key=lambda x: x[1])
    return str(normalized)


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


# Command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Knot identification via Jones polynomials')
    parser.add_argument('--build', action='store_true', help='Build/rebuild the database')
    parser.add_argument('--max-crossings', type=int, default=12, help='Max crossings for database')
    parser.add_argument('--braid', type=str, help='Identify a braid (comma-separated generators)')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    args = parser.parse_args()

    if args.build:
        db = get_database(args.max_crossings, rebuild=True)
    else:
        db = get_database(args.max_crossings)

    if args.stats:
        print(f"\nDatabase statistics:")
        print(f"  Distinct Jones polynomials: {len(db)}")

        # Count knots with same Jones polynomial
        shared = [(k, v) for k, v in db.items() if len(v) > 1]
        print(f"  Jones polynomials shared by multiple knots: {len(shared)}")
        if shared:
            print("\n  Examples of shared Jones polynomials:")
            for key, names in sorted(shared, key=lambda x: -len(x[1]))[:5]:
                print(f"    {names}")

    if args.braid:
        word = [int(x.strip()) for x in args.braid.split(',')]
        matches = identify_braid(word, db)
        print(f"\nBraid {word} identified as: {matches}")
