"""
LaTeX formatting for braids and polynomials.
"""

from typing import Optional
from collections import defaultdict

try:
    import sympy
    from sympy import symbols, latex, Rational

    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False


def braid_to_latex(word: list[int], compact: bool = False) -> str:
    """
    Convert a braid word to LaTeX notation.

    Args:
        word: Braid word as list of generators
        compact: If True, use exponent notation for repeated generators

    Returns:
        LaTeX string

    Example:
        >>> braid_to_latex([1, 1, 1])
        '\\sigma_1 \\sigma_1 \\sigma_1'
        >>> braid_to_latex([1, 1, 1], compact=True)
        '\\sigma_1^{3}'
        >>> braid_to_latex([1, -2, 1])
        '\\sigma_1 \\sigma_2^{-1} \\sigma_1'
    """
    if not word:
        return "1"

    if compact:
        # Group consecutive identical generators
        groups = []
        current_gen = word[0]
        current_count = 1

        for g in word[1:]:
            if g == current_gen:
                current_count += 1
            else:
                groups.append((current_gen, current_count))
                current_gen = g
                current_count = 1
        groups.append((current_gen, current_count))

        parts = []
        for gen, count in groups:
            if gen > 0:
                if count == 1:
                    parts.append(f"\\sigma_{gen}")
                else:
                    parts.append(f"\\sigma_{gen}^{{{count}}}")
            else:
                if count == 1:
                    parts.append(f"\\sigma_{-gen}^{{-1}}")
                else:
                    parts.append(f"\\sigma_{-gen}^{{{-count}}}")
        return " ".join(parts)

    else:
        parts = []
        for g in word:
            if g > 0:
                parts.append(f"\\sigma_{g}")
            else:
                parts.append(f"\\sigma_{-g}^{{-1}}")
        return " ".join(parts)


def jones_to_latex(
    jones: list[tuple[int, int]],
    variable: str = "t",
    use_sympy: bool = True,
) -> str:
    """
    Convert Jones polynomial to LaTeX.

    Args:
        jones: Jones polynomial as [(coefficient, exponent), ...]
        variable: Variable name ('t', 'q', or 'x')
        use_sympy: Use sympy for simplification (if available)

    Returns:
        LaTeX string

    Note:
        tl-tensor outputs in x-variable where t = x^4.
        If variable='t', exponents are divided by 4.
        If variable='q', exponents are divided by 2 (q = x^2).

    Example:
        >>> jones_to_latex([(-1, -16), (1, -12), (1, -4)], 't')
        '-t^{-4} + t^{-3} + t^{-1}'
    """
    if not jones:
        return "0"

    # Determine exponent divisor
    if variable == "t":
        divisor = 4
    elif variable == "q":
        divisor = 2
    else:  # x
        divisor = 1

    if use_sympy and HAS_SYMPY:
        var = symbols(variable)
        expr = sum(coeff * var ** (exp // divisor) for coeff, exp in jones)
        return latex(expr)
    else:
        # Manual formatting
        terms = []
        sorted_jones = sorted(jones, key=lambda x: -x[1])  # Descending exponent

        for i, (coeff, exp) in enumerate(sorted_jones):
            exp = exp // divisor

            if coeff == 0:
                continue

            # Sign
            if coeff > 0 and i > 0:
                sign = " + "
            elif coeff < 0:
                sign = " - " if i > 0 else "-"
            else:
                sign = "" if i == 0 else " + "

            abs_coeff = abs(coeff)

            # Coefficient
            if abs_coeff == 1 and exp != 0:
                coeff_str = ""
            else:
                coeff_str = str(abs_coeff)

            # Exponent
            if exp == 0:
                exp_str = "" if coeff_str else "1"
            elif exp == 1:
                exp_str = variable
            else:
                exp_str = f"{variable}^{{{exp}}}"

            terms.append(f"{sign}{coeff_str}{exp_str}")

        return "".join(terms) if terms else "0"


def jones_to_sympy(jones: list[tuple[int, int]], variable: str = "t"):
    """
    Convert Jones polynomial to sympy expression.

    Args:
        jones: Jones polynomial as [(coefficient, exponent), ...]
        variable: Variable name

    Returns:
        sympy expression

    Example:
        >>> poly = jones_to_sympy([(-1, -16), (1, -12), (1, -4)], 't')
        >>> poly.expand()
        -1/t**4 + 1/t**3 + 1/t
    """
    if not HAS_SYMPY:
        raise ImportError("sympy required for jones_to_sympy")

    # Determine exponent divisor
    if variable == "t":
        divisor = 4
    elif variable == "q":
        divisor = 2
    else:  # x
        divisor = 1

    var = symbols(variable)
    return sum(coeff * var ** (exp // divisor) for coeff, exp in jones)
