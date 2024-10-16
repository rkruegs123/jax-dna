"""Sympy math helpers for testing"""

import sympy as sp


def smooth_abs(x: float, eps: float = 1e-10) -> float:
    """A smooth absolute value function.

    Note that a non-zero eps gives continuous first dervatives.

    https://math.stackexchange.com/questions/1172472/differentiable-approximation-of-the-absolute-value-function
    """
    return float(sp.parsing.sympy_parser.parse_expr("sqrt(x**2 + eps)").evalf(subs={"x": x, "eps": eps}))


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a value between two values."""
    return float(
        sp.parsing.sympy_parser.parse_expr("Piecewise((lo, x <= lo), (hi, x >= hi), (x, True))").evalf(
            subs={"x": x, "lo": lo, "hi": hi}
        )
    )
