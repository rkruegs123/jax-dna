
import sympy as sp

def _solve_f1_b(
    x: float,
    a: float,
    x0: float,
    xc: float
) -> float:
    """Solve for the smoothing parameter b in the f1 smoothing function.

    For derivation see:
    https://github.com/rkruegs123/jax-dna/blob/master/bak/v1/src/smoothing.py
    """
    return float(
        sp.parsing.sympy_parser.parse_expr(
            "a**2*(-exp(a*(3*x0 + 2*xc)) + 2*exp(a*(x + 2*x0 + 2*xc)) - exp(a*(2*x + x0 + 2*xc)))*exp(-2*a*x)/(2*exp(a*(x + 2*xc)) + exp(a*(2*x + x0)) - 2*exp(a*(2*x + xc)) - exp(a*(x0 + 2*xc)))"
        ).evalf(
            subs={"x": x, "a": a, "x0": x0, "xc": xc}
        )
    )