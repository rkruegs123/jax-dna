import sympy as sp


def _solve_f1_b(x: float, a: float, x0: float, xc: float) -> float:
    """Solve for the smoothing parameter b in the f1 smoothing function.

    For derivation see:
    https://github.com/rkruegs123/jax-dna/blob/master/bak/v1/src/smoothing.py
    """
    return float(
        sp.parsing.sympy_parser.parse_expr(
            "a**2*(-exp(a*(3*x0 + 2*xc)) + 2*exp(a*(x + 2*x0 + 2*xc)) - exp(a*(2*x + x0 + 2*xc)))*exp(-2*a*x)/(2*exp(a*(x + 2*xc)) + exp(a*(2*x + x0)) - 2*exp(a*(2*x + xc)) - exp(a*(x0 + 2*xc)))"
        ).evalf(subs={"x": x, "a": a, "x0": x0, "xc": xc})
    )


def _solve_f1_xc_star(x: float, a: float, x0: float, xc: float) -> float:
    """Solve for the smoothing parameter b in the f1 smoothing function.

    For derivation see:
    https://github.com/rkruegs123/jax-dna/blob/master/bak/v1/src/smoothing.py
    """
    return float(
        sp.parsing.sympy_parser.parse_expr(
            "(a*x*exp(a*(x + 2*xc)) - a*x*exp(a*(x0 + 2*xc)) + 2*exp(a*(x + 2*xc)) + exp(a*(2*x + x0)) - 2*exp(a*(2*x + xc)) - exp(a*(x0 + 2*xc)))*exp(-2*a*xc)/(a*(exp(a*x) - exp(a*x0)))"
        ).evalf(subs={"x": x, "a": a, "x0": x0, "xc": xc})
    )


def get_f1_smoothing_params(
    x0: float, a: float, xc: float, x_low: float, x_high: float
) -> tuple[float, float, float, float]:
    """Get the smoothing parameters for the f1 smoothing function."""
    solved_b_low = _solve_f1_b(x_low, a, x0, xc)
    solved_b_high = _solve_f1_b(x_high, a, x0, xc)

    solved_xc_low = _solve_f1_xc_star(x_low, a, x0, xc)
    solved_xc_high = _solve_f1_xc_star(x_high, a, x0, xc)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def _solve_f2_b(x: float, x0: float, xc: float) -> float:
    """Solve for the smoothing parameter b in the f2 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("(x - x0)**2/(2*(x - xc)*(x - 2*x0 + xc))").evalf(
            subs={"x": x, "x0": x0, "xc": xc}
        )
    )


def _solve_f2_xc_star(x: float, x0: float, xc: float) -> float:
    """Solve for the smoothing parameter xc_star in the f2 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("-(x*x0 - 2*x0*xc + xc**2)/(-x + x0)").evalf(
            subs={"x": x, "x0": x0, "xc": xc}
        )
    )


def get_f2_smoothing_params(x0: float, xc: float, x_low: float, x_high: float) -> tuple[float, float, float, float]:
    """Get the smoothing parameters for the f2 smoothing function."""
    solved_b_low = _solve_f2_b(x_low, x0, xc)
    solved_b_high = _solve_f2_b(x_high, x0, xc)

    solved_xc_low = _solve_f2_xc_star(x_low, x0, xc)
    solved_xc_high = _solve_f2_xc_star(x_high, x0, xc)

    return solved_b_low, solved_xc_low, solved_b_high, solved_xc_high


def _solve_f3_b(x: float, sigma: float) -> float:
    """Solve for the smoothing parameter b in the f3 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr(
            "-36*sigma**6*(-2*sigma**6 + x**6)**2/(x**14*(-sigma + x)*(sigma + x)*(sigma**2 - sigma*x + x**2)*(sigma**2 + sigma*x + x**2))"
        ).evalf(subs={"x": x, "sigma": sigma})
    )


def _solve_f3_xc(x: float, sigma: float) -> float:
    """Solve for the smoothing parameter xc in the f3 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("x*(-7*sigma**6 + 4*x**6)/(3*(-2*sigma**6 + x**6))").evalf(
            subs={"x": x, "sigma": sigma}
        )
    )


def get_f3_smoothing_params(r_star: float, sigma: float) -> tuple[float, float]:
    """Get the smoothing parameters for the f3 smoothing function."""
    solved_b = _solve_f3_b(r_star, sigma)
    solved_xc = _solve_f3_xc(r_star, sigma)

    return solved_b, solved_xc


def _solve_f4_b(
    x: float,
    x0: float,
    a: float,
) -> float:
    """Solve for the smoothing parameter b in the f4 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("-a**2*(-x + x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)").evalf(
            subs={"x": x, "x0": x0, "a": a}
        )
    )


def _solve_f4_xc(x: float, x0: float, a: float) -> float:
    """Solve for the smoothing parameter xc in the f4 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("(a*x*x0 - a*x0**2 + 1)/(a*(x - x0))").evalf(subs={"x": x, "x0": x0, "a": a})
    )


def get_f4_smoothing_params(
    a: float,
    x0: float,
    delta_x_star: float,
) -> tuple[float, float, float, float]:
    """Get the smoothing parameters for the f4 smoothing function."""
    solved_b_plus = _solve_f4_b(x0 + delta_x_star, x0, a)

    solved_xc_plus = _solve_f4_xc(x0 + delta_x_star, x0, a)
    solved_delta_xc_plus = solved_xc_plus - x0

    return solved_b_plus, solved_delta_xc_plus


def _solve_f5_b(
    x: float,
    x0: float,
    a: float,
) -> float:
    """Solve for the smoothing parameter b in the f5 smoothing function."""
    return float(
        sp.parsing.sympy_parser.parse_expr("-a**2*(-x + x0)**2/(a*x**2 - 2*a*x*x0 + a*x0**2 - 1)").evalf(
            subs={"x": x, "x0": x0, "a": a}
        )
    )
