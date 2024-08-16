import sympy as sp

# functional forms from oxDNA paper
# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1


def v_fene(r: float, eps: float, r0: float, delt: float) -> float:
    """String form of Equation 2.1 from the oxDNA paper."""
    return float(
        sp.parsing.sympy_parser.parse_expr("-eps / 2 * log(1 - (r-r0)**2 / delt**2)").evalf(
            subs={"r": r, "eps": eps, "r0": r0, "delt": delt}
        )
    )


def v_morse(r: float, eps: float, r0: float, a: float) -> float:
    """String form of Equation 2.2 from the oxDNA paper."""
    return float(
        sp.parsing.sympy_parser.parse_expr("eps * (1 - exp(-(r-r0)*a))**2").evalf(
            subs={"r": r, "eps": eps, "r0": r0, "a": a}
        )
    )


def v_harmonic(r: float, k: float, r0: float) -> float:
    """String form of Equation 2.3 from the oxDNA paper."""
    return float(sp.parsing.sympy_parser.parse_expr("k / 2 * (r - r0)**2").evalf(subs={"r": r, "k": k, "r0": r0}))


def v_lj(r: float, eps: float, sigma: float) -> float:
    """String form of the Lennard-Jones potential."""
    return float(
        sp.parsing.sympy_parser.parse_expr("4 * eps * ((sigma / r)**12 - (sigma / r)**6)").evalf(
            subs={"r": r, "eps": eps, "sigma": sigma}
        )
    )


def v_mod(theta: float, a: float, theta0: float) -> float:
    """String form of the modified potential."""
    return float(
        sp.parsing.sympy_parser.parse_expr("1 - a * (theta - theta0)**2").evalf(
            subs={"theta": theta, "a": a, "theta0": theta0}
        )
    )


def v_smooth(x: float, b: float, x_c: float) -> float:
    """String form of the smooth potential."""
    return float(sp.parsing.sympy_parser.parse_expr("b * (x_c - x)**2").evalf(subs={"x": x, "b": b, "x_c": x_c}))
