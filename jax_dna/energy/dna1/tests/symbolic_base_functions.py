"""Symbolic base functions for the DNA1 energy function.

Defined in the oxDNA paper https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
Section 2.4.1
"""

import jax_dna.energy.tests.symbolic_potentials as sp
import jax_dna.utils.types as typ


def f1(
    r: float,
    r_low: float,
    r_high: float,
    r_c_low: float,
    r_c_high: float,
    eps: float,
    a: float,
    r0: float,
    r_c: float,
    b_low: float,
    b_high: float,
) -> float:
    """This is a symbolic representation of the f1 base function.

    Equation 2.7 from the oxDNA paper.

    This function has described has 4 cases:
    1. r_low < r < r_high
    2. r_c_low < r < r_low
    3. r_high < r < r_c_high
    4. Otherwise
    """
    if r_low < r < r_high:
        return sp.v_morse(r, eps, r0, a) - sp.v_morse(r_c, eps, r0, a)
    elif r_c_low < r < r_low:
        return eps * sp.v_smooth(r, b_low, r_c_low)
    elif r_high < r < r_c_high:
        return eps * sp.v_smooth(r, b_high, r_c_high)
    else:
        return 0


def f2(
    r: float,
    r_low: float,
    r_high: float,
    r_c_low: float,
    r_c_high: float,
    k: float,
    r0: float,
    r_c: float,
    b_low: float,
    b_high: float,
) -> float:
    """This is a symbolic representation of the f1 base function.

    Equation 2.8 from the oxDNA paper.

    This function has described has 4 cases:
    1. r_low < r < r_high
    2. r_c_low < r < r_low
    3. r_high < r < r_c_high
    4. Otherwise
    """
    if r_low < r < r_high:
        return sp.v_harmonic(r, k, r0) - sp.v_harmonic(r_c, k, r0)
    elif r_c_low < r < r_low:
        return k * sp.v_smooth(r, b_low, r_c_low)
    elif r_high < r < r_c_high:
        return k * sp.v_smooth(r, b_high, r_c_high)
    else:
        return 0


def f3(
    r: float,
    r_star: float,
    r_c: float,
    eps: float,
    sigma: float,
    b: float,
) -> float:
    """The radial part of the excluded volume potential.

    This is based on equation 2.9 from the oxDNA paper.

    This function has described has 3 cases:
    1. r < r_star
    2. r_star < r < r_c
    3. Otherwise
    """

    if r < r_star:
        return sp.v_lj(r, eps, sigma)
    elif r_star < r < r_c:
        return eps * sp.v_smooth(r, b, r_c)
    else:
        return 0


def f4(
    theta: typ.ARR_OR_SCALAR,
    theta0: typ.Scalar,
    delta_theta_star: typ.Scalar,
    delta_theta_c: typ.Scalar,
    a: typ.Scalar,
    b: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """The angular modulation factor used in stacking, hydrogen-bonding, cross-stacking and coaxial stacking.

    This is based on equation 2.10 from the oxDNA paper.

    This function has 4 cases:
    1. theta0 - delta_theta_star < theta < theta0 + delta_theta_star
    2  theta0 - delta_theta_c < theta < theta0 - delta_theta_star
    3. theta0 + delta_theta_star < theta < theta0 + delta_theta_c
    4. Otherwise
    """

    if theta0 - delta_theta_star < theta < theta0 + delta_theta_star:
        return sp.v_mod(theta, a, theta0)
    elif theta0 - delta_theta_c < theta < theta0 - delta_theta_star:
        return sp.v_smooth(theta, b, theta0 - delta_theta_c)
    elif theta0 + delta_theta_star < theta < theta0 + delta_theta_c:
        return sp.v_smooth(theta, b, theta0 + delta_theta_c)
    else:
        return 0


def f5(
    x: float,
    x_star: float,
    x_c: float,
    a: float,
    b: float,
) -> float:
    """Another modulating term which is used to impose right-handedness (effectively a one-sided modulation).

    This is based on equation 2.11 from the oxDNA paper.

    This functionn has 4 cases:
    1. x > 0
    2. x_star < x < 0
    3. x_c < x < x_star
    4. Otherwise
    """
    if x > 0:
        return 1
    elif x_star < x < 0:
        return sp.v_mod(x, a, 0)
    elif x_c < x < x_star:
        return sp.v_smooth(x, b, x_c)
    else:
        return 0
