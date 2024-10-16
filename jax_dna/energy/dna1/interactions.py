"""DNA1 interactions.

These functions are based on the oxDNA1 model paper found here:
https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
"""

import jax.numpy as jnp
import jax.tree_util as tu

import jax_dna.energy.dna1.base_functions as jd_base_functions
import jax_dna.energy.potentials as jd_potentials
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as typ


def v_fene_smooth(
    r: typ.ARR_OR_SCALAR,
    eps_backbone: typ.Scalar,
    r0_backbone: typ.Scalar,
    delta_backbone: typ.Scalar,
    fmax: typ.Scalar = 500,
    finf: typ.Scalar = 4.0,
) -> typ.ARR_OR_SCALAR:
    """Smoothed version of the FENE potential."""
    eps = eps_backbone
    r0 = r0_backbone
    delt = delta_backbone

    diff = jd_math.smooth_abs(r - r0)

    delt2 = delt**2
    eps2 = eps**2
    fmax2 = fmax**2
    xmax = (-eps + jnp.sqrt(eps2 + 4 * fmax2 * delt2)) / (2 * fmax)

    # precompute terms for smoothed case
    fene_xmax = -(eps / 2.0) * jnp.log(1.0 - xmax**2 / delt2)
    long_xmax = (fmax - finf) * xmax * jnp.log(xmax) + finf * xmax
    smoothed_energy = (fmax - finf) * xmax * jnp.log(diff) + finf * diff - long_xmax + fene_xmax

    return jnp.where(diff > xmax, smoothed_energy, jd_potentials.v_fene(r, eps, r0, delt))


def exc_vol_bonded(
    dr_base: typ.ARR_OR_SCALAR,
    dr_back_base: typ.ARR_OR_SCALAR,
    dr_base_back: typ.ARR_OR_SCALAR,
    eps_exc: typ.Scalar,
    # reference to f3(dr_base)
    dr_star_base: typ.Scalar,
    sigma_base: typ.Scalar,
    b_base: typ.Scalar,
    dr_c_base: typ.Scalar,
    # reference to f3(dr_back_base)
    dr_star_back_base: typ.Scalar,
    sigma_back_base: typ.Scalar,
    b_back_base: typ.Scalar,
    dr_c_back_base: typ.Scalar,
    # reference to f3(dr_base_back)
    dr_star_base_back: typ.Scalar,
    sigma_base_back: typ.Scalar,
    b_base_back: typ.Scalar,
    dr_c_base_back: typ.Scalar,
) -> typ.Scalar:
    """Excluded volume energy for bonded interactions."""
    # Note: r_c must be greater than r*
    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    f3_base_exc_vol = jd_base_functions.f3(
        r_base, r_star=dr_star_base, r_c=dr_c_base, eps=eps_exc, sigma=sigma_base, b=b_base
    )

    f3_back_base_exc_vol = jd_base_functions.f3(
        r_back_base, r_star=dr_star_back_base, r_c=dr_c_back_base, eps=eps_exc, sigma=sigma_back_base, b=b_back_base
    )

    f3_base_back_exc_vol = jd_base_functions.f3(
        r_base_back, r_star=dr_star_base_back, r_c=dr_c_base_back, eps=eps_exc, sigma=sigma_base_back, b=b_base_back
    )

    return f3_base_exc_vol + f3_back_base_exc_vol + f3_base_back_exc_vol


def exc_vol_unbonded(
    dr_base: typ.ARR_OR_SCALAR,
    dr_backbone: typ.ARR_OR_SCALAR,
    dr_back_base: typ.ARR_OR_SCALAR,
    dr_base_back: typ.ARR_OR_SCALAR,
    eps_exc: typ.Scalar,
    # reference to f3(dr_base)
    dr_star_base: typ.Scalar,
    sigma_base: typ.Scalar,
    b_base: typ.Scalar,
    dr_c_base: typ.Scalar,
    # reference to f3(dr_back_base)
    dr_star_back_base: typ.Scalar,
    sigma_back_base: typ.Scalar,
    b_back_base: typ.Scalar,
    dr_c_back_base: typ.Scalar,
    # reference to f3(dr_base_back)
    dr_star_base_back: typ.Scalar,
    sigma_base_back: typ.Scalar,
    b_base_back: typ.Scalar,
    dr_c_base_back: typ.Scalar,
    # reference to f3(backbone)
    dr_star_backbone: typ.Scalar,
    sigma_backbone: typ.Scalar,
    b_backbone: typ.Scalar,
    dr_c_backbone: typ.Scalar,
) -> typ.Scalar:
    """Excluded volume energy for unbonded interactions."""
    r_back = jnp.linalg.norm(dr_backbone, axis=1)
    f3_back_exc_vol = jd_base_functions.f3(
        r_back, r_star=dr_star_backbone, r_c=dr_c_backbone, eps=eps_exc, sigma=sigma_backbone, b=b_backbone
    )
    return f3_back_exc_vol + exc_vol_bonded(
        dr_base,
        dr_back_base,
        dr_base_back,
        eps_exc,
        dr_star_base,
        sigma_base,
        b_base,
        dr_c_base,
        dr_star_back_base,
        sigma_back_base,
        b_back_base,
        dr_c_back_base,
        dr_star_base_back,
        sigma_base_back,
        b_base_back,
        dr_c_base_back,
    )


# comments from original code
# Note that we use r_stack instead of dr_stack
# TODO(rkruegs123): fix this one with bs and rcs
# https://github.com/ssec-jhu/jax-dna/issues/7
def stacking(
    # obervables
    r_stack: typ.ARR_OR_SCALAR,
    theta4: typ.ARR_OR_SCALAR,
    theta5: typ.ARR_OR_SCALAR,
    theta6: typ.ARR_OR_SCALAR,
    cosphi1: typ.ARR_OR_SCALAR,
    cosphi2: typ.ARR_OR_SCALAR,
    # params
    dr_low_stack: typ.Scalar,
    dr_high_stack: typ.Scalar,
    eps_stack: typ.Scalar,
    a_stack: typ.Scalar,
    dr0_stack: typ.Scalar,
    dr_c_stack: typ.Scalar,
    dr_c_low_stack: typ.Scalar,
    dr_c_high_stack: typ.Scalar,
    b_low_stack: typ.Scalar,
    b_high_stack: typ.Scalar,
    theta0_stack_4: typ.Scalar,
    delta_theta_star_stack_4: typ.Scalar,
    a_stack_4: typ.Scalar,
    delta_theta_stack_4_c: typ.Scalar,
    b_stack_4: typ.Scalar,
    theta0_stack_5: typ.Scalar,
    delta_theta_star_stack_5: typ.Scalar,
    a_stack_5: typ.Scalar,
    delta_theta_stack_5_c: typ.Scalar,
    b_stack_5: typ.Scalar,
    theta0_stack_6: typ.Scalar,
    delta_theta_star_stack_6: typ.Scalar,
    a_stack_6: typ.Scalar,
    delta_theta_stack_6_c: typ.Scalar,
    b_stack_6: typ.Scalar,
    neg_cos_phi1_star_stack: typ.Scalar,
    a_stack_1: typ.Scalar,
    neg_cos_phi1_c_stack: typ.Scalar,
    b_neg_cos_phi1_stack: typ.Scalar,
    neg_cos_phi2_star_stack: typ.Scalar,
    a_stack_2: typ.Scalar,
    neg_cos_phi2_c_stack: typ.Scalar,
    b_neg_cos_phi2_stack: typ.Scalar,
) -> typ.Scalar:
    """Stacking energy."""
    f1_dr_stack = jd_base_functions.f1(
        r_stack,
        r_low=dr_low_stack,
        r_high=dr_high_stack,
        r_c_low=dr_c_low_stack,
        r_c_high=dr_c_high_stack,
        eps=eps_stack,
        a=a_stack,
        r0=dr0_stack,
        r_c=dr_c_stack,
        b_low=b_low_stack,
        b_high=b_high_stack,
    )

    f4_theta_4_stack = jd_base_functions.f4(
        theta4,
        theta0=theta0_stack_4,
        delta_theta_star=delta_theta_star_stack_4,
        delta_theta_c=delta_theta_stack_4_c,
        a=a_stack_4,
        b=b_stack_4,
    )

    f4_theta_5p_stack = jd_base_functions.f4(
        theta5,
        theta0=theta0_stack_5,
        delta_theta_star=delta_theta_star_stack_5,
        delta_theta_c=delta_theta_stack_5_c,
        a=a_stack_5,
        b=b_stack_5,
    )

    f4_theta_6p_stack = jd_base_functions.f4(
        theta6,
        theta0=theta0_stack_6,
        delta_theta_star=delta_theta_star_stack_6,
        delta_theta_c=delta_theta_stack_6_c,
        a=a_stack_6,
        b=b_stack_6,
    )

    f5_neg_cosphi1_stack = jd_base_functions.f5(
        -cosphi1,
        x_star=neg_cos_phi1_star_stack,
        x_c=neg_cos_phi1_c_stack,
        a=a_stack_1,
        b=b_neg_cos_phi1_stack,
    )

    f5_neg_cosphi2_stack = jd_base_functions.f5(
        -cosphi2,
        x_star=neg_cos_phi2_star_stack,
        x_c=neg_cos_phi2_c_stack,
        a=a_stack_2,
        b=b_neg_cos_phi2_stack,
    )

    return (
        f1_dr_stack
        * f4_theta_4_stack
        * f4_theta_5p_stack
        * f4_theta_6p_stack
        * f5_neg_cosphi1_stack
        * f5_neg_cosphi2_stack
    )


def cross_stacking(
    # observables
    r_hb: typ.ARR_OR_SCALAR,
    theta1: typ.ARR_OR_SCALAR,
    theta2: typ.ARR_OR_SCALAR,
    theta3: typ.ARR_OR_SCALAR,
    theta4: typ.ARR_OR_SCALAR,
    theta7: typ.ARR_OR_SCALAR,
    theta8: typ.ARR_OR_SCALAR,
    # reference to f2_dr_cross
    dr_low_cross: typ.Scalar,
    dr_high_cross: typ.Scalar,
    dr_c_low_cross: typ.Scalar,
    dr_c_high_cross: typ.Scalar,
    k_cross: typ.Scalar,
    r0_cross: typ.Scalar,
    dr_c_cross: typ.Scalar,
    b_low_cross: typ.Scalar,
    b_high_cross: typ.Scalar,
    # reference to f4(theta1)
    theta0_cross_1: typ.Scalar,
    delta_theta_star_cross_1: typ.Scalar,
    delta_theta_cross_1_c: typ.Scalar,
    a_cross_1: typ.Scalar,
    b_cross_1: typ.Scalar,
    # reference to f4(theta2)
    theta0_cross_2: typ.Scalar,
    delta_theta_star_cross_2: typ.Scalar,
    delta_theta_cross_2_c: typ.Scalar,
    a_cross_2: typ.Scalar,
    b_cross_2: typ.Scalar,
    # reference to f4(theta3)
    theta0_cross_3: typ.Scalar,
    delta_theta_star_cross_3: typ.Scalar,
    delta_theta_cross_3_c: typ.Scalar,
    a_cross_3: typ.Scalar,
    b_cross_3: typ.Scalar,
    # reference to f4(theta4)
    theta0_cross_4: typ.Scalar,
    delta_theta_star_cross_4: typ.Scalar,
    delta_theta_cross_4_c: typ.Scalar,
    a_cross_4: typ.Scalar,
    b_cross_4: typ.Scalar,
    # reference to f7(theta7)
    theta0_cross_7: typ.Scalar,
    delta_theta_star_cross_7: typ.Scalar,
    delta_theta_cross_7_c: typ.Scalar,
    a_cross_7: typ.Scalar,
    b_cross_7: typ.Scalar,
    # reference to f8(theta8)
    theta0_cross_8: typ.Scalar,
    delta_theta_star_cross_8: typ.Scalar,
    delta_theta_cross_8_c: typ.Scalar,
    a_cross_8: typ.Scalar,
    b_cross_8: typ.Scalar,
) -> typ.Scalar:
    """Cross-stacking energy."""
    f2_dr_cross = jd_base_functions.f2(
        r_hb,
        r_low=dr_low_cross,
        r_high=dr_high_cross,
        r_c_low=dr_c_low_cross,
        r_c_high=dr_c_high_cross,
        k=k_cross,
        r0=r0_cross,
        r_c=dr_c_cross,
        b_low=b_low_cross,
        b_high=b_high_cross,
    )

    f4_theta_1_cross = jd_base_functions.f4(
        theta1,
        theta0=theta0_cross_1,
        delta_theta_star=delta_theta_star_cross_1,
        delta_theta_c=delta_theta_cross_1_c,
        a=a_cross_1,
        b=b_cross_1,
    )

    f4_theta_2_cross = jd_base_functions.f4(
        theta2,
        theta0=theta0_cross_2,
        delta_theta_star=delta_theta_star_cross_2,
        delta_theta_c=delta_theta_cross_2_c,
        a=a_cross_2,
        b=b_cross_2,
    )

    f4_theta_3_cross = jd_base_functions.f4(
        theta3,
        theta0=theta0_cross_3,
        delta_theta_star=delta_theta_star_cross_3,
        delta_theta_c=delta_theta_cross_3_c,
        a=a_cross_3,
        b=b_cross_3,
    )

    f4_theta_4_cross_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_cross_4,
        delta_theta_star=delta_theta_star_cross_4,
        delta_theta_c=delta_theta_cross_4_c,
        a=a_cross_4,
        b=b_cross_4,
    )

    f4_theta_7_cross_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_cross_7,
        delta_theta_star=delta_theta_star_cross_7,
        delta_theta_c=delta_theta_cross_7_c,
        a=a_cross_7,
        b=b_cross_7,
    )

    f4_theta_8_cross_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_cross_8,
        delta_theta_star=delta_theta_star_cross_8,
        delta_theta_c=delta_theta_cross_8_c,
        a=a_cross_8,
        b=b_cross_8,
    )

    return (
        f2_dr_cross
        * f4_theta_1_cross
        * f4_theta_2_cross
        * f4_theta_3_cross
        * (f4_theta_4_cross_fn(theta4) + f4_theta_4_cross_fn(jnp.pi - theta4))
        * (f4_theta_7_cross_fn(theta7) + f4_theta_7_cross_fn(jnp.pi - theta7))
        * (f4_theta_8_cross_fn(theta8) + f4_theta_8_cross_fn(jnp.pi - theta8))
    )


def coaxial_stacking(
    # obersvables
    dr_stack: typ.ARR_OR_SCALAR,
    theta4: typ.ARR_OR_SCALAR,
    theta1: typ.ARR_OR_SCALAR,
    theta5: typ.ARR_OR_SCALAR,
    theta6: typ.ARR_OR_SCALAR,
    cosphi3: typ.ARR_OR_SCALAR,
    cosphi4: typ.ARR_OR_SCALAR,
    # reference to f2(dr_stack)
    dr_low_coax: typ.Scalar,
    dr_high_coax: typ.Scalar,
    dr_c_low_coax: typ.Scalar,
    dr_c_high_coax: typ.Scalar,
    k_coax: typ.Scalar,
    dr0_coax: typ.Scalar,
    dr_c_coax: typ.Scalar,
    b_low_coax: typ.Scalar,
    b_high_coax: typ.Scalar,
    # reference to f4(theta4)
    theta0_coax_4: typ.Scalar,
    delta_theta_star_coax_4: typ.Scalar,
    delta_theta_coax_4_c: typ.Scalar,
    a_coax_4: typ.Scalar,
    b_coax_4: typ.Scalar,
    # reference to f4(theta1)
    theta0_coax_1: typ.Scalar,
    delta_theta_star_coax_1: typ.Scalar,
    delta_theta_coax_1_c: typ.Scalar,
    a_coax_1: typ.Scalar,
    b_coax_1: typ.Scalar,
    # reference to f4(theta5)
    theta0_coax_5: typ.Scalar,
    delta_theta_star_coax_5: typ.Scalar,
    delta_theta_coax_5_c: typ.Scalar,
    a_coax_5: typ.Scalar,
    b_coax_5: typ.Scalar,
    # reference to f4(theta6)
    theta0_coax_6: typ.Scalar,
    delta_theta_star_coax_6: typ.Scalar,
    delta_theta_coax_6_c: typ.Scalar,
    a_coax_6: typ.Scalar,
    b_coax_6: typ.Scalar,
    # reference to f5(cosphi3)
    cos_phi3_star_coax: typ.Scalar,
    cos_phi3_c_coax: typ.Scalar,
    a_coax_3p: typ.Scalar,
    b_cos_phi3_coax: typ.Scalar,
    # reference to f5(cosphi4)
    cos_phi4_star_coax: typ.Scalar,
    cos_phi4_c_coax: typ.Scalar,
    a_coax_4p: typ.Scalar,
    b_cos_phi4_coax: typ.Scalar,
) -> typ.Scalar:
    """Coaxial stacking energy."""
    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    f2_dr_coax = jd_base_functions.f2(
        r_stack,
        r_low=dr_low_coax,
        r_high=dr_high_coax,
        r_c_low=dr_c_low_coax,
        r_c_high=dr_c_high_coax,
        k=k_coax,
        r0=dr0_coax,
        r_c=dr_c_coax,
        b_low=b_low_coax,
        b_high=b_high_coax,
    )

    f4_theta_4_coax = jd_base_functions.f4(
        theta4,
        theta0=theta0_coax_4,
        delta_theta_star=delta_theta_star_coax_4,
        delta_theta_c=delta_theta_coax_4_c,
        a=a_coax_4,
        b=b_coax_4,
    )

    f4_theta_1_coax_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_coax_1,
        delta_theta_star=delta_theta_star_coax_1,
        delta_theta_c=delta_theta_coax_1_c,
        a=a_coax_1,
        b=b_coax_1,
    )

    f4_theta_5_coax_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_coax_5,
        delta_theta_star=delta_theta_star_coax_5,
        delta_theta_c=delta_theta_coax_5_c,
        a=a_coax_5,
        b=b_coax_5,
    )

    f4_theta_6_coax_fn = tu.Partial(
        jd_base_functions.f4,
        theta0=theta0_coax_6,
        delta_theta_star=delta_theta_star_coax_6,
        delta_theta_c=delta_theta_coax_6_c,
        a=a_coax_6,
        b=b_coax_6,
    )

    f5_cosphi3_coax = jd_base_functions.f5(
        cosphi3, x_star=cos_phi3_star_coax, x_c=cos_phi3_c_coax, a=a_coax_3p, b=b_cos_phi3_coax
    )

    f5_cosphi4_coax = jd_base_functions.f5(
        cosphi4, x_star=cos_phi4_star_coax, x_c=cos_phi4_c_coax, a=a_coax_4p, b=b_cos_phi4_coax
    )

    return (
        f2_dr_coax
        * f4_theta_4_coax
        * (f4_theta_1_coax_fn(theta1) + f4_theta_1_coax_fn(2 * jnp.pi - theta1))
        * (f4_theta_5_coax_fn(theta5) + f4_theta_5_coax_fn(jnp.pi - theta5))
        * (f4_theta_6_coax_fn(theta6) + f4_theta_6_coax_fn(jnp.pi - theta6))
        * f5_cosphi3_coax
        * f5_cosphi4_coax
    )


def hydrogen_bonding(
    # observables
    dr_hb: typ.ARR_OR_SCALAR,
    theta1: typ.ARR_OR_SCALAR,
    theta2: typ.ARR_OR_SCALAR,
    theta3: typ.ARR_OR_SCALAR,
    theta4: typ.ARR_OR_SCALAR,
    theta7: typ.ARR_OR_SCALAR,
    theta8: typ.ARR_OR_SCALAR,
    # reference to f1_dr_hb
    dr_low_hb: typ.Scalar,
    dr_high_hb: typ.Scalar,
    dr_c_low_hb: typ.Scalar,
    dr_c_high_hb: typ.Scalar,
    eps_hb: typ.Scalar,
    a_hb: typ.Scalar,
    dr0_hb: typ.Scalar,
    dr_c_hb: typ.Scalar,
    b_low_hb: typ.Scalar,
    b_high_hb: typ.Scalar,
    # reference to f4_theta_1_hb
    theta0_hb_1: typ.Scalar,
    delta_theta_star_hb_1: typ.Scalar,
    a_hb_1: typ.Scalar,
    delta_theta_hb_1_c: typ.Scalar,
    b_hb_1: typ.Scalar,
    # reference to f4_theta_2_hb
    theta0_hb_2: typ.Scalar,
    delta_theta_star_hb_2: typ.Scalar,
    a_hb_2: typ.Scalar,
    delta_theta_hb_2_c: typ.Scalar,
    b_hb_2: typ.Scalar,
    # reference to f4_theta_3_hb
    theta0_hb_3: typ.Scalar,
    delta_theta_star_hb_3: typ.Scalar,
    a_hb_3: typ.Scalar,
    delta_theta_hb_3_c: typ.Scalar,
    b_hb_3: typ.Scalar,
    # reference to f4_theta_4_hb
    theta0_hb_4: typ.Scalar,
    delta_theta_star_hb_4: typ.Scalar,
    a_hb_4: typ.Scalar,
    delta_theta_hb_4_c: typ.Scalar,
    b_hb_4: typ.Scalar,
    # reference to f4_theta_7_hb
    theta0_hb_7: typ.Scalar,
    delta_theta_star_hb_7: typ.Scalar,
    a_hb_7: typ.Scalar,
    delta_theta_hb_7_c: typ.Scalar,
    b_hb_7: typ.Scalar,
    # reference to f4_theta_8_hb
    theta0_hb_8: typ.Scalar,
    delta_theta_star_hb_8: typ.Scalar,
    a_hb_8: typ.Scalar,
    delta_theta_hb_8_c: typ.Scalar,
    b_hb_8: typ.Scalar,
) -> typ.Scalar:
    """Hydrogen bonding energy."""
    r_hb = jnp.linalg.norm(dr_hb, axis=1)
    f1_dr_hb = jd_base_functions.f1(
        r_hb,
        r_low=dr_low_hb,
        r_high=dr_high_hb,
        r_c_low=dr_c_low_hb,
        r_c_high=dr_c_high_hb,
        eps=eps_hb,
        a=a_hb,
        r0=dr0_hb,
        r_c=dr_c_hb,
        b_low=b_low_hb,
        b_high=b_high_hb,
    )

    f4_theta_1_hb = jd_base_functions.f4(
        theta1,
        theta0=theta0_hb_1,
        delta_theta_star=delta_theta_star_hb_1,
        delta_theta_c=delta_theta_hb_1_c,
        a=a_hb_1,
        b=b_hb_1,
    )

    f4_theta_2_hb = jd_base_functions.f4(
        theta2,
        theta0=theta0_hb_2,
        delta_theta_star=delta_theta_star_hb_2,
        delta_theta_c=delta_theta_hb_2_c,
        a=a_hb_2,
        b=b_hb_2,
    )

    f4_theta_3_hb = jd_base_functions.f4(
        theta3,
        theta0=theta0_hb_3,
        delta_theta_star=delta_theta_star_hb_3,
        delta_theta_c=delta_theta_hb_3_c,
        a=a_hb_3,
        b=b_hb_3,
    )

    f4_theta_4_hb = jd_base_functions.f4(
        theta4,
        theta0=theta0_hb_4,
        delta_theta_star=delta_theta_star_hb_4,
        delta_theta_c=delta_theta_hb_4_c,
        a=a_hb_4,
        b=b_hb_4,
    )

    f4_theta_7_hb = jd_base_functions.f4(
        theta7,
        theta0=theta0_hb_7,
        delta_theta_star=delta_theta_star_hb_7,
        delta_theta_c=delta_theta_hb_7_c,
        a=a_hb_7,
        b=b_hb_7,
    )

    f4_theta_8_hb = jd_base_functions.f4(
        theta8,
        theta0=theta0_hb_8,
        delta_theta_star=delta_theta_star_hb_8,
        delta_theta_c=delta_theta_hb_8_c,
        a=a_hb_8,
        b=b_hb_8,
    )

    return f1_dr_hb * f4_theta_1_hb * f4_theta_2_hb * f4_theta_3_hb * f4_theta_4_hb * f4_theta_7_hb * f4_theta_8_hb
