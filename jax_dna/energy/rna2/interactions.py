"""RNA2 interactions.

These functions are based on the RNA2 model paper found here:
https://arxiv.org/abs/1403.4180
"""

import jax.numpy as jnp
import jax.tree_util as tu

import jax_dna.energy.dna1.base_functions as jd_base_functions
import jax_dna.utils.types as typ


def stacking(
    # obervables
    r_stack: typ.ARR_OR_SCALAR,
    theta5: typ.ARR_OR_SCALAR,
    theta6: typ.ARR_OR_SCALAR,
    theta9: typ.ARR_OR_SCALAR,
    theta10: typ.ARR_OR_SCALAR,
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
    theta0_stack_9: typ.Scalar,
    delta_theta_star_stack_9: typ.Scalar,
    a_stack_9: typ.Scalar,
    delta_theta_stack_9_c: typ.Scalar,
    b_stack_9: typ.Scalar,
    theta0_stack_10: typ.Scalar,
    delta_theta_star_stack_10: typ.Scalar,
    a_stack_10: typ.Scalar,
    delta_theta_stack_10_c: typ.Scalar,
    b_stack_10: typ.Scalar,
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

    f4_theta_9_stack = jd_base_functions.f4(
        theta9,
        theta0=theta0_stack_9,
        delta_theta_star=delta_theta_star_stack_9,
        delta_theta_c=delta_theta_stack_9_c,
        a=a_stack_9,
        b=b_stack_9,
    )

    f4_theta_10_stack = jd_base_functions.f4(
        theta10,
        theta0=theta0_stack_10,
        delta_theta_star=delta_theta_star_stack_10,
        delta_theta_c=delta_theta_stack_10_c,
        a=a_stack_10,
        b=b_stack_10,
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
        * f4_theta_5p_stack
        * f4_theta_6p_stack
        * f4_theta_9_stack
        * f4_theta_10_stack
        * f5_neg_cosphi1_stack
        * f5_neg_cosphi2_stack
    )


def cross_stacking(
    # observables
    r_hb: typ.ARR_OR_SCALAR,
    theta1: typ.ARR_OR_SCALAR,
    theta2: typ.ARR_OR_SCALAR,
    theta3: typ.ARR_OR_SCALAR,
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
        * (f4_theta_7_cross_fn(theta7) + f4_theta_7_cross_fn(jnp.pi - theta7))
        * (f4_theta_8_cross_fn(theta8) + f4_theta_8_cross_fn(jnp.pi - theta8))
    )
