"""DNA2 interactions.

These functions are based on the oxDNA2 model paper found here:
https://arxiv.org/abs/1504.00821
"""

import jax.numpy as jnp
import jax.tree_util as tu

import jax_dna.energy.dna1.base_functions as jd_base_functions1
import jax_dna.energy.dna2.base_functions as jd_base_functions2
import jax_dna.utils.types as typ


def debye(
    r: typ.ARR_OR_SCALAR,
    kappa: typ.Scalar,
    prefactor: typ.Scalar,
    smoothing_coeff: typ.Scalar,
    r_cut: typ.Scalar,
    r_high: typ.Scalar,
) -> typ.ARR_OR_SCALAR:
    """Debye-huckel potential."""
    energy_full = jnp.exp(r * -kappa) * (prefactor / r)
    energy_smooth =  smoothing_coeff * (r - r_cut)**2
    cond = r < r_high
    energy = jnp.where(cond, energy_full, energy_smooth)
    return jnp.where(r < r_cut, energy, 0.0)



def coaxial_stacking(
    # obersvables
    dr_stack: typ.ARR_OR_SCALAR,
    theta4: typ.ARR_OR_SCALAR,
    theta1: typ.ARR_OR_SCALAR,
    theta5: typ.ARR_OR_SCALAR,
    theta6: typ.ARR_OR_SCALAR,
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
    # reference to f6(theta1)
    a_coax_1_f6: typ.Scalar,
    b_coax_1_f6: typ.Scalar,
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
) -> typ.Scalar:
    """Coaxial stacking energy."""
    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    f2_dr_coax = jd_base_functions1.f2(
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

    f4_theta_4_coax = jd_base_functions1.f4(
        theta4,
        theta0=theta0_coax_4,
        delta_theta_star=delta_theta_star_coax_4,
        delta_theta_c=delta_theta_coax_4_c,
        a=a_coax_4,
        b=b_coax_4,
    )

    f4_theta_1_coax = jd_base_functions1.f4(
        theta1,
        theta0=theta0_coax_1,
        delta_theta_star=delta_theta_star_coax_1,
        delta_theta_c=delta_theta_coax_1_c,
        a=a_coax_1,
        b=b_coax_1,
    )

    f6_theta_1_coax = jd_base_functions2.f6(
        theta1,
        a=a_coax_1_f6,
        b=b_coax_1_f6
    )

    f4_theta_5_coax_fn = tu.Partial(
        jd_base_functions1.f4,
        theta0=theta0_coax_5,
        delta_theta_star=delta_theta_star_coax_5,
        delta_theta_c=delta_theta_coax_5_c,
        a=a_coax_5,
        b=b_coax_5,
    )

    f4_theta_6_coax_fn = tu.Partial(
        jd_base_functions1.f4,
        theta0=theta0_coax_6,
        delta_theta_star=delta_theta_star_coax_6,
        delta_theta_c=delta_theta_coax_6_c,
        a=a_coax_6,
        b=b_coax_6,
    )


    return (
        f2_dr_coax
        * f4_theta_4_coax
        * (f4_theta_1_coax + f6_theta_1_coax)
        * (f4_theta_5_coax_fn(theta5) + f4_theta_5_coax_fn(jnp.pi - theta5))
        * (f4_theta_6_coax_fn(theta6) + f4_theta_6_coax_fn(jnp.pi - theta6))
    )
