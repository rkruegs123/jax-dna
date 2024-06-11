# ruff: noqa
# fmt: off
import pdb
import jax.numpy as jnp
from jax.tree_util import Partial
from jax import jit

from jax_dna.common.base_functions import f1, f2, f3, f4, f5, f6, v_fene


@jit
def smooth_abs(x, eps=1e-10):
    """
    A smooth absolute value function. Note that a non-zero eps
    gives continuous first dervatives.

    https://math.stackexchange.com/questions/1172472/differentiable-approximation-of-the-absolute-value-function
    """
    return jnp.sqrt(x**2 + eps)

@jit
def v_fene_smooth(r, eps_backbone, r0_backbone, delta_backbone, fmax=500, finf=4.0):
    eps = eps_backbone; r0 = r0_backbone; delt = delta_backbone

    diff = smooth_abs(r - r0)

    delt2 = delt**2
    eps2 = eps**2
    fmax2 = fmax**2
    xmax = (-eps + jnp.sqrt(eps2 + 4*fmax2*delt2)) / (2*fmax)

    # precompute terms for smoothed case
    fene_xmax = -(eps / 2.0) * jnp.log(1.0 - xmax**2 / delt2)
    long_xmax = (fmax - finf)*xmax*jnp.log(xmax) + finf*xmax
    smoothed_energy = (fmax - finf)*xmax*jnp.log(diff) + finf*diff - long_xmax + fene_xmax

    return jnp.where(diff > xmax, smoothed_energy, v_fene(r, eps, r0, delt))


@jit
def exc_vol_bonded(
        dr_base, dr_back_base, dr_base_back,
        eps_exc,
        dr_star_base, sigma_base, b_base, dr_c_base, # f3(dr_base)
        dr_star_back_base, sigma_back_base, b_back_base, dr_c_back_base, # f3(dr_back_base)
        dr_star_base_back, sigma_base_back, b_base_back, dr_c_base_back # f3(dr_base_back)
):
    # Note: r_c must be greater than r*
    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    f3_base_exc_vol = f3(r_base,
                         r_star=dr_star_base,
                         r_c=dr_c_base,
                         eps=eps_exc,
                         sigma=sigma_base,
                         b=b_base)

    f3_back_base_exc_vol = f3(r_back_base,
                              r_star=dr_star_back_base,
                              r_c=dr_c_back_base,
                              eps=eps_exc,
                              sigma=sigma_back_base,
                              b=b_back_base)

    f3_base_back_exc_vol = f3(r_base_back,
                              r_star=dr_star_base_back,
                              r_c=dr_c_base_back,
                              eps=eps_exc,
                              sigma=sigma_base_back,
                              b=b_base_back)

    return f3_base_exc_vol + f3_back_base_exc_vol + f3_base_back_exc_vol


@jit
def exc_vol_unbonded(
        dr_base, dr_backbone, dr_back_base, dr_base_back,
        eps_exc,
        dr_star_base, sigma_base, b_base, dr_c_base, # f3(dr_base)
        dr_star_back_base, sigma_back_base, b_back_base, dr_c_back_base, # f3(dr_back_base)
        dr_star_base_back, sigma_base_back, b_base_back, dr_c_base_back, # f3(dr_base_back)
        dr_star_backbone, sigma_backbone, b_backbone, dr_c_backbone # f3(backbone)
):
    r_back = jnp.linalg.norm(dr_backbone, axis=1)
    f3_back_exc_vol = f3(r_back,
                         r_star=dr_star_backbone,
                         r_c=dr_c_backbone,
                         eps=eps_exc,
                         sigma=sigma_backbone,
                         b=b_backbone)

    return f3_back_exc_vol + exc_vol_bonded(
        dr_base, dr_back_base, dr_base_back,
        eps_exc,
        dr_star_base, sigma_base, b_base, dr_c_base,
        dr_star_back_base, sigma_back_base, b_back_base, dr_c_back_base,
        dr_star_base_back, sigma_base_back, b_base_back, dr_c_base_back)


# Note that we use r_stack instead of dr_stack
# FIXME: fix this one with bs and rcs
@jit
def stacking(
        r_stack, theta4, theta5, theta6, cosphi1, cosphi2, # observables
        dr_low_stack, dr_high_stack, eps_stack, a_stack, dr0_stack, dr_c_stack,
        dr_c_low_stack, dr_c_high_stack, b_low_stack, b_high_stack,
        theta0_stack_4, delta_theta_star_stack_4, a_stack_4, delta_theta_stack_4_c, b_stack_4,
        theta0_stack_5, delta_theta_star_stack_5, a_stack_5, delta_theta_stack_5_c, b_stack_5,
        theta0_stack_6, delta_theta_star_stack_6, a_stack_6, delta_theta_stack_6_c, b_stack_6,
        neg_cos_phi1_star_stack, a_stack_1, neg_cos_phi1_c_stack, b_neg_cos_phi1_stack,
        neg_cos_phi2_star_stack, a_stack_2, neg_cos_phi2_c_stack, b_neg_cos_phi2_stack
):
    # need dr_stack, theta_4, theta_5, theta_6, phi1, and phi2
    # theta_4: angle between base normal vectors
    # theta_5: angle between base normal and line passing throug stacking
    # theta_6: theta_5 but with the other base normal
    # note: for above, really just need dr_stack and base normals

    # r_stack = jnp.linalg.norm(dr_stack, axis=1)

    f1_dr_stack = f1(r_stack,
                     r_low=dr_low_stack,
                     r_high=dr_high_stack,
                     r_c_low=dr_c_low_stack,
                     r_c_high=dr_c_high_stack,
                     eps=eps_stack,
                     a=a_stack,
                     r0=dr0_stack,
                     r_c=dr_c_stack,
                     b_low=b_low_stack,
                     b_high=b_high_stack)

    f4_theta_4_stack = f4(theta4,
                          theta0=theta0_stack_4,
                          delta_theta_star=delta_theta_star_stack_4,
                          delta_theta_c=delta_theta_stack_4_c,
                          a=a_stack_4,
                          b=b_stack_4)

    f4_theta_5p_stack = f4(theta5,
                           theta0=theta0_stack_5,
                           delta_theta_star=delta_theta_star_stack_5,
                           delta_theta_c=delta_theta_stack_5_c,
                           a=a_stack_5,
                           b=b_stack_5)

    f4_theta_6p_stack = f4(theta6,
                           theta0=theta0_stack_6,
                           delta_theta_star=delta_theta_star_stack_6,
                           delta_theta_c=delta_theta_stack_6_c,
                           a=a_stack_6,
                           b=b_stack_6)

    f5_neg_cosphi1_stack = f5(-cosphi1,
                              x_star=neg_cos_phi1_star_stack,
                              x_c=neg_cos_phi1_c_stack,
                              a=a_stack_1,
                              b=b_neg_cos_phi1_stack)

    f5_neg_cosphi2_stack = f5(-cosphi2,
                              x_star=neg_cos_phi2_star_stack,
                              x_c=neg_cos_phi2_c_stack,
                              a=a_stack_2,
                              b=b_neg_cos_phi2_stack)

    return f1_dr_stack * f4_theta_4_stack \
        * f4_theta_5p_stack * f4_theta_6p_stack \
        * f5_neg_cosphi1_stack * f5_neg_cosphi2_stack

@jit
def hydrogen_bonding(
        dr_hb, theta1, theta2, theta3, theta4, theta7, theta8, # observables
        dr_low_hb, dr_high_hb, dr_c_low_hb, dr_c_high_hb, eps_hb, a_hb, # f1_dr_hb
        dr0_hb, dr_c_hb, b_low_hb, b_high_hb, # f1_dr_hb (cont.)
        theta0_hb_1, delta_theta_star_hb_1, a_hb_1, delta_theta_hb_1_c, b_hb_1, # f4_theta_1_hb
        theta0_hb_2, delta_theta_star_hb_2, a_hb_2, delta_theta_hb_2_c, b_hb_2, # f4_theta_2_hb
        theta0_hb_3, delta_theta_star_hb_3, a_hb_3, delta_theta_hb_3_c, b_hb_3, # f4_theta_3_hb
        theta0_hb_4, delta_theta_star_hb_4, a_hb_4, delta_theta_hb_4_c, b_hb_4, # f4_theta_4_hb
        theta0_hb_7, delta_theta_star_hb_7, a_hb_7, delta_theta_hb_7_c, b_hb_7, # f4_theta_7_hb
        theta0_hb_8, delta_theta_star_hb_8, a_hb_8, delta_theta_hb_8_c, b_hb_8, # f4_theta_8_hb
):
    r_hb = jnp.linalg.norm(dr_hb, axis=1)

    f1_dr_hb = f1(r_hb,
                  r_low=dr_low_hb,
                  r_high=dr_high_hb,
                  r_c_low=dr_c_low_hb,
                  r_c_high=dr_c_high_hb,
                  eps=eps_hb,
                  a=a_hb,
                  r0=dr0_hb,
                  r_c=dr_c_hb,
                  b_low=b_low_hb,
                  b_high=b_high_hb)

    f4_theta_1_hb = f4(theta1,
                       theta0=theta0_hb_1,
                       delta_theta_star=delta_theta_star_hb_1,
                       delta_theta_c=delta_theta_hb_1_c,
                       a=a_hb_1,
                       b=b_hb_1)

    f4_theta_2_hb = f4(theta2,
                       theta0=theta0_hb_2,
                       delta_theta_star=delta_theta_star_hb_2,
                       delta_theta_c=delta_theta_hb_2_c,
                       a=a_hb_2,
                       b=b_hb_2)

    f4_theta_3_hb = f4(theta3,
                       theta0=theta0_hb_3,
                       delta_theta_star=delta_theta_star_hb_3,
                       delta_theta_c=delta_theta_hb_3_c,
                       a=a_hb_3,
                       b=b_hb_3)

    f4_theta_4_hb = f4(theta4,
                       theta0=theta0_hb_4,
                       delta_theta_star=delta_theta_star_hb_4,
                       delta_theta_c=delta_theta_hb_4_c,
                       a=a_hb_4,
                       b=b_hb_4)

    f4_theta_7_hb = f4(theta7,
                       theta0=theta0_hb_7,
                       delta_theta_star=delta_theta_star_hb_7,
                       delta_theta_c=delta_theta_hb_7_c,
                       a=a_hb_7,
                       b=b_hb_7)

    f4_theta_8_hb = f4(theta8,
                       theta0=theta0_hb_8,
                       delta_theta_star=delta_theta_star_hb_8,
                       delta_theta_c=delta_theta_hb_8_c,
                       a=a_hb_8,
                       b=b_hb_8)

    return f1_dr_hb * f4_theta_1_hb * f4_theta_2_hb \
        * f4_theta_3_hb * f4_theta_4_hb * f4_theta_7_hb * f4_theta_8_hb

@jit
def cross_stacking(r_hb, theta1, theta2, theta3, theta4, theta7, theta8,
                   dr_low_cross, dr_high_cross, dr_c_low_cross, dr_c_high_cross, # f2_dr_cross
                   k_cross, r0_cross, dr_c_cross, b_low_cross, b_high_cross, # f2_dr_cross (cont.)
                   theta0_cross_1, delta_theta_star_cross_1, delta_theta_cross_1_c, a_cross_1, b_cross_1, # f4(theta1)
                   theta0_cross_2, delta_theta_star_cross_2, delta_theta_cross_2_c, a_cross_2, b_cross_2, #f4(theta2)
                   theta0_cross_3, delta_theta_star_cross_3, delta_theta_cross_3_c, a_cross_3, b_cross_3, #f4(theta3)
                   theta0_cross_4, delta_theta_star_cross_4, delta_theta_cross_4_c, a_cross_4, b_cross_4, #f4(theta4)
                   theta0_cross_7, delta_theta_star_cross_7, delta_theta_cross_7_c, a_cross_7, b_cross_7, #f7(theta7)
                   theta0_cross_8, delta_theta_star_cross_8, delta_theta_cross_8_c, a_cross_8, b_cross_8 #f8(theta8)
):
    f2_dr_cross = f2(r_hb,
                     r_low=dr_low_cross,
                     r_high=dr_high_cross,
                     r_c_low=dr_c_low_cross,
                     r_c_high=dr_c_high_cross,
                     k=k_cross,
                     r0=r0_cross,
                     r_c=dr_c_cross,
                     b_low=b_low_cross,
                     b_high=b_high_cross)

    f4_theta_1_cross = f4(theta1,
                          theta0=theta0_cross_1,
                          delta_theta_star=delta_theta_star_cross_1,
                          delta_theta_c=delta_theta_cross_1_c,
                          a=a_cross_1,
                          b=b_cross_1)
    f4_theta_2_cross = f4(theta2,
                          theta0=theta0_cross_2,
                          delta_theta_star=delta_theta_star_cross_2,
                          delta_theta_c=delta_theta_cross_2_c,
                          a=a_cross_2,
                          b=b_cross_2)
    f4_theta_3_cross = f4(theta3,
                          theta0=theta0_cross_3,
                          delta_theta_star=delta_theta_star_cross_3,
                          delta_theta_c=delta_theta_cross_3_c,
                          a=a_cross_3,
                          b=b_cross_3)
    f4_theta_4_cross_fn = Partial(f4,
                                  theta0=theta0_cross_4,
                                  delta_theta_star=delta_theta_star_cross_4,
                                  delta_theta_c=delta_theta_cross_4_c,
                                  a=a_cross_4,
                                  b=b_cross_4)
    f4_theta_7_cross_fn = Partial(f4,
                               theta0=theta0_cross_7,
                               delta_theta_star=delta_theta_star_cross_7,
                               delta_theta_c=delta_theta_cross_7_c,
                               a=a_cross_7,
                               b=b_cross_7)
    f4_theta_8_cross_fn = Partial(f4,
                                  theta0=theta0_cross_8,
                                  delta_theta_star=delta_theta_star_cross_8,
                                  delta_theta_c=delta_theta_cross_8_c,
                                  a=a_cross_8,
                                  b=b_cross_8)
    return f2_dr_cross * f4_theta_1_cross * f4_theta_2_cross * f4_theta_3_cross \
        * (f4_theta_4_cross_fn(theta4) + f4_theta_4_cross_fn(jnp.pi - theta4)) \
        * (f4_theta_7_cross_fn(theta7) + f4_theta_7_cross_fn(jnp.pi - theta7)) \
        * (f4_theta_8_cross_fn(theta8) + f4_theta_8_cross_fn(jnp.pi - theta8))

@jit
def coaxial_stacking(dr_stack, theta4, theta1, theta5, theta6, cosphi3, cosphi4, # observables
                     dr_low_coax, dr_high_coax, dr_c_low_coax, dr_c_high_coax, # f2(dr_stack)
                     k_coax, dr0_coax, dr_c_coax, b_low_coax, b_high_coax, # f2(dr_stack), cont.
                     theta0_coax_4, delta_theta_star_coax_4, delta_theta_coax_4_c, a_coax_4, b_coax_4, # f4(theta4)
                     theta0_coax_1, delta_theta_star_coax_1, delta_theta_coax_1_c, a_coax_1, b_coax_1, # f4(theta1)
                     theta0_coax_5, delta_theta_star_coax_5, delta_theta_coax_5_c, a_coax_5, b_coax_5, # f4(theta5)
                     theta0_coax_6, delta_theta_star_coax_6, delta_theta_coax_6_c, a_coax_6, b_coax_6, # f4(theta6)
                     cos_phi3_star_coax, cos_phi3_c_coax, a_coax_3p, b_cos_phi3_coax, # f5(cosphi3)
                     cos_phi4_star_coax, cos_phi4_c_coax, a_coax_4p, b_cos_phi4_coax # f5(cosphi4)
):
    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    f2_dr_coax = f2(r_stack,
                    r_low=dr_low_coax,
                    r_high=dr_high_coax,
                    r_c_low=dr_c_low_coax,
                    r_c_high=dr_c_high_coax,
                    k=k_coax,
                    r0=dr0_coax,
                    r_c=dr_c_coax,
                    b_low=b_low_coax,
                    b_high=b_high_coax)

    f4_theta_4_coax = f4(theta4,
                         theta0=theta0_coax_4,
                         delta_theta_star=delta_theta_star_coax_4,
                         delta_theta_c=delta_theta_coax_4_c,
                         a=a_coax_4,
                         b=b_coax_4)

    f4_theta_1_coax_fn = Partial(f4,
                                 theta0=theta0_coax_1,
                                 delta_theta_star=delta_theta_star_coax_1,
                                 delta_theta_c=delta_theta_coax_1_c,
                                 a=a_coax_1,
                                 b=b_coax_1)

    f4_theta_5_coax_fn = Partial(f4,
                                 theta0=theta0_coax_5,
                                 delta_theta_star=delta_theta_star_coax_5,
                                 delta_theta_c=delta_theta_coax_5_c,
                                 a=a_coax_5,
                                 b=b_coax_5)

    f4_theta_6_coax_fn = Partial(f4,
                                 theta0=theta0_coax_6,
                                 delta_theta_star=delta_theta_star_coax_6,
                                 delta_theta_c=delta_theta_coax_6_c,
                                 a=a_coax_6,
                                 b=b_coax_6)

    f5_cosphi3_coax = f5(cosphi3,
                         x_star=cos_phi3_star_coax,
                         x_c=cos_phi3_c_coax,
                         a=a_coax_3p,
                         b=b_cos_phi3_coax)

    f5_cosphi4_coax = f5(cosphi4,
                         x_star=cos_phi4_star_coax,
                         x_c=cos_phi4_c_coax,
                         a=a_coax_4p,
                         b=b_cos_phi4_coax)

    return f2_dr_coax * f4_theta_4_coax \
        * (f4_theta_1_coax_fn(theta1) + f4_theta_1_coax_fn(2 * jnp.pi - theta1)) \
        * (f4_theta_5_coax_fn(theta5) + f4_theta_5_coax_fn(jnp.pi - theta5)) \
        * (f4_theta_6_coax_fn(theta6) + f4_theta_6_coax_fn(jnp.pi - theta6)) \
        * f5_cosphi3_coax * f5_cosphi4_coax


@jit
def coaxial_stacking2(dr_stack, theta4, theta1, theta5, theta6, # observables
                      dr_low_coax, dr_high_coax, dr_c_low_coax, dr_c_high_coax, # f2(dr_stack)
                      k_coax, dr0_coax, dr_c_coax, b_low_coax, b_high_coax, # f2(dr_stack), cont.
                      theta0_coax_4, delta_theta_star_coax_4, delta_theta_coax_4_c, a_coax_4, b_coax_4, # f4(theta4)
                      theta0_coax_1, delta_theta_star_coax_1, delta_theta_coax_1_c, a_coax_1, b_coax_1, # f4(theta1)
                      A_coax_1_f6, B_coax_1_f6, # f6(theta1)
                      theta0_coax_5, delta_theta_star_coax_5, delta_theta_coax_5_c, a_coax_5, b_coax_5, # f4(theta5)
                      theta0_coax_6, delta_theta_star_coax_6, delta_theta_coax_6_c, a_coax_6, b_coax_6, # f4(theta6)
):
    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    f2_dr_coax = f2(r_stack,
                    r_low=dr_low_coax,
                    r_high=dr_high_coax,
                    r_c_low=dr_c_low_coax,
                    r_c_high=dr_c_high_coax,
                    k=k_coax,
                    r0=dr0_coax,
                    r_c=dr_c_coax,
                    b_low=b_low_coax,
                    b_high=b_high_coax)

    f4_theta_4_coax = f4(theta4,
                         theta0=theta0_coax_4,
                         delta_theta_star=delta_theta_star_coax_4,
                         delta_theta_c=delta_theta_coax_4_c,
                         a=a_coax_4,
                         b=b_coax_4)

    f4_theta_1_coax_fn = Partial(f4,
                                 theta0=theta0_coax_1,
                                 delta_theta_star=delta_theta_star_coax_1,
                                 delta_theta_c=delta_theta_coax_1_c,
                                 a=a_coax_1,
                                 b=b_coax_1)

    f6_theta_1_coax_fn = Partial(f6,
                                 a=A_coax_1_f6,
                                 b=B_coax_1_f6)

    f4_theta_5_coax_fn = Partial(f4,
                                 theta0=theta0_coax_5,
                                 delta_theta_star=delta_theta_star_coax_5,
                                 delta_theta_c=delta_theta_coax_5_c,
                                 a=a_coax_5,
                                 b=b_coax_5)

    f4_theta_6_coax_fn = Partial(f4,
                                 theta0=theta0_coax_6,
                                 delta_theta_star=delta_theta_star_coax_6,
                                 delta_theta_c=delta_theta_coax_6_c,
                                 a=a_coax_6,
                                 b=b_coax_6)

    return f2_dr_coax * f4_theta_4_coax \
        * (f4_theta_1_coax_fn(theta1) + f6_theta_1_coax_fn(theta1)) \
        * (f4_theta_5_coax_fn(theta5) + f4_theta_5_coax_fn(jnp.pi - theta5)) \
        * (f4_theta_6_coax_fn(theta6) + f4_theta_6_coax_fn(jnp.pi - theta6))


if __name__ == "__main__":
    pass
