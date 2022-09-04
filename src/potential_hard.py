from functools import partial
import jax.numpy as jnp
import toml
from jax_md import energy
import pdb

from jax.tree_util import Partial # FIXME: only update v_fene for now...
from jax.config import config
config.update("jax_enable_x64", True)


def _v_fene(r, eps, r0, delt): # Note: named as helper as we make `v_fene` the name of the parameterized potential
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, which will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)

def v_morse(r, eps, r0, a):
    x = -(r - r0) * a
    return eps * (1 - jnp.exp(x))**2

def v_harmonic(r, k, r0):
    return k / 2 * (r - r0)**2

def v_lj(r, eps, sigma):
    x = (sigma / r)**12 - (sigma / r)**6
    return 4 * eps * x

def v_mod(theta, a, theta0):
    return 1 - a*(theta - theta0)**2


# Define functional forms
def f1(r, r_low, r_high,
       eps, a, r0, r_c):
    return jnp.where((r_low < r) & (r < r_high),
                     v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a), 0.0)

def f2(r, r_low, r_high,
       k, r0, r_c):
    return jnp.where((r_low < r) & (r < r_high),
                     v_harmonic(r, k, r0) - v_harmonic(r_c, k, r0), 0.0)

def f3(r, r_star, eps, sigma):
    return jnp.where(r < r_star,
                     v_lj(r, eps, sigma), 0.0)

def f4(theta, theta0, delta_theta_star, a):
    return jnp.where((theta0 - delta_theta_star < theta) & (theta < theta0 + delta_theta_star),
                     v_mod(theta, a, theta0), 0.0)

def f5(x, x_star, a):
    return jnp.where(x > 0.0, 1.0,
                     jnp.where((x_star < x) & (x < 0.0),
                               v_mod(x, a, 0), 0.0))


def v_fene(r, params):
    FENE_PARAMS = params["fene"]
    fene_val = _v_fene(r, eps=FENE_PARAMS["eps_backbone"], r0=FENE_PARAMS["r0_backbone"],
                       delt=FENE_PARAMS["delta_backbone"])
    # return fene_val


    # Thresholded version -- analogous to allowing broken bakcbone from oxDNA
    rbackr0 = r - FENE_PARAMS["r0_backbone"]

    return jnp.where(jnp.abs(rbackr0) >= FENE_PARAMS["delta_backbone"],
                     1.0e12,
                     fene_val)

def v_fene2(r, eps_backbone, delta_backbone, r0_backbone):
    fene_val = _v_fene(r, eps=eps_backbone, r0=r0_backbone,
                       delt=delta_backbone)
    # return fene_val
    rbackr0 = r - r0_backbone

    return jnp.where(jnp.abs(rbackr0) >= delta_backbone,
                     1.0e12,
                     fene_val)

def exc_vol_bonded(dr_base, dr_back_base, dr_base_back, params):
    # Note: r_c must be greater than r*
    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    EXC_VOL_PARAMS = params["excluded_volume"]

    f3_base_exc_vol = f3(r_base,
                         r_star=EXC_VOL_PARAMS["dr_star_base"],
                         eps=EXC_VOL_PARAMS["eps_exc"],
                         sigma=EXC_VOL_PARAMS["sigma_base"])
    f3_back_base_exc_vol = f3(r_back_base,
                              r_star=EXC_VOL_PARAMS["dr_star_back_base"],
                              eps=EXC_VOL_PARAMS["eps_exc"],
                              sigma=EXC_VOL_PARAMS["sigma_back_base"])
    f3_base_back_exc_vol = f3(r_base_back,
                              r_star=EXC_VOL_PARAMS["dr_star_base_back"],
                              eps=EXC_VOL_PARAMS["eps_exc"],
                              sigma=EXC_VOL_PARAMS["sigma_base_back"])

    return f3_base_exc_vol + f3_back_base_exc_vol + f3_base_back_exc_vol


# FIXME: lots of duplicated computation with exc_vol_bonded
# E.g. Should (a) compute the r's outside these functions, and (b) have a base that we then add on to
def exc_vol_unbonded(dr_base, dr_backbone, dr_back_base, dr_base_back, params):
    r_back = jnp.linalg.norm(dr_backbone, axis=1)

    EXC_VOL_PARAMS = params["excluded_volume"]

    f3_back_exc_vol = f3(r_back,
                         r_star=EXC_VOL_PARAMS["dr_star_backbone"],
                         eps=EXC_VOL_PARAMS["eps_exc"],
                         sigma=EXC_VOL_PARAMS["sigma_backbone"])
    return f3_back_exc_vol + exc_vol_bonded(dr_base, dr_back_base, dr_base_back, params)


def stacking(dr_stack, theta4, theta5, theta6, cosphi1, cosphi2, params):
    # need dr_stack, theta_4, theta_5, theta_6, phi1, and phi2
    # theta_4: angle between base normal vectors
    # theta_5: angle between base normal and line passing throug stacking
    # theta_6: theta_5 but with the other base normal
    # note: for above, really just need dr_stack and base normals

    STACK_PARAMS = params["stacking"]
    r_stack = jnp.linalg.norm(dr_stack, axis=1)


    f1_dr_stack = f1(r_stack,
                     r_low=STACK_PARAMS["dr_low_stack"],
                     r_high=STACK_PARAMS["dr_high_stack"],
                     eps=STACK_PARAMS["eps_stack"],
                     a=STACK_PARAMS["a_stack"],
                     r0=STACK_PARAMS["dr0_stack"],
                     r_c=STACK_PARAMS["dr_c_stack"])

    f4_theta_4_stack = f4(theta4,
                          theta0=STACK_PARAMS["theta0_stack_4"],
                          delta_theta_star=STACK_PARAMS["delta_theta_star_stack_4"],
                          a=STACK_PARAMS["a_stack_4"])

    f4_theta_5p_stack = f4(theta5,
                           theta0=STACK_PARAMS["theta0_stack_5"],
                           delta_theta_star=STACK_PARAMS["delta_theta_star_stack_5"],
                           a=STACK_PARAMS["a_stack_5"])

    f4_theta_6p_stack = f4(theta6,
                           theta0=STACK_PARAMS["theta0_stack_6"],
                           delta_theta_star=STACK_PARAMS["delta_theta_star_stack_6"],
                           a=STACK_PARAMS["a_stack_6"])

    f5_neg_cosphi1_stack = f5(-cosphi1,
                              x_star=STACK_PARAMS["neg_cos_phi1_star_stack"],
                              a=STACK_PARAMS["a_stack_1"])

    f5_neg_cosphi2_stack = f5(-cosphi2,
                              x_star=STACK_PARAMS["neg_cos_phi2_star_stack"],
                              a=STACK_PARAMS["a_stack_2"])

    return f1_dr_stack * f4_theta_4_stack \
        * f4_theta_5p_stack * f4_theta_6p_stack \
        * f5_neg_cosphi1_stack * f5_neg_cosphi2_stack



def hydrogen_bonding(dr_hb, theta1, theta2, theta3, theta4, theta7, theta8, params):
    HB_PARAMS = params["hydrogen_bonding"]
    r_hb = jnp.linalg.norm(dr_hb, axis=1)

    f1_dr_hb = f1(r_hb,
                  r_low=HB_PARAMS["dr_low_hb"],
                  r_high=HB_PARAMS["dr_high_hb"],
                  eps=HB_PARAMS["eps_hb"],
                  a=HB_PARAMS["a_hb"],
                  r0=HB_PARAMS["dr0_hb"],
                  r_c=HB_PARAMS["dr_c_hb"])

    f4_theta_1_hb = f4(theta1,
                       theta0=HB_PARAMS["theta0_hb_1"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_1"],
                       a=HB_PARAMS["a_hb_1"])

    f4_theta_2_hb = f4(theta2,
                       theta0=HB_PARAMS["theta0_hb_2"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_2"],
                       a=HB_PARAMS["a_hb_2"])

    f4_theta_3_hb = f4(theta3,
                       theta0=HB_PARAMS["theta0_hb_3"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_3"],
                       a=HB_PARAMS["a_hb_3"])

    f4_theta_4_hb = f4(theta4,
                       theta0=HB_PARAMS["theta0_hb_4"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_4"],
                       a=HB_PARAMS["a_hb_4"])

    f4_theta_7_hb = f4(theta7,
                       theta0=HB_PARAMS["theta0_hb_7"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_7"],
                       a=HB_PARAMS["a_hb_7"])

    f4_theta_8_hb = f4(theta8,
                       theta0=HB_PARAMS["theta0_hb_8"],
                       delta_theta_star=HB_PARAMS["delta_theta_star_hb_8"],
                       a=HB_PARAMS["a_hb_8"])

    return f1_dr_hb * f4_theta_1_hb * f4_theta_2_hb \
        * f4_theta_3_hb * f4_theta_4_hb * f4_theta_7_hb * f4_theta_8_hb


# Cross Stacking
def cross_stacking(dr_hb, theta1, theta2, theta3, theta4, theta7, theta8, params):
    CROSS_PARAMS = params["cross_stacking"]
     # FIXME: repeated computation. May just want to do in the energy function and pass along
    r_hb = jnp.linalg.norm(dr_hb, axis=1)

    f2_dr_cross = f2(r_hb,
                     r_low=CROSS_PARAMS["dr_low_cross"],
                     r_high=CROSS_PARAMS["dr_high_cross"],
                     k=CROSS_PARAMS["k"],
                     r0=CROSS_PARAMS["r0_cross"],
                     r_c=CROSS_PARAMS["dr_c_cross"])

    f4_theta_1_cross = f4(theta1,
                          theta0=CROSS_PARAMS["theta0_cross_1"],
                          delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_1"],
                          a=CROSS_PARAMS["a_cross_1"])

    f4_theta_2_cross = f4(theta2,
                          theta0=CROSS_PARAMS["theta0_cross_2"],
                          delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_2"],
                          a=CROSS_PARAMS["a_cross_2"])

    f4_theta_3_cross = f4(theta3,
                          theta0=CROSS_PARAMS["theta0_cross_3"],
                          delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_3"],
                          a=CROSS_PARAMS["a_cross_3"])

    f4_theta_4_cross_fn = Partial(f4,
                                  theta0=CROSS_PARAMS["theta0_cross_4"],
                                  delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_4"],
                                  a=CROSS_PARAMS["a_cross_4"])

    f4_theta_7_cross_fn = Partial(f4,
                                  theta0=CROSS_PARAMS["theta0_cross_7"],
                                  delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_7"],
                                  a=CROSS_PARAMS["a_cross_7"])

    f4_theta_8_cross_fn = Partial(f4,
                                  theta0=CROSS_PARAMS["theta0_cross_8"],
                                  delta_theta_star=CROSS_PARAMS["delta_theta_star_cross_8"],
                                  a=CROSS_PARAMS["a_cross_8"])

    return f2_dr_cross * f4_theta_1_cross \
        * f4_theta_2_cross * f4_theta_3_cross \
        * (f4_theta_4_cross_fn(theta4) + f4_theta_4_cross_fn(jnp.pi - theta4)) \
        * (f4_theta_7_cross_fn(theta7) + f4_theta_7_cross_fn(jnp.pi - theta7)) \
        * (f4_theta_8_cross_fn(theta8) + f4_theta_8_cross_fn(jnp.pi - theta8))


# Coaxial stacking
def coaxial_stacking(dr_stack, theta4, theta1, theta5, theta6, cosphi3, cosphi4, params):
    COAX_PARAMS = params["coaxial_stacking"]
    r_stack = jnp.linalg.norm(dr_stack, axis=1)


    f2_dr_coax = f2(r_stack, # FIXME: naming, and many like this. Should be f2_dr_stack_coax
                    r_low=COAX_PARAMS["dr_low_coax"],
                    r_high=COAX_PARAMS["dr_high_coax"],
                    k=COAX_PARAMS["k_coax"],
                    r0=COAX_PARAMS["dr0_coax"],
                    r_c=COAX_PARAMS["dr_c_coax"])

    f4_theta_4_coax = f4(theta4,
                         theta0=COAX_PARAMS["theta0_coax_4"],
                         delta_theta_star=COAX_PARAMS["delta_theta_star_coax_4"],
                         a=COAX_PARAMS["a_coax_4"])

    f4_theta_1_coax_fn = Partial(f4,
                                 theta0=COAX_PARAMS["theta0_coax_1"],
                                 delta_theta_star=COAX_PARAMS["delta_theta_star_coax_1"],
                                 a=COAX_PARAMS["a_coax_1"])

    f4_theta_5_coax_fn = Partial(f4,
                                 theta0=COAX_PARAMS["theta0_coax_5"],
                                 delta_theta_star=COAX_PARAMS["delta_theta_star_coax_5"],
                                 a=COAX_PARAMS["a_coax_5"])

    f4_theta_6_coax_fn = Partial(f4,
                                 theta0=COAX_PARAMS["theta0_coax_6"],
                                 delta_theta_star=COAX_PARAMS["delta_theta_star_coax_6"],
                                 a=COAX_PARAMS["a_coax_6"])

    f5_cosphi3_coax = f5(cosphi3,
                         x_star=COAX_PARAMS["cos_phi3_star_coax"],
                         a=COAX_PARAMS["a_coax_3p"])

    f5_cosphi4_coax = f5(cosphi4,
                         x_star=COAX_PARAMS["cos_phi4_star_coax"],
                         a=COAX_PARAMS["a_coax_4p"])

    return f2_dr_coax * f4_theta_4_coax \
        * (f4_theta_1_coax_fn(theta1) + f4_theta_1_coax_fn(2 * jnp.pi - theta1)) \
        * (f4_theta_5_coax_fn(theta5) + f4_theta_5_coax_fn(jnp.pi - theta5)) \
        * (f4_theta_6_coax_fn(theta6) + f4_theta_6_coax_fn(jnp.pi - theta6)) \
        * f5_cosphi3_coax * f5_cosphi4_coax



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    rs = np.linspace(0.0, 3.0, 50)
    ys = f1(rs, 1.0, 2.0,
            0.5, 1.0, 1.0, 1.5)
    plt.plot(rs, ys)
    plt.show()
