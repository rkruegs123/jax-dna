from functools import partial
import jax.numpy as jnp
import toml
from jax_md import energy


PARAMS = toml.load("config.toml")
PARAMS = PARAMS['default'] # FIXME: Simple for now, but eventually want to override with other namespaces and handle default correctly


# Define individual potentials
# FIXME: Could use ones from JAX-MD when appropriate (e.g. morse, harmonic). Could just add ones to JAX-MD that are missing (e.g. FENE)

# FIXME: need some initial positions from Megan
def v_fene(r, eps=PARAMS["eps_backbone"], r0=PARAMS["r0_backbone"], delt=PARAMS["delta_backbone"]):
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, wihch will yield `nan`
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

def v_smooth(x, b, x_c):
    return b*(x_c - x)**2


# Define functional forms
# FIXME: Do cutoff with Carl's method. Likely don't need r_c_low and r_c_high
def f1(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       eps, a, r0, r_c, # morse parameters
       b_low, b_high, # smoothing parameters
):
    if r_low < r and r < r_high:
        return v_morse(r, eps, r0, a) - v_morse(r_c, eps, r0, a)
    elif r_c_low < r and r < r_low:
        return eps * v_smooth(r, b_low, r_c_low)
    elif r_high < r and r < r_c_high:
        return eps * v_smooth(r, b_high, r_c_high)
    else:
        return 0.0


def f2(r, r_low, r_high, r_c_low, r_c_high, # thresholding/smoothing parameters
       k, r0, r_c, # harmonic parameters
       b_low, b_high # smoothing parameters
):
    if r_low < r and r < r_high:
        return v_harm(r, k, r0) - v_harm(r_c, k, r0)
    elif r_c_low < r and r < r_low:
        return k * v_smooth(r, b_low, r_c_low)
    elif r_high < r and r < r_c_high:
        return k * v_smooth(r, b_high, r_c_high)
    else:
        return 0.0


def get_f3(r_star, r_c,
           eps, sigma):
    return energy.multiplicative_isotropic_cutoff(lambda r: v_lj(r, eps, sigma),
                                                  r_onset=r_star, r_cutoff=r_c)

def f3(r, r_star, r_c, # thresholding/smoothing parameters
       eps, sigma, # lj parameters
       b # smoothing parameters
):

    return jnp.where(jnp.less(r, r_star),
                     v_lj(r, eps, sigma),
                     jnp.where(jnp.logical_and(jnp.less(r_star, r), jnp.less(r, r_c)),
                     # jnp.where(jnp.less(r_star, r) and jnp.less(r, r_c), # throws an error
                               eps * v_smooth(r, b, r_c),
                               jnp.zeros(r.shape[0])))


    """
    if r < r_star:
        return v_lj(r, eps, sigma)
    elif r_star < r and r < r_c:
        return eps * v_smooth(r, b, r_c)
    else:
        return 0.0
    """


def f4(theta, theta0, delta_theta_star, delta_theta_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    if theta0 - delta_theta_star < theta and theta < theta0 + delta_theta_star:
        return v_mod(theta, a, theta0)
    elif theta0 - delta_theta_c < theta and theta < theta0 - delta_theta_star:
        return v_smooth(theta, b, theta0 - delta_theta_c)
    elif theta0 + delta_theta_star < theta and theta < theta0 + delta_theta_c:
        return v_smooth(theta, b, theta0 + delta_theta_c)
    else:
        return 0.0

# FIXME: Confirm with megan that phi should be x in def of f5.
# Note: for stacking, e.g. x = cos(phi)
def f5(x, x_star, x_c, # thresholding/smoothing parameters
       a, # mod parameters
       b # smoothing parameters
):
    if x > 0:
        return 1.0
    elif x_star < x and x < 0:
        return v_mod(x, a, 0)
    elif x_c < x and x < x_star:
        return v_smooth(x, b, x_c)
    else:
        return 0.0


# FIXME: placeholders
tmp_b = 1.0
tmp_r_c_diff = 1.0
"""
f3_base = partial(f3, r_star=PARAMS["dr_star_base"], r_c=PARAMS["dr_star_base"] + tmp_r_c_diff,
                  eps=PARAMS["eps_exc"], sigma=PARAMS["sigma_base"],
                  b=tmp_b)
"""
f3_base = get_f3(r_star=PARAMS["dr_star_base"], r_c=PARAMS["dr_star_base"] + tmp_r_c_diff,
                 eps=PARAMS["eps_exc"], sigma=PARAMS["sigma_base"])
f3_back_base = partial(f3, r_star=PARAMS["dr_star_back_base"], r_c=PARAMS["dr_star_back_base"] + tmp_r_c_diff,
                       eps=PARAMS["eps_exc"], sigma=PARAMS["sigma_back_base"],
                       b=tmp_b)
f3_base_back = partial(f3, r_star=PARAMS["dr_star_base_back"], r_c=PARAMS["dr_star_base_back"] + tmp_r_c_diff,
                       eps=PARAMS["eps_exc"], sigma=PARAMS["sigma_base_back"],
                       b=tmp_b)
def exc_vol_bonded(dr_base, dr_back_base, dr_base_back):

    r_base = jnp.linalg.norm(dr_base, axis=1)
    r_back_base = jnp.linalg.norm(dr_back_base, axis=1)
    r_base_back = jnp.linalg.norm(dr_base_back, axis=1)

    # FIXME: need to add rc's and b's
    # Note: r_c must be greater than r*

    t1 = f3_base(r_base)
    t2 = f3_back_base(r_back_base)
    t3 = f3_base_back(r_base_back)
    """
    t1 = f3(r_base, PARAMS["dr_star_base"], PARAMS["dr_star_base"] + tmp_r_c_diff,
            PARAMS["eps_exc"], PARAMS["sigma_base"],
            tmp_b
    )
    t2 = f3(r_back_base, PARAMS["dr_star_back_base"], PARAMS["dr_star_back_base"] + tmp_r_c_diff,
            PARAMS["eps_exc"], PARAMS["sigma_back_base"],
            tmp_b
    )
    t3 = f3(r_base_back, PARAMS["dr_star_base_back"], PARAMS["dr_star_base_back"] + tmp_r_c_diff,
            PARAMS["eps_exc"], PARAMS["sigma_base_back"],
            tmp_b
    )
    return t1 + t2 + t3
    """
    return t1 + t2 + t3



normal = jnp.array(
  [0.0, 1.0, 0.0]
) # body frame
def stacking(dr_stack, orientations):
    # need dr_stack, theta_4, theta_5, theta_6, phi1, and phi2
    # theta_4: angle between base normal vectors
    # theta_5: angle between base normal and line passing throug stacking
    # theta_6: theta_5 but with the other base normal
    # note: for above, really just need dr_stack and base normals

    r_stack = jnp.linalg.norm(dr_stack, axis=1)

    # note: don't add body.center (and don't pass it) because we don't need it. angle can still be computed
    normals_sf = rigid_body.quaternion_rotate(orientations, normal) # space frame
    # TODO: check that normals_sf are still normalized

    # t4 = jnp.arccos(jnp.einsum('ij, ij->i', normals_sf, normals_sf)) # FIXME: should probably be (N, N) rather than (N,)
    t4 = normals_sf @ normals_sf.T # TODO: check that transposing keeps the appropriate ordering of the indices

    pdb.set_trace()


    # FIXME: FORGOT that we had already selected the neighbors by this point. Don't really want to have access to them here -- e.g., just compute t4 outside and pass in. Should check that we similarly abide by this logic in exc_volume_bonded


    # for the phi's, we also need dr_backbone and the cross product of the normal and backbone-base vectors
    # phi1:
    pass



if __name__ == "__main__":
    pass
