import functools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import pdb

from jax import jit
from jax import vmap
from jax import random
from jax import lax
from jax import test_util as jtu

from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import quantity
from jax_md import simulate
from jax_md import space
from jax_md import smap
from jax_md import energy
from jax_md import test_util
from jax_md import partition
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion
# from jax_md.colab_tools import renderer

from functools import partial



FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 100


f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
  DTYPE += [f64]

@partial(vmap, in_axes=(0, None))
def rand_quat(key, dtype):
  return rigid_body.random_quaternion(key, dtype)


# Define individual potentials
# FIXME: Could use ones from JAX-MD when appropriate (e.g. morse, harmonic). Could just add ones to JAX-MD that are missing (e.g. FENE)

# FIXME: need some initial positions from Megan
# FIXME: naming a bit off here, because we made all the others take in r instead of dr
def v_fene(dr_bb, eps=2.0, r0=0.7525, delt=0.25):
    r = jnp.linalg.norm(dr_bb, axis=2)
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

def f3(r, r_star, r_c, # thresholding/smoothing parameters
       eps, sigma, # lj parameters
       b # smoothing parameters
):
    if r < r_star:
        return v_lj(r, eps, sigma)
    elif r_star < r and r < r_c:
        return eps * v_smooth(r, b, r_c)
    else:
        return 0.0

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

# f3_bb = partial(displacement_fn, **kwargs)
def exc_vol_bonded(dr1, dr2, dr3):

    # FIXME: real values of parameters
    eps_exc = 1.0
    sigma_base = 1.0
    r_base_star = 1.0
    sigma_bb_base = 1.0 # FIXME: maybe bb -> back
    r_bb_base_star = 1.0
    sigma_

    if dr1.shape != (3,):
        pdb.set_trace()
        return 1.0
    if dr2.shape != (3,):
        pdb.set_trace()
        return 2.0
    if dr3.shape != (3,):
        pdb.set_trace()
        return 3.0


    return jnp.linalg.norm(dr1) + jnp.linalg.norm(dr2) + jnp.linalg.norm(dr3)



mapped_exc_vol_bonded = vmap(vmap(exc_vol_bonded, (0, 0, None)), (0, None, 0))


def static_energy_fn_factory(displacement_fn, bb_site, stack_site, base_site,
                             neighbors=None):

    d = space.map_bond(partial(displacement_fn))

    def energy_fn(body: RigidBody, **kwargs) -> float:
        Q = body.orientation
        bb_sites = body.center + rigid_body.quaternion_rotate(Q, bb_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # FIXME: flatten, make
        # Note: I believe we don't have to flatten. In Sam's original code, R_sites contained *all* N*3 interaction sites

        # FIXME: for neighbors, this will change
        pdb.set_trace()
        # d = space.map_product(partial(displacement_fn, **kwargs))

        dr_bb_bb = d(bb_sites, bb_sites) # N x N x 3
        fene = v_fene(dr_bb_bb)
        pdb.set_trace()

        dr_base_base = d(base_sites, base_sites)
        dr_bb_base = d(bb_sites, base_sites)
        dr_base_bb = d(base_sites, bb_sites)

        pdb.set_trace()

        return jnp.sum(fene) / 2.0 # FIXME: placeholder

    return energy_fn


if __name__ == "__main__":


    # Next: Read original carl notebook, then look at my oxDNA notebook and corroborate with data, smooth versions for each f_i, then simple...


    # Bug in rigid body -- Nose-Hoover defaults to f32(1.0) rather than a RigidBody with this value
    shape = rigid_body.point_union_shape(
      onp.array([[0.0, 0.0, 0.0]], f32),
      f32(1.0)
    ) # just to get the mass from

    mass = shape.mass()

    box_size = 20.0
    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = DTYPE[0]

    ## Uncomment to: Get random rigid body
    # R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)

    N = 5

    ## Initialize centers of mass via evenly spaced vertical heights
    R = jnp.array([
        [0.0, 0.0, 4.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 6.0],
        [0.0, 0.0, 7.0],
        [0.0, 0.0, 8.0]
    ])

    ## Uncomment to: Get 5 different quaternions
    # quat_key = random.split(quat_key, N)
    # quaternion = rand_quat(quat_key, dtype) # FIXME: Does this not generate *pure* quaternions?

    ## Get one quaternion and copy it 5 times
    quat_key = random.split(quat_key, 1)
    single_quat = rand_quat(quat_key, dtype)
    quaternion = Quaternion(jnp.tile(single_quat.vec[0], (N, 1)))

    body = rigid_body.RigidBody(R, quaternion)

    base_site = jnp.array(
        [1.0, 0.0, 0.0]
    )
    stack_site = jnp.array(
        [0.5, 0.0, 0.0]
    )
    bb_site = jnp.array(
        [-1.0, 0.0, 0.0]
    )
    bonded_neighbors = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5]
    ]

    energy_fn = static_energy_fn_factory(displacement,
                                         bb_site=bb_site,
                                         stack_site=stack_site,
                                         base_site=base_site,
                                         neighbors=bonded_neighbors)


    # Simulate with the energy function via Nose-Hoover

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    # step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    trajectory = list()

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)

      trajectory.append(state.position)

    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)


    # Add excluded volume and stacking
