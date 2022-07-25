import numpy as onp
import pdb
import toml
from functools import partial

from jax import jit
from jax import vmap
from jax import random
from jax import lax

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


PARAMS = toml.load("config.toml")
PARAMS = PARAMS['default'] # FIXME: Simple for now, but eventually want to override with other namespaces and handle default correctly

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
f3_base = partial(f3, r_star=PARAMS["dr_star_base"], r_c=PARAMS["dr_star_base"] + tmp_r_c_diff,
                  eps=PARAMS["eps_exc"], sigma=PARAMS["sigma_base"],
                  b=tmp_b)
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

def static_energy_fn_factory(displacement_fn, back_site, stack_site, base_site,
                             neighbors):

    d = space.map_bond(partial(displacement_fn))
    nbs_i = neighbors[:, 0]
    nbs_j = neighbors[:, 1]

    def energy_fn(body: RigidBody, **kwargs) -> float:
        Q = body.orientation
        back_sites = body.center + rigid_body.quaternion_rotate(Q, back_site) # (N, 3)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)
        base_sites = body.center + rigid_body.quaternion_rotate(Q, base_site)

        # FIXME: flatten, make
        # Note: I believe we don't have to flatten. In Sam's original code, R_sites contained *all* N*3 interaction sites

        # FIXME: for neighbors, this will change
        # d = space.map_product(partial(displacement_fn, **kwargs))

        # FENE
        dr_back = d(back_sites[nbs_i], back_sites[nbs_j]) # N x N x 3
        r_back = jnp.linalg.norm(dr_back, axis=1)
        fene = v_fene(r_back)

        # Excluded volume (bonded)
        dr_base = d(base_sites[nbs_i], base_sites[nbs_j])
        dr_back_base = d(back_sites[nbs_i], base_sites[nbs_j])
        dr_base_back = d(base_sites[nbs_i], back_sites[nbs_j])
        exc_vol = exc_vol_bonded(dr_base, dr_back_base, dr_base_back)

        # dr_stack = d(stack_sites[nbs_i], stack_sites[nbs_j])
        # stacking(dr_stack, Q)

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
    back_site = jnp.array(
        [-1.0, 0.0, 0.0]
    )
    bonded_neighbors = onp.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4]
    ])

    energy_fn = static_energy_fn_factory(displacement,
                                         back_site=back_site,
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
