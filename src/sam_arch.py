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


# FIXME: need some initial positions from Megan
def fene_potential(dr_bb, eps=2.0, r0=0.7525, delt=0.25):
    pdb.set_trace()
    r = jnp.linalg.norm(dr_bb, axis=2)
    x = (r - r0)**2 / delt**2
    # Note: if `x` is too big, we will easily try to take the log of negatives, wihch will yield `nan`
    return -eps / 2.0 * jnp.log(1 - x)

def static_energy_fn_factory(displacement_fn, bb_site, stack_site, neighbors=None):

    def energy_fn(body: RigidBody, **kwargs) -> float:
        Q = body.orientation
        bb_sites = body.center + rigid_body.quaternion_rotate(Q, bb_site)
        stack_sites = body.center + rigid_body.quaternion_rotate(Q, stack_site)

        # FIXME: flatten, make
        # Note: I believe we don't have to flatten. In Sam's original code, R_sites contained *all* N*3 interaction sites

        # FIXME: for neighbors, this will change
        d = space.map_product(partial(displacement_fn, **kwargs))

        dr_bb = d(bb_sites, bb_sites) # N x N x 3
        fene = fene_potential(dr_bb)
        return jnp.sum(fene) / 2.0 # FIXME: placeholder

    return energy_fn


if __name__ == "__main__":


    # Next: commit, push, smooth versions for each f_i, then simple...


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

    stack_site = jnp.array(
        [0.5, 0.0, 0.0]
    )
    bb_site = jnp.array(
        [-1.0, 0.0, 0.0]
    )
    energy_fn = static_energy_fn_factory(displacement,
                                         bb_site=bb_site,
                                         stack_site=stack_site,
                                         neighbors=None)


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
