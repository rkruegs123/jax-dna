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



if __name__ == "__main__":
    box_size = 20.0
    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = DTYPE[0]

    # Get random rigid body
    N = 5
    R = box_size * random.uniform(pos_key, (N, 3), dtype=dtype)
    quat_key = random.split(quat_key, N)
    quaternion = rand_quat(quat_key, dtype)

    body = rigid_body.RigidBody(R, quaternion)

    # Use a shape of one particle to create an energy function
    shape = rigid_body.point_union_shape(
      onp.array([[0.0, 0.0, 0.0]], f32),
      f32(1.0)
    )

    neighbor_fn, energy_fn = energy.soft_sphere_neighbor_list(displacement,
                                                              box_size)
    neighbor_fn, energy_fn = rigid_body.point_energy_neighbor_list(energy_fn,
                                                                   neighbor_fn,
                                                                   shape)

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    nbrs = neighbor_fn.allocate(body)
    state = init_fn(key, body, mass=shape.mass(), neighbor=nbrs)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, neighbor=nbrs)

    # trajectory = list()
    for i in range(DYNAMICS_STEPS):
      state = step_fn(state, neighbor=nbrs)
      nbrs = jit(neighbor_fn.update)(state.position, nbrs)
      # trajectory.append(state.position)
    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, neighbor=nbrs)


    pdb.set_trace()
    print("done")
