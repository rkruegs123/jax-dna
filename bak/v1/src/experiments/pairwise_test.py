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

    shape = rigid_body.point_union_shape(
      onp.array([[0.0, 0.0, 0.0]], f32),
      f32(1.0)
    ) # just to get the mass from

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

    # FIXME: what should our metric return?
    # Maybe should just do all things here. Could be very inefficient, e.g .if vectorizing f(displacmenet_or_metric) is more efficient than vectorizing the actual function.
    # However, maybe best to just get it right first with appropriating masking and neighboring
    # Also, would be great if we could just return an ARRAY of metrics. Then we could do all 8 variables at once!
    def rb_metric(rb1, rb2): # analogous to displacement
        return rb1.center[0] - rb2.center[0]
        # other_val = 2.0
        # return jnp.array([other_val, rb1.center[0] - rb2.center[0]])

    def soft_func(metr_val): # analogous to soft_sphere
        pdb.set_trace()
        return metr_val[0] / 2.0 + 3.0

    energy_fn = smap.pair(soft_func, rb_metric) # analogous to soft_sphere_pair

    # energy_fn : RigidBody (actually a system of many RigidBody's) -> energy

    kT = 1e-3
    dt = 5e-4

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    # step_fn = jit(step_fn)

    state = init_fn(key, body, mass=shape.mass())
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    trajectory = list()

    for i in range(DYNAMICS_STEPS):
      state = step_fn(state)

      trajectory.append(state.position)

    E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    pdb.set_trace()

    remapped = list()
    for pt in trajectory:
      remapped.append(vmap(rigid_body.transform, (0, None))(pt, shape))

    remapped = jnp.array(remapped)
    remapped.shape
    remapped = remapped.reshape(remapped.shape[0], -1, 3) # squash the particles in on each other

    pdb.set_trace()

    print("done")
