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

from potential import v_fene, exc_vol_bonded, stacking, TEMP
from utils import read_config, jax_traj_to_oxdna_traj
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import Q_to_back_base, Q_to_cross_prod, Q_to_base_normal

from static_nbrs import static_energy_fn_factory

FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 3000


f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]

if __name__ == "__main__":
    mass = rigid_body.RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))

    body, box_size = read_config("data/polyA_10bp/equilibrated.dat")

    box_size = box_size[0]

    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = DTYPE[0]

    N = body.center.shape[0]

    base_site = jnp.array(
        [com_to_hb, 0.0, 0.0]
    )
    stack_site = jnp.array(
        [com_to_stacking, 0.0, 0.0]
    )
    back_site = jnp.array(
        [com_to_backbone, 0.0, 0.0]
    )

    n = 10 # FIXME: redundant. Use `N` from above
    bonded_neighbors = onp.array(
        [[i, i+1] for i in range(n - 1)]
    )

    static_energy_fn = static_energy_fn_factory(displacement,
                                                back_site=back_site,
                                                stack_site=stack_site,
                                                base_site=base_site,
                                                neighbors=bonded_neighbors)


    # Simulate with the energy function via Nose-Hoover
    kT = get_kt(t=TEMP) # 300 Kelvin = 0.1 kT
    dt = 5e-3

    init_fn, step_fn = simulate.nvt_nose_hoover(static_energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass)
    E_initial = simulate.nvt_nose_hoover_invariant(static_energy_fn, state, kT)

    trajectory = [state.position]

    for i in range(DYNAMICS_STEPS):
        state = step_fn(state)
        trajectory.append(state.position)


    # E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    pdb.set_trace()

    jax_traj_to_oxdna_traj(trajectory, box_size, every_n=50)

    print("done")
