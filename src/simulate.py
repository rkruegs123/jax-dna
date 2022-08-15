import numpy as onp
import pdb

from jax import jit
from jax import random

from jax.config import config as jax_config
import jax.numpy as jnp

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md import rigid_body
from jax_md.rigid_body import RigidBody, Quaternion

from potential import TEMP # FIXME: TEMP should really be an argument to the potentials... Should have getters that take in a temp
from utils import read_config, jax_traj_to_oxdna_traj
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import get_one_hot

from static_nbrs import static_energy_fn_factory
from dynamic_nbrs import dynamic_energy_fn_factory_fixed

from jax.config import config
config.update("jax_enable_x64", True)

FLAGS = jax_config.FLAGS
DYNAMICS_STEPS = 1000

f32 = util.f32
f64 = util.f64

DTYPE = [f32]
if FLAGS.jax_enable_x64:
    DTYPE += [f64]




def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors):
    static_energy_fn, _ = static_energy_fn_factory(displacement,
                                                   back_site=back_site,
                                                   stack_site=stack_site,
                                                   base_site=base_site,
                                                   neighbors=bonded_neighbors)
    dynamic_energy_fn, _ = dynamic_energy_fn_factory_fixed(
        displacement,
        back_site=back_site,
        stack_site=stack_site,
        base_site=base_site,
        neighbors=unbonded_neighbors
    )

    def energy_fn(body: RigidBody, seq: util.Array) -> float:
        return static_energy_fn(body) + dynamic_energy_fn(body, seq)

    return energy_fn


if __name__ == "__main__":
    mass = rigid_body.RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))

    # conf_path = "data/polyA_10bp/equilibrated.dat"
    # top_path = "data/polyA_10bp/generated.top"


    conf_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/start.dat"
    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"

    body, box_size, n_strands, bonded_neighbors, unbonded_neighbors, seq = read_config(conf_path, top_path)
    seq = jnp.array(get_one_hot(seq), dtype=f64)
    n = body.center.shape[0]
    bonded_neighbors = onp.array(bonded_neighbors)
    unbonded_neighbors = onp.array(unbonded_neighbors)

    box_size = box_size[1]

    displacement, shift = space.periodic(box_size)
    key = random.PRNGKey(0)
    key, pos_key, quat_key = random.split(key, 3)
    dtype = f64

    base_site = jnp.array(
        [com_to_hb, 0.0, 0.0], dtype=f64
    )
    stack_site = jnp.array(
        [com_to_stacking, 0.0, 0.0], dtype=f64
    )
    back_site = jnp.array(
        [com_to_backbone, 0.0, 0.0], dtype=f64
    )

    energy_fn = energy_fn_factory(displacement,
                                  back_site, stack_site, base_site,
                                  bonded_neighbors, unbonded_neighbors)


    # Simulate with the energy function via Nose-Hoover
    kT = get_kt(t=TEMP) # 300 Kelvin = 0.1 kT
    dt = 5e-3

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass, seq=seq)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, seq=seq)

    trajectory = [state.position]


    for i in range(DYNAMICS_STEPS):
        state = step_fn(state, seq=seq)
        trajectory.append(state.position)

    pdb.set_trace()

    # E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    jax_traj_to_oxdna_traj(trajectory, box_size, every_n=10)

    print("done")
