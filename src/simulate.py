import numpy as onp
import pdb
from itertools import combinations

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

from static_nbrs import static_energy_fn_factory
from dynamic_nbrs import dynamic_energy_fn_factory_fixed


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
    static_energy_fn = static_energy_fn_factory(displacement,
                                                back_site=back_site,
                                                stack_site=stack_site,
                                                base_site=base_site,
                                                neighbors=bonded_neighbors)
    dynamic_energy_fn = dynamic_energy_fn_factory_fixed(
        displacement,
        back_site=back_site,
        stack_site=stack_site,
        base_site=base_site,
        neighbors=unbonded_neighbors
    )

    def energy_fn(body: RigidBody, **kwargs) -> float:
        return static_energy_fn(body) + dynamic_energy_fn(body)

    return energy_fn


if __name__ == "__main__":
    mass = rigid_body.RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))

    body, box_size = read_config("data/polyA_10bp/equilibrated.dat")

    # box_size = box_size[0]
    box_size = 30.0

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
    bonded_neighbors = [(i, i+1) for i in range(n - 1)] # first store as tuples because lists aren't hashable

    pdb.set_trace()
    unbonded_neighbors = set(combinations(range(n), 2)) # first, set to all neighbors
    unbonded_neighbors -= set(bonded_neighbors) # remove bonded neighbors
    unbonded_neighbors -= set([(i, i) for i in range(n)]) # remove identities (they shouldn't be in there)
    unbonded_neighbors = list(unbonded_neighbors)

    # convert to numpy arrays
    bonded_neighbors = onp.array(bonded_neighbors)
    unbonded_neighbors = onp.array(unbonded_neighbors)

    energy_fn = energy_fn_factory(displacement,
                                  back_site, stack_site, base_site,
                                  bonded_neighbors, unbonded_neighbors)


    # Simulate with the energy function via Nose-Hoover
    kT = get_kt(t=TEMP) # 300 Kelvin = 0.1 kT
    dt = 5e-3

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift, dt, kT)

    step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    trajectory = [state.position]

    for i in range(DYNAMICS_STEPS):
        state = step_fn(state)
        trajectory.append(state.position)

    pdb.set_trace()
    # FIXME: now turns nan around 139... why?? Energy of 138 is around -9, which is great.
    # FIXME: first, try smooth FENE first...

    # E_final = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT)

    jax_traj_to_oxdna_traj(trajectory, box_size, every_n=10)

    print("done")
