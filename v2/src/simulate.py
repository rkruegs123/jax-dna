import pdb
import jax
from jax import jit
import jax.numpy as jnp
from jax import random
from jax.tree_util import Partial
from tqdm import tqdm

from jax_md.rigid_body import RigidBody
from jax_md import space, util

from utils import nucleotide_mass, get_kt, moment_of_inertia, get_one_hot, DEFAULT_TEMP
from utils import base_site, stack_site, back_site
import langevin
from energy import factory
from loader import get_params
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo


f64 = util.f64

def run_single_langevin(top_path, conf_path, n_steps, key, T=DEFAULT_TEMP, dt=5e-3):
    mass = RigidBody(center=jnp.array([nucleotide_mass]),
                     orientation=jnp.array([moment_of_inertia]))
    gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]),
                      orientation=jnp.array([DEFAULT_TEMP/7.5]))
    params = get_params.get_default_params(t=T, no_smoothing=False)

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                             back_site, stack_site, base_site,
                                             top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq, params=params)

    init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)


    state = init_fn(key, body, mass=mass, seq=seq, params=params)

    trajectory = [state.position]
    energies = [energy_fn(state.position)]
    for i in tqdm(range(n_steps)):
        state = step_fn(state, seq=seq, params=params)

        if i % 1000 == 0:
            energies.append(energy_fn(state.position))
            trajectory.append(state.position)

    print("Finished Simulation")
    return trajectory, energies



if __name__ == "__main__":
    import time

    top_path = "data/simple-helix/generated.top"
    conf_path = "data/simple-helix/start.conf"
    key = random.PRNGKey(0)

    start = time.time()
    run_single_langevin(top_path, conf_path, n_steps=1000, key=key)
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
