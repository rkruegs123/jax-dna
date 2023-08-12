import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import functools

import jax.numpy as jnp
from jax import jit, vmap, random, lax
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1 import model

from jax.config import config
config.update("jax_enable_x64", True)



displacement_fn, shift_fn = space.free()
dt = 5e-3
t_kelvin = utils.DEFAULT_TEMP
kT = utils.get_kt(t_kelvin)
gamma = rigid_body.RigidBody(
    center=jnp.array([kT/2.5], dtype=jnp.float64),
    orientation=jnp.array([kT/7.5], dtype=jnp.float64))
mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                            orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))
params = deepcopy(model.EMPTY_BASE_PARAMS)
em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)


def sim_for_loop(conf_info, top_info, n_steps, key, save_every=1):
    init_body = conf_info.get_states()[0]
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                         bonded_nbrs=top_info.bonded_nbrs,
                         unbonded_nbrs=top_info.unbonded_nbrs.T)


    trajectory = list()
    state = deepcopy(init_state)
    for i in tqdm(range(n_steps), colour="green"):
        state = step_fn(state,
                        seq=seq_oh,
                        bonded_nbrs=top_info.bonded_nbrs,
                        unbonded_nbrs=top_info.unbonded_nbrs.T)

        if i % save_every == 0:
            trajectory.append(state.position)
    return trajectory

def sim_scan(conf_info, top_info, n_steps, key):
    init_body = conf_info.get_states()[0]
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                         bonded_nbrs=top_info.bonded_nbrs,
                         unbonded_nbrs=top_info.unbonded_nbrs.T)

    @jit
    def scan_fn(state, step):
        state = step_fn(state,
                        seq=seq_oh,
                        bonded_nbrs=top_info.bonded_nbrs,
                        unbonded_nbrs=top_info.unbonded_nbrs.T)
        return state, state.position

    fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_steps))
    return traj

def sim_nested_scan(conf_info, top_info, n_inner_steps, n_outer_steps, key, save_every=1):
    n_steps = n_inner_steps, n_outer_steps

    init_body = conf_info.get_states()[0]
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                         bonded_nbrs=top_info.bonded_nbrs,
                         unbonded_nbrs=top_info.unbonded_nbrs.T)

    step_fn = functools.partial(step_fn, seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=top_info.unbonded_nbrs.T)
    step_fn = jit(step_fn)

    fori_step_fn = lambda t, state: step_fn(state)
    fori_step_fn = jit(fori_step_fn)

    @jit
    def scan_fn(state, step):
        state = lax.fori_loop(0, n_inner_steps, fori_step_fn, state)
        return state, state.position

    fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_outer_steps))
    return traj

def sim_nested_for_scan(conf_info, top_info, n_inner_steps, n_outer_steps, key):
    n_steps = n_inner_steps, n_outer_steps

    init_body = conf_info.get_states()[0]
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
    init_state = init_fn(key, init_body, mass=mass, seq=seq_oh,
                         bonded_nbrs=top_info.bonded_nbrs,
                         unbonded_nbrs=top_info.unbonded_nbrs.T)

    step_fn = functools.partial(step_fn, seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=top_info.unbonded_nbrs.T)
    step_fn = jit(step_fn)

    fori_step_fn = lambda t, state: step_fn(state)
    fori_step_fn = jit(fori_step_fn)

    state = deepcopy(init_state)
    trajectory = list()
    for i in tqdm(range(n_outer_steps)):
        state = lax.fori_loop(0, n_inner_steps, fori_step_fn, state)
        trajectory.append(state.position)

    return trajectory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate 202 bp duplex to eval. memory constraints")
    parser.add_argument('--method', type=str,
                        default="for-loop",
                        choices=["for-loop", "scan", "nested-scan", "nested-for-scan"],
                        help='Method for simulating')
    parser.add_argument('--n-steps', type=int,
                        help="Number of total steps")
    parser.add_argument('--n-points', type=int,
                        help="Number of total points in the final trajectory")
    args = vars(parser.parse_args())
    method = args['method']
    n_steps = args['n_steps']
    n_points = args['n_points']



    # Load the system
    sys_basedir = Path("data/sys-defs/persistence-length")
    top_path = sys_basedir / "init.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)

    conf_path = sys_basedir / "relaxed.dat"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )

    key = random.PRNGKey(0)

    sample_every = n_steps // n_points

    start = time.time()

    if method == "for-loop":
        trajectory = sim_for_loop(conf_info, top_info, n_steps, key, save_every=sample_every)
    elif method == "scan":
        trajectory = sim_scan(conf_info, top_info, n_steps, key)
    elif method == "nested-scan":
        trajectory = sim_nested_scan(conf_info, top_info, n_inner_steps=sample_every, n_outer_steps=n_points, key=key)
    elif method == "nested-for-scan":
        trajectory = sim_nested_for_scan(conf_info, top_info, n_inner_steps=sample_every, n_outer_steps=n_points, key=key)
    else:
        raise RuntimeError(f"Invalid method: {method}")

    end = time.time()
    print(f"Total time: {end - start}")
