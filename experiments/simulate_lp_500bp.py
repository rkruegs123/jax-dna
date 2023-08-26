import pdb
from pathlib import Path
from tqdm import tqdm
import time
import numpy as onp
from copy import deepcopy
import functools
import matplotlib.pyplot as plt
import shutil

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, lax, tree_util, pmap
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory
from jax_dna.dna1 import model
from jax_dna.loss import persistence_length

from jax.config import config
config.update("jax_enable_x64", True)


def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

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



def sim_nested_for_scan(conf_info, top_info, n_inner_steps, n_outer_steps, key):

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

    sample_point = lambda s: lax.fori_loop(0, n_inner_steps, fori_step_fn, s)
    sample_point = jit(sample_point)

    state = deepcopy(init_state)
    traj = list()
    for i in tqdm(range(n_outer_steps)):
        # state = lax.fori_loop(0, n_inner_steps, fori_step_fn, state)
        state = sample_point(state)
        traj.append(state.position)

    return tree_stack(traj)




def sim_pmap_nested_scan(conf_info, top_info, sample_every, n_outer_steps,
                         key, batch_size, n_eq_steps):

    init_body = conf_info.get_states()[0]
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    def eq_fn(eq_key):
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(eq_key, init_body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)
        def fori_step_fn(t, state):
            return step_fn(state, seq=seq_oh,
                           bonded_nbrs=top_info.bonded_nbrs,
                           unbonded_nbrs=top_info.unbonded_nbrs.T)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    key, eq_key = random.split(key)
    eq_keys = random.split(eq_key, batch_size)
    eq_bodies = pmap(eq_fn)(eq_keys)


    def batch_fn(batch_key, eq_body):
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(batch_key, eq_body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        def fori_step_fn(t, state):
            return step_fn(state, seq=seq_oh,
                           bonded_nbrs=top_info.bonded_nbrs,
                           unbonded_nbrs=top_info.unbonded_nbrs.T)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(state, step):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            return state, state.position

        fin_state, traj = lax.scan(scan_fn, init_state, jnp.arange(n_outer_steps))
        return traj

    batch_keys = random.split(key, batch_size)
    batch_trajs = pmap(batch_fn)(batch_keys, eq_bodies)

    num_bases = batch_trajs.center.shape[2]
    assert(batch_trajs.center.shape[3] == 3)

    combined_center = batch_trajs.center.reshape(-1, num_bases, 3)
    combined_quat_vec = batch_trajs.orientation.vec.reshape(-1, num_bases, 4)

    combined_traj = rigid_body.RigidBody(
        center=combined_center,
        orientation=rigid_body.Quaternion(combined_quat_vec))

    return combined_traj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate 202 bp duplex to eval. memory constraints")
    parser.add_argument('--run-name', type=str,
                        help='Run name')
    parser.add_argument('--n-steps-per-batch', type=int,
                        help="Number of total steps per device")
    parser.add_argument('--n-eq-steps', type=int,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int,
                        help="Frequency of sampling states from trajectory")
    parser.add_argument('--running-avg-interval', type=int,
                        help="Interval of computed trajectory for computing a running average. Note that this interval is on a trajectory of length n_points, not n_steps")
    parser.add_argument('--min-running-avg-idx', type=int,
                        help="Minimum index of the final trajectory for plotting a running average")
    parser.add_argument('--batch-size', type=int, help="Batch size")
    parser.add_argument('--key-seed', type=int, default=0,
                        help="Integer seed for key")


    args = vars(parser.parse_args())
    n_steps_per_batch = args['n_steps_per_batch']
    sample_every = args['sample_every']
    assert(n_steps_per_batch % sample_every == 0)
    n_points_per_batch = n_steps_per_batch // sample_every
    batch_size = args['batch_size']
    n_devices = jax.local_device_count()
    assert(batch_size <= n_devices)
    run_name = args['run_name']
    running_avg_interval = args['running_avg_interval']
    min_running_avg_idx = args['min_running_avg_idx']
    n_eq_steps = args['n_eq_steps']
    key_seed = args['key_seed']



    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    output_path = run_dir / "output.txt"


    # Load the system
    sys_basedir = Path("data/sys-defs/persistence-length-500bp")
    top_path = sys_basedir / "init.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    shutil.copy(top_path, run_dir)

    conf_path = sys_basedir / "relaxed.dat"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )

    key = random.PRNGKey(key_seed)

    start = time.time()
    traj = sim_pmap_nested_scan(conf_info, top_info, sample_every=sample_every,
                                n_outer_steps=n_points_per_batch, key=key,
                                batch_size=batch_size, n_eq_steps=n_eq_steps)
    end = time.time()

    with open(output_path, "a") as f:
        f.write(f"Time to generate trajectory: {end - start} seconds\n")

    # Compute the average persistence length
    def get_all_quartets(n_nucs_per_strand):
        s1_nucs = list(range(n_nucs_per_strand))
        s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
        s2_nucs.reverse()

        bps = list(zip(s1_nucs, s2_nucs))
        n_bps = len(s1_nucs)
        all_quartets = list()
        for i in range(n_bps-1):
            bp1 = bps[i]
            bp2 = bps[i+1]
            all_quartets.append(bp1 + bp2)
        return jnp.array(all_quartets, dtype=jnp.int32)

    quartets = get_all_quartets(n_nucs_per_strand=traj[0].center.shape[0] // 2)
    # quartets = quartets[25:]
    # quartets = quartets[:-25]
    quartets = quartets[5:]
    quartets = quartets[:-5]


    all_curves = list()
    all_l0_avg = list()

    intermediate_lps = dict()
    assert(isinstance(traj, rigid_body.RigidBody))
    n_states = traj.center.shape[0]

    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])

    for i in tqdm(range(n_states)):
        body = traj[i]
        correlation_curve, l0_avg = persistence_length.get_correlation_curve(body, quartets, base_site)
        Lp = persistence_length.persistence_length_fit(correlation_curve, l0_avg)

        all_curves.append(correlation_curve)
        all_l0_avg.append(l0_avg)

        if i % running_avg_interval == 0 and i != 0:
            mean_correlation_curve = jnp.mean(jnp.array(all_curves), axis=0)
            mean_Lp = persistence_length.persistence_length_fit(mean_correlation_curve, jnp.mean(jnp.array(all_l0_avg)))
            intermediate_lps[i*sample_every] = mean_Lp * utils.nm_per_oxdna_length


    # Plot the running average
    plt.plot(intermediate_lps.keys(), intermediate_lps.values())
    plt.xlabel("Time")
    plt.ylabel("Lp (nm)")
    plt.title("Running Average")
    # plt.show()
    plt.savefig(run_dir / "running_avg.png")
    plt.clf()

    # Plot a running average with the first bit cut off
    plt.plot(list(intermediate_lps.keys())[min_running_avg_idx:], list(intermediate_lps.values())[min_running_avg_idx:])
    plt.xlabel("Time")
    plt.ylabel("Lp (nm)")
    plt.title("Running Average, Initial Truncation")
    # plt.show()
    plt.savefig(run_dir / "truncated_running_avg.png")
    plt.clf()


    # Plot the final correlation curve
    all_curves = jnp.array(all_curves)
    all_l0_avg = jnp.array(all_l0_avg)
    mean_correlation_curve = jnp.mean(all_curves, axis=0)
    mean_l0_avg = jnp.mean(all_l0_avg)

    plt.plot(mean_correlation_curve)
    plt.title("Final Correlation Curve")
    plt.savefig(run_dir / "final_corr_curve.png")
    # plt.show()
    plt.clf()

    Lp = persistence_length.persistence_length_fit(mean_correlation_curve, mean_l0_avg)
    with open(output_path, "a") as f:
        f.write(f"\nFinal persistence length: {Lp*utils.nm_per_oxdna_length}\n")

    # Write the trajectory to file
    traj_info = trajectory.TrajectoryInfo(
        top_info, read_from_states=True, states=traj, box_size=conf_info.box_size)
    traj_info.write(run_dir / "traj.dat", reverse=True)
