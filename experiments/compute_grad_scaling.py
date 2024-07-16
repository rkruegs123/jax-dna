import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import argparse
import numpy as onp
import os

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.loss import pitch
from jax_dna.dna2 import model
from jax_dna import dna2, loss
from jax_dna.loss import geometry, pitch, propeller

from jax.config import config
config.update("jax_enable_x64", True)




def run(args):
    run_name = args['run_name']
    n_skip_quartets = args['n_skip_quartets']
    hi = args['hi']
    interval = args['interval']
    sample_every = args['sample_every']
    checkpoint_every = args['checkpoint_every']
    n_trials = args['n_trials']
    small_system = args['small_system']
    use_neighbors = args['use_neighbors']
    n_eq_steps = args['n_eq_steps']

    lengths = onp.arange(interval, hi+1, interval) * sample_every


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    log_path = run_dir / "log.txt"
    length_path = run_dir / "length.txt"
    time_path = run_dir / "time.txt"
    mean_grad_abs_path = run_dir / "mean_grad_abs.txt"

    nvidia_smi_path = run_dir / "nvidia-smi.txt"
    os.system(f"nvidia-smi >> {nvidia_smi_path}")

    displacement_fn, shift_fn = space.free()

    if small_system:
        sys_basedir = Path("data/templates/simple-helix-12bp")
    else:
        sys_basedir = Path("data/templates/simple-helix-60bp")
    top_path = sys_basedir / "sys.top"
    conf_path = sys_basedir / "init.conf"


    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    n_bp = (seq_oh.shape[0] // 2)
    quartets = utils.get_all_quartets(n_bp)[n_skip_quartets:-n_skip_quartets]

    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        quartets, displacement_fn, model.com_to_hb)

    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    init_body = centered_conf_info.get_states()[0]
    box_size = conf_info.box_size


    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    # gamma_scale = 2500
    gamma_scale = 1
    gamma = rigid_body.RigidBody(
        center=gamma.center * gamma_scale,
        orientation=gamma.orientation * gamma_scale)
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint.checkpoint_scan,
                                 checkpoint_every=checkpoint_every)


    def body_metadata_fn(body):
        mean_pitch = compute_avg_pitch(body)
        return mean_pitch


    @jit
    def body_loss_fn(body):
        loss = pitch_loss_fn(body)
        return loss, body_metadata_fn(body)

    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses, all_metadata = vmap(body_loss_fn)(states_to_eval)
        return losses.mean(), all_metadata

    r_cutoff = 10.0
    dr_threshold = 0.2
    neighbor_fn = top_info.get_neighbor_list_fn(
        displacement_fn, conf_info.box_size, r_cutoff, dr_threshold)
    neighbor_fn = jit(neighbor_fn)
    neighbors = neighbor_fn.allocate(init_body.center) # We use the COMs

    def sim_fn(params, body, n_steps, key, gamma):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)

        if use_neighbors:

            @jit
            def scan_fn(in_state, step):
                state, neighbors = in_state
                state = step_fn(state,
                                seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=neighbors.idx)
                neighbors = neighbors.update(state.position.center)
                return (state, neighbors), state.position

            init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                                 unbonded_nbrs=neighbors.idx)

            (fin_state, _), traj = scan(scan_fn, (init_state, neighbors), jnp.arange(n_steps))
        else:
            neighbors_idx = top_info.unbonded_nbrs.T
            init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                                 bonded_nbrs=top_info.bonded_nbrs,
                                 unbonded_nbrs=neighbors_idx)
            @jit
            def scan_fn(state, step):
                neighbors_idx = top_info.unbonded_nbrs.T
                state = step_fn(state,
                                seq=seq_oh,
                                bonded_nbrs=top_info.bonded_nbrs,
                                unbonded_nbrs=neighbors_idx)
                return state, state.position
            fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj

    eq_fn = lambda params, key: sim_fn(params, init_body, n_eq_steps, key, gamma)
    eq_fn = jit(eq_fn)


    target_pitch = pitch.TARGET_AVG_PITCH
    def get_grad_abs(n_steps, checkpoint_every, key_seed):

        if checkpoint_every is None:
            scan = lax.scan
        else:
            scan = functools.partial(checkpoint.checkpoint_scan,
                                     checkpoint_every=checkpoint_every)

        @jit
        def loss_fn(params, eq_body, key):
            fin_pos, traj = sim_fn(params, eq_body, n_steps, key, gamma)
            states_to_eval = traj[::sample_every]
            pitches = vmap(compute_avg_pitch)(states_to_eval)
            avg_pitch = pitches.mean()
            rmse = jnp.sqrt((avg_pitch - target_pitch)**2)
            return rmse, traj
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        grad_fn = jit(grad_fn)

        params = deepcopy(model.EMPTY_BASE_PARAMS)
        default_base_params = model.default_base_params_seq_avg
        # params["fene"] = default_base_params["fene"]
        params["stacking"] = default_base_params["stacking"]

        key = random.PRNGKey(key_seed)
        key, eq_key = random.split(key)
        eq_body = eq_fn(params, eq_key)
        try:
            start = time.time()
            (loss, traj), grads = grad_fn(params, eq_body, key)
            end = time.time()
            first_grad_time = end - start

            grad_vals = onp.array([float(v) for v in grads['stacking'].values()])
            grad_vals_abs = onp.abs(grad_vals)
            mean_grad_abs = grad_vals_abs.mean()

            failed = False
        except Exception as e:
            print(e)
            print("failed")

            mean_grad_abs = -1

            failed = True
        return failed, mean_grad_abs


    with open(log_path, "a") as f:
        f.write(f"Checkpoint every: {checkpoint_every}\n")

    for sim_length in lengths:

        tot_times = list()
        mean_grad_abss = list()
        for i in range(n_trials):
            start = time.time()
            failed, mean_grad_abs = get_grad_abs(sim_length, checkpoint_every, i)
            end = time.time()
            tot_time = end - start

            tot_times.append(tot_time)
            mean_grad_abss.append(mean_grad_abs)

        with open(log_path, "a") as f:
            f.write(f"- # steps: {sim_length}\n")
            f.write(f"\t- 1st grad time (mean): {onp.mean(tot_times)}\n")
            f.write(f"\t- 1st grad time (var): {onp.var(tot_times)}\n")
            f.write(f"\t- Mean grad abs. (mean): {onp.mean(mean_grad_abss)}\n")
            f.write(f"\t- Mean grad abs. (var): {onp.var(mean_grad_abss)}\n")

        with open(length_path, "a") as f:
            f.write(f"{sim_length}\n")
        with open(time_path, "a") as f:
            f.write(f"{onp.mean(tot_times)}\n")
        with open(mean_grad_abs_path, "a") as f:
            f.write(f"{onp.mean(mean_grad_abss)}\n")

        if failed:
            break

    return


def get_parser():
    parser = argparse.ArgumentParser(description="Get gradient scaling of oxDNA2 in JAX-MD")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Sampling frequency for trajectories")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-skip-quartets', type=int, default=5,
                        help="Number of quartets to skip on either end of the duplex")
    parser.add_argument('--interval', type=int, default=5,
                        help="Interval of sample-every's for plotting")
    parser.add_argument('--hi', type=int, default=100,
                        help="Maximum multiplier of sample_every for binary search")

    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help="Checkpoint frequency")

    parser.add_argument('--n-trials', type=int, default=10,
                        help="Number of trials per simulation length")

    parser.add_argument('--small-system', action='store_true',
                        help="If set, uses a 12 bp system instead of a 60 bp system")
    parser.add_argument('--use-neighbors', action='store_true',
                        help="If set, will use neighbor lists")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
