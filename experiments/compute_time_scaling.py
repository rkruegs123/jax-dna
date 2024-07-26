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
    assert(checkpoint_every is not None)
    small_system = args['small_system']
    use_neighbors = args['use_neighbors']
    include_no_ckpt = args['include_no_ckpt']

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

    sim_time_ckpt_path = run_dir / "sim_time_ckpt.txt"
    grad_time_ckpt_path = run_dir / "grad_time_ckpt.txt"
    sim_time_no_ckpt_path = run_dir / "sim_time_no_ckpt.txt"
    grad_time_no_ckpt_path = run_dir / "grad_time_no_ckpt.txt"
    sim_length_path = run_dir / "sim_length.txt"

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
    init_body = conf_info.get_states()[0]
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

    scan_ckpt = functools.partial(checkpoint.checkpoint_scan,
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

    def get_sim_fn(n_steps, gamma, checkpoint):
        if checkpoint:
            sim_scan = scan_ckpt
        else:
            sim_scan = lax.scan

        r_cutoff = 10.0
        dr_threshold = 0.2
        neighbor_fn = top_info.get_neighbor_list_fn(
            displacement_fn, conf_info.box_size, r_cutoff, dr_threshold)
        neighbor_fn = jit(neighbor_fn)
        neighbors = neighbor_fn.allocate(init_body.center) # We use the COMs

        def sim_fn(body, params, key):
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

                (fin_state, _), traj = sim_scan(scan_fn, (init_state, neighbors), jnp.arange(n_steps))
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
                fin_state, traj = sim_scan(scan_fn, init_state, jnp.arange(n_steps))
            return fin_state.position, traj
        return sim_fn
            
    def get_time(n_steps, checkpoint_every):

        sim_fn_no_ckpt = get_sim_fn(n_steps, gamma, checkpoint=False)
        sim_fn_no_ckpt = jit(sim_fn_no_ckpt)
        sim_fn_ckpt = get_sim_fn(n_steps, gamma, checkpoint=True)
        sim_fn_ckpt = jit(sim_fn_ckpt)

        @jit
        def loss_fn_no_ckpt(params, eq_body, key):
            fin_pos, traj = sim_fn_no_ckpt(eq_body, params, key)
            loss, metadata = traj_loss_fn(traj)
            return loss, traj
        grad_fn_no_ckpt = value_and_grad(loss_fn_no_ckpt, has_aux=True)
        grad_fn_no_ckpt = jit(grad_fn_no_ckpt)

        @jit
        def loss_fn_ckpt(params, eq_body, key):
            fin_pos, traj = sim_fn_ckpt(eq_body, params, key)
            loss, metadata = traj_loss_fn(traj)
            return loss, traj
        grad_fn_ckpt = value_and_grad(loss_fn_ckpt, has_aux=True)
        grad_fn_ckpt = jit(grad_fn_ckpt)

        params = deepcopy(model.EMPTY_BASE_PARAMS)
        default_base_params = model.default_base_params_seq_avg
        # params["fene"] = default_base_params["fene"]
        params["stacking"] = default_base_params["stacking"]

        key = random.PRNGKey(0)

        # No checkpoint
        if include_no_ckpt:
            start = time.time()
            loss, traj = loss_fn_no_ckpt(params, init_body, key)
            end = time.time()
            first_sim_time_no_ckpt = end - start

            start = time.time()
            loss, traj = loss_fn_no_ckpt(params, init_body, key)
            end = time.time()
            second_sim_time_no_ckpt = end - start

            start = time.time()
            (loss, traj), grads = grad_fn_no_ckpt(params, init_body, key)
            end = time.time()
            first_grad_time_no_ckpt = end - start

            start = time.time()
            (loss, traj), grads = grad_fn_no_ckpt(params, init_body, key)
            end = time.time()
            second_grad_time_no_ckpt = end - start
        else:
            first_sim_time_no_ckpt = -1
            second_sim_time_no_ckpt = -1
            first_grad_time_no_ckpt = -1
            second_grad_time_no_ckpt = -1

        # Checkpoint

        start = time.time()
        loss, traj = loss_fn_ckpt(params, init_body, key)
        end = time.time()
        first_sim_time_ckpt = end - start

        start = time.time()
        loss, traj = loss_fn_ckpt(params, init_body, key)
        end = time.time()
        second_sim_time_ckpt = end - start

        start = time.time()
        (loss, traj), grads = grad_fn_ckpt(params, init_body, key)
        end = time.time()
        first_grad_time_ckpt = end - start

        start = time.time()
        (loss, traj), grads = grad_fn_ckpt(params, init_body, key)
        end = time.time()
        second_grad_time_ckpt = end - start

        return first_sim_time_no_ckpt, second_sim_time_no_ckpt, first_grad_time_no_ckpt, second_grad_time_no_ckpt, \
            first_sim_time_ckpt, second_sim_time_ckpt, first_grad_time_ckpt, second_grad_time_ckpt


    for sim_length in lengths:

        all_times = get_time(sim_length, checkpoint_every)

        sim_time_no_ckpt = all_times[1]
        grad_time_no_ckpt = all_times[3]
        sim_time_ckpt = all_times[5]
        grad_time_ckpt = all_times[7]

        with open(sim_length_path, "a") as f:
            f.write(f"{sim_length}\n")
        with open(sim_time_ckpt_path, "a") as f:
            f.write(f"{sim_time_ckpt}\n")
        with open(grad_time_ckpt_path, "a") as f:
            f.write(f"{grad_time_ckpt}\n")
        with open(sim_time_no_ckpt_path, "a") as f:
            f.write(f"{sim_time_no_ckpt}\n")
        with open(grad_time_no_ckpt_path, "a") as f:
            f.write(f"{grad_time_no_ckpt}\n")

    return


def get_parser():
    parser = argparse.ArgumentParser(description="Get gradient scaling of oxDNA2 in JAX-MD")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Sampling frequency for trajectories")
    parser.add_argument('--n-skip-quartets', type=int, default=5,
                        help="Number of quartets to skip on either end of the duplex")
    parser.add_argument('--interval', type=int, default=5,
                        help="Interval of sample-every's for plotting")
    parser.add_argument('--hi', type=int, default=100,
                        help="Maximum multiplier of sample_every for binary search")

    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help="Checkpoint frequency")

    parser.add_argument('--small-system', action='store_true',
                        help="If set, uses a 12 bp system instead of a 60 bp system")
    parser.add_argument('--use-neighbors', action='store_true',
                        help="If set, will use neighbor lists")
    parser.add_argument('--include-no-ckpt', action='store_true',
                        help="If set, will run an additional check with no checkpointing")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
