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
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2
from jax_dna import dna2, loss
from jax_dna.loss import pitch, pitch2

from jax.config import config
config.update("jax_enable_x64", True)




def run(args):
    run_name = args['run_name']
    offset = args['offset']
    hi = args['hi']
    lo = args['lo']
    interval = args['interval']
    sample_every = args['sample_every']
    checkpoint_every = args['checkpoint_every']
    n_trials = args['n_trials']
    n_eq_steps = args['n_eq_steps']

    assert((hi - lo) % interval == 0)
    assert(interval % sample_every == 0)
    lengths = onp.arange(lo, hi+1, interval)

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

    displacement_fn, shift_fn = space.free()

    sys_basedir = Path("data/templates/simple-helix-60bp")
    top_path = sys_basedir / "sys.top"
    conf_path = sys_basedir / "init.conf"

    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    n_bp = (seq_oh.shape[0] // 2)
    quartets = utils.get_all_quartets(n_bp)
    quartets = quartets[offset:-offset-1]

    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        quartets, displacement_fn, model2.com_to_hb)

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

    r_cutoff = 10.0
    dr_threshold = 0.2
    neighbors_idx = top_info.unbonded_nbrs.T

    def sim_fn(params, body, n_steps, key, gamma):
        em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)

        neighbors_idx = top_info.unbonded_nbrs.T
        init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=neighbors_idx)
        @jit
        def scan_fn(state, step):
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=neighbors_idx)
            return state, state.position
        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj

    eq_fn = lambda params, key: sim_fn(params, init_body, n_eq_steps, key, gamma)
    eq_fn = jit(eq_fn)


    target_pitch = 11.0
    def get_grad_abs(n_steps, checkpoint_every, key_seed):

        if checkpoint_every is None:
            scan = lax.scan
        else:
            scan = functools.partial(checkpoint.checkpoint_scan,
                                     checkpoint_every=checkpoint_every)

        @jit
        def loss_fn(params, ref_states, ref_energies, ref_avg_angles):

            em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

            # Compute the weights
            energy_fn = lambda body: em.energy_fn(body,
                                                  seq=seq_oh,
                                                  bonded_nbrs=top_info.bonded_nbrs,
                                                  unbonded_nbrs=top_info.unbonded_nbrs.T,
                                                  is_end=top_info.is_end)
            energy_fn = jit(energy_fn)
            new_energies = vmap(energy_fn)(ref_states)
            diffs = new_energies - ref_energies # element-wise subtraction
            boltzs = jnp.exp(-beta * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom

            # Compute the expected pitch
            expected_angle = jnp.dot(weights, ref_avg_angles)
            expected_pitch = 2*jnp.pi / expected_angle
            mse = (expected_pitch - target_pitch)**2
            rmse = jnp.sqrt(mse)

            # Compute effective sample size
            n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

            return rmse, (n_eff, expected_pitch, expected_angle)
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        grad_fn = jit(grad_fn)

        params = deepcopy(model2.EMPTY_BASE_PARAMS)
        default_base_params = model2.default_base_params_seq_avg
        # params["fene"] = default_base_params["fene"]
        params["stacking"] = default_base_params["stacking"]

        key = random.PRNGKey(key_seed)
        key, eq_key = random.split(key)
        eq_body, _ = eq_fn(params, eq_key)
        fin_pos, traj_states = sim_fn(params, eq_body, n_steps, key, gamma)
        n_traj_states = len(traj_states.center)

        em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T,
                                              is_end=top_info.is_end)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)

        n_quartets = quartets.shape[0]
        ref_avg_angles = list()
        for rs_idx in range(n_traj_states):
            body = traj_states[rs_idx]
            angles = pitch2.get_all_angles(body, quartets, displacement_fn, model2.com_to_hb, model1.com_to_backbone, 0.0)
            state_avg_angle = onp.mean(angles)
            ref_avg_angles.append(state_avg_angle)
        ref_avg_angles = onp.array(ref_avg_angles)

        start = time.time()
        (loss, aux), grads = grad_fn(params, traj_states, calc_energies, ref_avg_angles)
        end = time.time()
        first_grad_time = end - start

        grad_vals = onp.array([float(v) for v in grads['stacking'].values()])
        grad_vals_abs = onp.abs(grad_vals)
        mean_grad_abs = grad_vals_abs.mean()

        return mean_grad_abs


    with open(log_path, "a") as f:
        f.write(f"Checkpoint every: {checkpoint_every}\n")

    for sim_length in lengths:

        tot_times = list()
        mean_grad_abss = list()
        for i in range(n_trials):
            start = time.time()
            mean_grad_abs = get_grad_abs(sim_length, checkpoint_every, i)
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

    return


def get_parser():
    parser = argparse.ArgumentParser(description="Get gradient scaling of oxDNA2 in JAX-MD")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Sampling frequency for trajectories")
    parser.add_argument('--n-eq-steps', type=int, default=50000,
                        help="Number of equilibration steps")
    parser.add_argument('--offset', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--interval', type=int, default=50000,
                        help="Interval of sample-every's for plotting")
    parser.add_argument('--lo', type=int, default=50000,
                        help="Minimum number of steps")
    parser.add_argument('--hi', type=int, default=500000,
                        help="Maximum number of steps")

    parser.add_argument('--checkpoint-every', type=int, default=50,
                        help="Checkpoint frequency")

    parser.add_argument('--n-trials', type=int, default=10,
                        help="Number of trials per simulation length")



    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
