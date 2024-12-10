import pdb
from pathlib import Path
from copy import deepcopy
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import pandas as pd
import shutil
import seaborn as sns
import argparse
import functools

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, lax, tree_util, random
from jax_md import space, rigid_body, simulate

from jax_dna.loss import rmse
from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.rna2 import model, oxrna_utils
from jax_dna.rna2.load_params import read_seq_specific, DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS
import jax_dna.input.trajectory as jdt

jax.config.update("jax_enable_x64", True)


checkpoint_every = 1
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args):
    # Load parameters
    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    assert(n_ref_states >= checkpoint_every)
    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    use_nbrs = args['use_nbrs']

    use_symm_coax = args['use_symm_coax']

    seq_avg_opt_keys = args['seq_avg_opt_keys']
    opt_seq_dep_stacking = args['opt_seq_dep_stacking']

    full_system = args['full_system']

    # t_kelvin = utils.DEFAULT_TEMP
    t_kelvin = 293.15

    ss_hb_weights, ss_stack_weights, ss_cross_weights = read_seq_specific(DEFAULT_BASE_PARAMS)
    # ss_hb_weights = utils.HB_WEIGHTS_SA
    # ss_stack_weights = utils.STACK_WEIGHTS_SA
    salt_conc = 1.0


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    times_path = log_dir / "times.txt"
    params_per_iter_path = log_dir / "params_per_iter.txt"
    pct_change_path = log_dir / "pct_change.txt"
    grads_path = log_dir / "grads.txt"
    neff_path = log_dir / "neff.txt"
    rmse_path = log_dir / "rmse.txt"
    resample_log_path = log_dir / "resample_log.txt"

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    if full_system:
        sys_basedir = Path("data/templates/5ht-tc-rmse-rna")
    else:
        sys_basedir = Path("data/templates/2ht-tc-rmse-rna")
    input_template_path = sys_basedir / "input"
    ss_path = sys_basedir / "rna_sequence_dependent_parameters.txt"

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)

    target_path = sys_basedir / "target.conf"
    target_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=target_path,
        reverse_direction=False
    )
    target_state = target_info.get_states()[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    default_neighbors = None
    if use_nbrs:
        r_cutoff = 10.0
        dr_threshold = 0.2
        neighbor_fn = top_info.get_neighbor_list_fn(
            displacement_fn, box_size, r_cutoff, dr_threshold)
        # Note that we only allocate once
        neighbors = neighbor_fn.allocate(target_state.center) # We use the COMs.
        default_neighbors = deepcopy(neighbors)

    dt = 3e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    def sim_fn(params, body, key, curr_stack_weights):

        if use_nbrs:
            neighbors_idx = default_neighbors.idx
        else:
            neighbors_idx = top_info.unbonded_nbrs.T

        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body, neighbors_idx: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=neighbors_idx)
        energy_fn = jit(energy_fn)

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, neighbors_idx=neighbors_idx)

        @jit
        def fori_step_fn(t, carry):
            state, neighbors = carry
            if use_nbrs:
                neighbors = neighbors.update(state.position.center)
                neighbors_idx = neighbors.idx
            else:
                neighbors_idx = top_info.unbonded_nbrs.T
            state = step_fn(state, neighbors_idx=neighbors_idx)
            return (state, neighbors)

        @jit
        def scan_fn(carry, step):
            (state, neighbors) = lax.fori_loop(0, sample_every, fori_step_fn, carry)
            return (state, neighbors), state.position

        (eq_state, eq_neighbors) = lax.fori_loop(0, n_eq_steps, fori_step_fn, (init_state, default_neighbors))
        (fin_state, _), traj = scan(scan_fn, (eq_state, eq_neighbors), jnp.arange(n_ref_states_per_sim))

        return traj

    def get_ref_states(params, i, iter_key, init_body):

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)


        if "stacking" in params["seq_dep"]:
            curr_stack_weights = params["seq_dep"]["stacking"]
        else:
            curr_stack_weights = ss_stack_weights

        iter_key, sim_key = random.split(iter_key)
        sim_keys = random.split(sim_key, n_sims)
        sim_start = time.time()
        all_batch_ref_states = vmap(sim_fn, (None, None, 0, None))(params, init_body, sim_keys, curr_stack_weights)
        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulating took {sim_end - sim_start} seconds\n")

        combined_center = all_batch_ref_states.center.reshape(-1, top_info.n, 3)
        combined_quat_vec = all_batch_ref_states.orientation.vec.reshape(-1, top_info.n, 4)

        traj_states = rigid_body.RigidBody(
            center=combined_center,
            orientation=rigid_body.Quaternion(combined_quat_vec))

        # n_traj_states = len(ref_states)
        n_traj_states = traj_states.center.shape[0]


        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        ## Calculate energies
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")

        ## Calculate RMSDs
        RMSDs, RMSFs = rmse.compute_rmses(traj_states, target_state, top_info)

        ## Plot logging information
        analyze_start = time.time()

        sns.histplot(RMSDs)
        plt.savefig(iter_dir / f"rmsd_hist.png")
        plt.clf()

        running_avg = onp.cumsum(RMSDs) / onp.arange(1, (n_ref_states)+1)
        plt.plot(running_avg)
        plt.savefig(iter_dir / "running_avg_rmsd.png")
        plt.clf()

        last_half = int((n_ref_states) // 2)
        plt.plot(running_avg[-last_half:])
        plt.savefig(iter_dir / "running_avg_rmsd_second_half.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()


        mean_rmsd = onp.mean(RMSDs)
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Mean RMSD: {mean_rmsd}\n")
            f.write(f"# Traj. States: {n_traj_states}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")


        return traj_states, calc_energies, jnp.array(RMSDs), iter_dir


    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, unweighted_rmses):
        if "stacking" in params["seq_dep"]:
            curr_stack_weights = params["seq_dep"]["stacking"]
        else:
            curr_stack_weights = ss_stack_weights
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            # ss_stack_weights=ss_stack_weights)
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, new_energies = scan(energy_scan_fn, None, ref_states)
        diffs = new_energies - ref_energies
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        expected_rmse = jnp.dot(weights, unweighted_rmses)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return expected_rmse, n_eff
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    seq_avg_params = deepcopy(EMPTY_BASE_PARAMS)
    for opt_key in seq_avg_opt_keys:
        seq_avg_params[opt_key] = deepcopy(DEFAULT_BASE_PARAMS[opt_key])
    params = {"seq_avg": seq_avg_params, "seq_dep": dict()}
    if opt_seq_dep_stacking:
        params["seq_dep"]["stacking"] = jnp.array(ss_stack_weights)

    if use_symm_coax:
        assert("coaxial_stacking" in seq_avg_opt_keys)

        params["seq_avg"]["coaxial_stacking"]["theta0_coax_1_bonus"] = 0.35

    init_params = deepcopy(params)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_rmses = list()
    all_n_effs = list()
    all_ref_rmses = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    key = random.PRNGKey(0)
    init_body = conf_info.get_states()[0]
    start = time.time()
    ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=0, iter_key=key, init_body=init_body)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)
        num_resample_iters += 1

        if i == 0:
            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            key, split = random.split(key)
            start = time.time()
            ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=i, iter_key=split, init_body=ref_states[-1])
            end = time.time()
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)

            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        iter_end = time.time()

        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(rmse_path, "a") as f:
            f.write(f"{curr_rmse}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        with open(params_per_iter_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")
        with open(pct_change_path, "a") as f:
            pct_changes = tree_util.tree_map(lambda x, y: (y - x) / jnp.abs(x) * 100, init_params, params)
            f.write(f"{pprint.pformat(pct_changes)}\n")

        all_n_effs.append(n_eff)
        all_rmses.append(curr_rmse)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


        plt.plot(onp.arange(i+1), all_rmses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_rmses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"rmses_iter{i}.png")
        plt.clf()

def get_parser():

    parser = argparse.ArgumentParser(description="Optimize persistence length via standalone oxDNA package")

    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--full-system', action='store_true')
    parser.add_argument('--use-nbrs', action='store_true')

    parser.add_argument(
        '--seq-avg-opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "cross_stacking", "coaxial_stacking"],
        help='Parameter keys to optimize'
    )
    parser.add_argument('--opt-seq-dep-stacking', action='store_true')

    parser.add_argument('--use-symm-coax', action='store_true')


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
