import pdb
from pathlib import Path
import argparse
import numpy as onp
import random
import time
import shutil
from copy import deepcopy
import subprocess
import pandas as pd
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
from tqdm import tqdm
import zipfile
import os
from scipy.stats import norm
import seaborn as sns

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap, tree_util
from jax_md import space
import optax
from jaxopt import OptaxSolver

from jax_dna.common import utils, trajectory, topology, checkpoint, center_configuration
# from jax_dna.loss import pitch, pitch2
from jax_dna.loss import propeller
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.dna2.oxdna_utils import recompile_oxdna
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2



checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


# https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
def compute_weighted_avg_and_var(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = jnp.average(values, weights=weights)
    variance = jnp.average((values-average)**2, weights=weights)
    return (average, variance)


# Negative log-likelihood function that uses additional metadata
def neg_log_likelihood(params, data, weights):
    mu, sigma = params
    n = data.shape[0]
    # Incorporating metadata in the log likelihood (just an illustrative example)
    weighted_data = data * weights
    # log_likelihood = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2) - jnp.sum((weighted_data - mu) ** 2) / (2 * sigma**2)

    term1 = -0.5 * n * jnp.log(2 * jnp.pi * sigma**2)
    term2_denom = 2 * sigma**2

    sqr_res_fn = lambda idx: weights[idx]*n * (data[idx] - mu)**2
    all_sqr_res = vmap(sqr_res_fn)(jnp.arange(n))
    term2_num = jnp.sum(all_sqr_res)

    log_likelihood = term1 - term2_num / term2_denom
    return -log_likelihood


@jit
def kl_divergence(true_mean, true_var, est_mean, est_var):
    return jnp.log(est_var / true_var) + (true_var**2 + (true_mean - est_mean)**2) / (2 * est_var**2) - 1/2


def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))

INF = 1e6
def relative_diff(init_val, fin_val, eps=1e-10):
    denom = jnp.where(init_val != 0, init_val, init_val + eps)
    return (fin_val - init_val) / denom

def run(args):
    # Load parameters

    n_threads = args['n_threads']
    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    offset = args['offset']

    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"

    n_iters = args['n_iters']
    lr = args['lr']
    optimizer_type = args['optimizer_type']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']

    target_mean = args['target_mean']
    target_var = args['target_var']
    target_std = onp.sqrt(target_var)

    opt_keys = args['opt_keys']

    no_delete = args['no_delete']
    no_archive = args['no_archive']

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

    obj_dir = run_dir / "obj"
    obj_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    neff_path = log_dir / "neff.txt"
    mean_path = log_dir / "mean.txt"
    std_path = log_dir / "std.txt"
    rel_diff_path = log_dir / "rel_diff.txt"
    angle_path = log_dir / "angle.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/simple-helix-60bp")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    # quartets = utils.get_all_quartets(n_nucs_per_strand=seq_oh.shape[0] // 2)
    # quartets = quartets[offset:-offset-1]
    all_bps = utils.get_all_bps(n_nucs_per_strand=seq_oh.shape[0] // 2)
    all_bps = all_bps[offset:-offset]
    compute_avg_ptwist, _ = propeller.get_propeller_loss_fn(all_bps)
    compute_all_ptwists = vmap(compute_avg_ptwist)

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    # Do the simulation
    def get_ref_states(params, i, seed, prev_basedir):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        recompile_start = time.time()
        recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
        recompile_end = time.time()

        with open(resample_log_path, "a") as f:
            f.write(f"- Recompiling took {recompile_end - recompile_start} seconds\n")

        sim_start = time.time()
        procs = list()

        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")

            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(100),
                equilibration_steps=n_eq_steps, dt=dt,
                no_stdout_energy=0, backend="CPU",
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="DNA2_nomesh"
            )

            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulation took {sim_end - sim_start} seconds\n")

        combine_cmd = "cat "
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {iter_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        if not no_delete:
            files_to_remove = ["output.dat"]
            for r in range(n_sims):
                repeat_dir = iter_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()

        # Analyze

        ## Load states from oxDNA simulation
        load_start = time.time()
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / "output.dat",
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        n_traj_states = len(traj_states)
        traj_states = utils.tree_stack(traj_states)
        load_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Loading took {load_end - load_start} seconds\n")

        ## Load the oxDNA energies

        energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
        energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T,
            is_end=top_info.is_end
        )
        energy_fn = jit(energy_fn)

        # Check energies

        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")


        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)


        # Compute the pitches
        analyze_start = time.time()

        ref_ptwists = compute_all_ptwists(traj_states)

        running_avg = onp.cumsum(ref_ptwists) / onp.arange(1, n_traj_states + 1)
        plt.plot(running_avg_pitches)
        plt.savefig(iter_dir / f"running_avg.png")
        plt.clf()

        plt.plot(running_avg_pitches[-int(n_traj_states // 2):])
        plt.savefig(iter_dir / f"running_avg_second_half.png")
        plt.clf()

        # Plot the energies
        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        # Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.clf()

        # Plot the gaussian against the target

        sns.histplot(ref_ptwists, label="Samples", color="red", stat="density")

        xs_target = onp.linspace(target_mean-4*target_std, target_mean+4*target_std, 1000)
        ys_target = norm.pdf(xs_target, target_mean, target_std)
        plt.plot(xs_target, ys_target, label=f"Target fit (mu={onp.round(float(target_mean), 3)}, sigma={onp.round(float(onp.sqrt(target_var)), 3)})")

        curr_mean = onp.mean(ref_ptwists)
        curr_var = onp.var(ref_ptwists)
        curr_std = onp.sqrt(curr_var)
        xs_curr = onp.linspace(curr_mean-4*curr_std, curr_mean+4*curr_std, 1000)
        ys_curr = norm.pdf(xs_curr, curr_mean, curr_std)
        plt.plot(xs_curr, ys_curr, label=f"Curr fit (mu={onp.round(float(curr_mean), 3)}, sigma={onp.round(float(onp.sqrt(curr_var)), 3)})")

        # Fit via jaxopt instead
        init_params = jnp.array([curr_mean, curr_std])
        optimizer = optax.adam(learning_rate=0.0001)
        solver = OptaxSolver(fun=neg_log_likelihood, opt=optimizer, maxiter=5000, implicit_diff=True)
        weights = jnp.array([1/n_traj_states for _ in range(n_traj_states)])
        opt_params = solver.run(init_params, data=ref_ptwists, weights=weights).params
        mu_opt, sigma_opt = opt_params

        xs_jaxopt = onp.linspace(mu_opt-4*sigma_opt, mu_opt+4*sigma_opt, 1000)
        ys_jaxopt = norm.pdf(xs_jaxopt, mu_opt, sigma_opt)
        plt.plot(xs_jaxopt, ys_jaxopt, label=f"Curr fit, JAXopt (mu={onp.round(float(mu_opt), 3)}, sigma={onp.round(float(onp.sqrt(sigma_opt)), 3)})")

        plt.legend()
        plt.savefig(iter_dir / "gaussian_fit.png")
        plt.clf()

        curr_kl_divergence = kl_divergence(target_mean, target_var, curr_mean, curr_var)

        # Record the loss
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")
            f.write(f"Prop. twist: {curr_mean} deg\n")
            f.write(f"Prop. twist, variance: {curr_var} deg\n")
            f.write(f"KL divergence: {curr_kl_divergence}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        if not no_archive:
            zip_file(str(iter_dir / "output.dat"), str(iter_dir / "output.dat.zip"))
            os.remove(str(iter_dir / "output.dat"))

        return traj_states, calc_energies, jnp.array(ref_ptwists), iter_dir

    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, ref_ptwists):

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
        expected_ptwist, expected_ptwist_var = compute_weighted_avg_and_var(ref_ptwists, weights)

        curr_kl_divergence = kl_divergence(target_mean, target_var, expected_ptwist, expected_ptwist_var)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return curr_kl_divergence, (n_eff, expected_ptwist, expected_ptwist_var)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    params = deepcopy(model2.EMPTY_BASE_PARAMS)
    for opt_key in opt_keys:
        params[opt_key] = deepcopy(model2.default_base_params_seq_avg[opt_key])

    init_params = deepcopy(params)
    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=lr)
    else:
        raise RuntimeError(f"Invalid optimizer: {optimizer_type}")
    opt_state = optimizer.init(params)

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, ref_ptwists, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    min_n_eff = int(n_ref_states * min_neff_factor)

    all_losses = list()
    all_means = list()
    all_vars = list()
    all_n_effs = list()

    all_ref_losses = list()
    all_ref_means = list()
    all_ref_vars = list()
    all_ref_times = list()

    num_resample_iters = 0
    plot_every = 10
    save_obj_every = args['save_obj_every']
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (loss, (n_eff, curr_mean, curr_var)), grads = grad_fn(params, ref_states, ref_energies, ref_ptwists)

        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_means.append(curr_mean)
            all_ref_vars.append(curr_var)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0

            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()

            ref_states, ref_energies, ref_ptwists, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff, curr_mean, curr_var)), grads = grad_fn(params, ref_states, ref_energies, ref_ptwists)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_means.append(curr_mean)
            all_ref_vars.append(curr_var)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(mean_path, "a") as f:
            f.write(f"{curr_mean}\n")
        with open(var_path, "a") as f:
            f.write(f"{curr_var}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        iter_params_str = f"\nIteration {i}:"
        for k, v in params.items():
            iter_params_str += f"\n- {k}"
            for vk, vv in v.items():
                iter_params_str += f"\n\t- {vk}: {vv}"
        with open(iter_params_path, "a") as f:
            f.write(iter_params_str)
        grads_str = f"\nIteration {i}:"
        for k, v in grads.items():
            grads_str += f"\n- {k}"
            for vk, vv in v.items():
                grads_str += f"\n\t- {vk}: {vv}"
        with open(grads_path, "a") as f:
            f.write(grads_str)
        rel_diffs = tree_util.tree_map(relative_diff, init_params, params)
        rel_diffs_str = f"\nIteration {i}:"
        for k, v in rel_diffs.items():
            rel_diffs_str += f"\n- {k}"
            for vk, vv in v.items():
                rel_diffs_str += f"\n\t- {vk}: {vv}"
        with open(rel_diff_path, "a") as f:
            f.write(rel_diffs_str)

        all_losses.append(loss)
        all_n_effs.append(n_eff)
        all_means.append(curr_mean)
        all_vars.append(curr_vars)

        if i % plot_every == 0 and i:

            plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")

            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_means, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_means, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Expected Prop. Twist")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"means_iter{i}.png")
            plt.clf()

        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)
            onp.save(obj_dir / f"ref_means_i{i}.npy", onp.array(all_ref_means), allow_pickle=False)
            onp.save(obj_dir / f"means_i{i}.npy", onp.array(all_means), allow_pickle=False)
            onp.save(obj_dir / f"ref_vars_i{i}.npy", onp.array(all_ref_vars), allow_pickle=False)
            onp.save(obj_dir / f"vars_i{i}.npy", onp.array(all_vars), allow_pickle=False)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_means.npy", onp.array(all_ref_means), allow_pickle=False)
    onp.save(obj_dir / f"fin_means.npy", onp.array(all_means), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_vars.npy", onp.array(all_ref_vars), allow_pickle=False)
    onp.save(obj_dir / f"fin_vars.npy", onp.array(all_vars), allow_pickle=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Optimize structural properties using differentiable trajectory reweighting")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps for sampling reference states per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--offset', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--optimizer-type', type=str,
                        default="adam",
                        choices=["adam", "rmsprop"])


    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--save-obj-every', type=int, default=50,
                        help="Frequency of saving numpy files")
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')

    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--no-archive', action='store_true')

    parser.add_argument('--target-mean', type=float, default=propeller.TARGET_PROPELLER_TWIST,
                        help="Target propeller twist in degrees")
    parser.add_argument('--target-var', type=float, default=1.0)


    parser.add_argument(
        '--opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["fene", "stacking"],
        help='Parameter keys to optimize'
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
