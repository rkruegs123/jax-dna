from jax.config import config
config.update('jax_platform_name', 'cpu')

import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import shlex
import argparse
import pandas as pd
import random
import seaborn as sns
import zipfile
import os

import optax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2
from jax_dna.dna1.oxdna_utils import rewrite_input_file

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')


checkpoint_every = 50
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))

compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))

# Compute the average persistence length

def run(args):
    # Load parameters
    device = args['device']
    if device == "cpu":
        backend = "CPU"
    else:
        raise RuntimeError(f"Invalid device: {device}")
    n_threads = args['n_threads']
    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    n_skipped_quartets = args['n_skipped_quartets']
    truncation = args['truncation']
    n_iters = args['n_iters']
    lr = args['lr']
    target_lp = args['target_lp']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    no_delete = args['no_delete']
    no_archive = args['no_archive']
    opt_keys = args['opt_keys']


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)


    loss_path = run_dir / "loss.txt"
    times_path = run_dir / "times.txt"
    grads_path = run_dir / "grads.txt"
    neff_path = run_dir / "neff.txt"
    lp_path = run_dir / "lp.txt"
    l0_avg_path = run_dir / "l0_avg.txt"
    resample_log_path = run_dir / "resample_log.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/persistence-length")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    # top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    quartets = utils.get_all_quartets(n_nucs_per_strand=seq_oh.shape[0] // 2)
    quartets = quartets[n_skipped_quartets:]
    quartets = quartets[:-n_skipped_quartets]
    base_site = jnp.array([model2.com_to_hb, 0.0, 0.0])

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
        oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
        recompile_end = time.time()

        with open(resample_log_path, "a") as f:
            f.write(f"- Recompiling took {recompile_end - recompile_start} seconds\n")

        sim_start = time.time()
        if device == "cpu":
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
                no_stdout_energy=0, backend=backend,
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="DNA2_nomesh"
            )

            if device == "cpu":
                procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        if device == "cpu":
            for p in procs:
                p.wait()

            for p in procs:
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
        else:
            raise RuntimeError(f"Invalid device: {device}")

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

        # FIXME: take out first row when loading them in
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
            is_end=top_info.is_end)
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



        # Compute the persistence lengths
        analyze_start = time.time()

        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(traj_states, quartets, base_site)
        mean_corr_curve = jnp.mean(unweighted_corr_curves, axis=0)
        mean_l0 = jnp.mean(unweighted_l0_avgs)
        # mean_Lp, _ = persistence_length.persistence_length_fit(mean_corr_curve, mean_l0)

        mean_Lp_truncated, offset = persistence_length.persistence_length_fit(mean_corr_curve[:truncation], mean_l0)
        mean_Lp_truncated *= utils.nm_per_oxdna_length

        compute_every = 10
        n_curves = unweighted_corr_curves.shape[0]
        all_inter_lps = list()
        all_inter_lps_truncated = list()
        for i in range(0, n_curves, compute_every):
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], mean_l0)
            all_inter_lps_truncated.append(inter_mean_Lp_truncated * utils.nm_per_oxdna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, mean_l0)
            all_inter_lps.append(inter_mean_Lp * utils.nm_per_oxdna_length)

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps)
        plt.ylabel("Lp")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(iter_dir / "running_avg.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated)
        plt.ylabel("Lp")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(iter_dir / "running_avg_truncated.png")
        plt.clf()

        plt.plot(mean_corr_curve)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(iter_dir / "full_corr_curve.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "full_log_corr_curve.png")
        plt.clf()

        fit_fn = lambda n: -n * mean_l0 / (mean_Lp_truncated/utils.nm_per_oxdna_length) + offset
        plt.plot(jnp.log(mean_corr_curve)[:truncation])
        neg_inverse_slope = (mean_Lp_truncated / utils.nm_per_oxdna_length) / mean_l0 # in nucleotides
        rounded_offset = onp.round(offset, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation)), linestyle='--', label=fit_str)

        plt.title(f"Log-Correlation Curve, Truncated.")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "log_corr_curve.png")
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

        # Record the loss
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        if not no_archive:
            zip_file(str(iter_dir / "output.dat"), str(iter_dir / "output.dat.zip"))
            shutil.rmtree(iter_dir / "output.dat")

        return traj_states, calc_energies, unweighted_corr_curves, unweighted_l0_avgs, iter_dir


    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs):
        em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T,
                                              is_end=top_info.is_end)
        energy_fn = jit(energy_fn)

        # new_energies = vmap(energy_fn)(ref_states)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        weighted_l0_avgs = vmap(lambda l0, w: l0 * w)(unweighted_l0_avgs, weights)
        expected_corr_curve = jnp.sum(weighted_corr_curves, axis=0)
        expected_l0_avg = jnp.sum(weighted_l0_avgs)
        expected_lp, expected_offset = persistence_length.persistence_length_fit(
            expected_corr_curve[:truncation],
            expected_l0_avg)
        expected_lp = expected_lp * utils.nm_per_oxdna_length

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        mse = (expected_lp - target_lp)**2
        rmse = jnp.sqrt(mse)

        return rmse, (n_eff, expected_lp, expected_corr_curve, expected_l0_avg, expected_offset)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    params = deepcopy(model2.EMPTY_BASE_PARAMS)
    for opt_key in opt_keys:
        params[opt_key] = deepcopy(model2.default_base_params_seq_avg[opt_key])

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)


    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_lps = list()
    all_l0s = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_lps = list()
    all_ref_l0s = list()
    all_ref_times = list()


    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_eff, curr_lp, expected_corr_curve, curr_l0_avg, curr_offset)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_l0s.append(curr_l0_avg)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()
            ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff, curr_lp, expected_corr_curve, curr_l0_avg, curr_offset)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_l0s.append(curr_l0_avg)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")
        with open(l0_avg_path, "a") as f:
            f.write(f"{curr_l0_avg}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")

        all_losses.append(loss)
        all_n_effs.append(n_eff)
        all_lps.append(curr_lp)
        all_l0s.append(curr_l0_avg)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        plt.plot(expected_corr_curve)
        plt.xlabel("Nuc. Index")
        plt.ylabel("Correlation")
        plt.savefig(img_dir / f"corr_iter{i}.png")
        plt.clf()

        log_corr_fn = lambda n: -n * curr_l0_avg / (curr_lp / utils.nm_per_oxdna_length) + curr_offset # oxDNA units
        plt.plot(jnp.log(expected_corr_curve))
        plt.plot(log_corr_fn(jnp.arange(expected_corr_curve.shape[0])), linestyle='--')
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.savefig(img_dir / f"log_corr_iter{i}.png")
        plt.clf()

        plt.plot(expected_corr_curve[:truncation])
        plt.xlabel("Nuc. Index")
        plt.ylabel("Correlation")
        plt.savefig(img_dir / f"corr_truncated_iter{i}.png")
        plt.clf()

        plt.plot(jnp.log(expected_corr_curve[:truncation]))
        plt.plot(log_corr_fn(jnp.arange(expected_corr_curve[:truncation].shape[0])), linestyle='--')
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.savefig(img_dir / f"log_corr_truncated_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_lps, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_lps, marker='o', label="Resample points", color="blue")
        plt.axhline(y=target_lp, linestyle='--', label="Target Lp", color='red')
        plt.legend()
        plt.ylabel("Expected Lp (nm)")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"lps_iter{i}.png")
        plt.clf()

def get_parser():

    parser = argparse.ArgumentParser(description="Optimize persistence length via standalone oxDNA package")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-skipped-quartets', type=int, default=5,
                        help="Number of quartets to skip on either side to account for fraying")
    parser.add_argument('--truncation', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--target-lp', type=float,
                        help="Target persistence length in nanometers")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--no-archive', action='store_true')

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
