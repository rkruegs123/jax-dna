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

from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
from oxDNA_analysis_tools.deviations import deviations

import jax.numpy as jnp
from jax import jit, value_and_grad, vmap, tree_util
from jax_md import space
import optax

from jax_dna.common import utils, trajectory, topology, checkpoint, center_configuration
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.dna2.oxdna_utils import recompile_oxdna
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2
import jax_dna.input.trajectory as jdt



checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)





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

    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"

    n_iters = args['n_iters']
    lr = args['lr']
    optimizer_type = args['optimizer_type']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']

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
    rmse_path = log_dir / "rmse.txt"
    rel_diff_path = log_dir / "rel_diff.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/burns_natnano_2015")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False, allow_circle=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    target_path = sys_basedir / "unrelaxed.conf"

    conf_path = sys_basedir / "relaxed.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 3e-3
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
                save_interval=sample_every, seed=random.randrange(1000),
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

        traj_path = iter_dir / "output.dat"

        ## Compute RMSDs using OAT

        ti_ref, di_ref = describe(None, str(target_path))
        ti_trj, di_trj = describe(None, str(traj_path))

        ref_conf = get_confs(ti_ref, di_ref, 0, 1)[0]
        RMSDs, RMSFs = deviations(di_trj, ti_trj, ref_conf, indexes=[], ncpus=1)

        ## Load states from oxDNA simulation
        load_start = time.time()
        """
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=traj_path,
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        """
        traj_ = jdt.from_file(
            traj_path,
            [seq_oh.shape[0]],
            is_oxdna=False,
            n_processes=n_threads
        )
        traj_states = [ns.to_rigid_body() for ns in traj_.states]
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


        ## Plot logging information
        analyze_start = time.time()


        sns.histplot(RMSDs)
        plt.savefig(iter_dir / f"rmsd_hist.png")
        plt.close()

        running_avg = onp.cumsum(RMSDs) / onp.arange(1, (n_ref_states)+1)
        plt.plot(running_avg)
        plt.savefig(iter_dir / "running_avg_rmsd.png")
        plt.close()

        last_half = int((n_ref_states) // 2)
        plt.plot(running_avg[-last_half:])
        plt.savefig(iter_dir / "running_avg_rmsd_second_half.png")
        plt.close()


        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.close()

        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.close()

        # Record the loss
        mean_rmsd = onp.mean(RMSDs)
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")
            f.write(f"Mean RMSD: {mean_rmsd}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")


        if not no_archive:
            zip_file(str(iter_dir / "output.dat"), str(iter_dir / "output.dat.zip"))
            os.remove(str(iter_dir / "output.dat"))

        return traj_states, calc_energies, jnp.array(RMSDs), iter_dir

    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, unweighted_rmses):

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

        # Compute the expected rmse
        expected_rmse = jnp.dot(weights, unweighted_rmses)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return expected_rmse, n_eff
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
    ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    min_n_eff = int(n_ref_states * min_neff_factor)

    all_losses = list()
    all_rmses = list()
    all_n_effs = list()

    all_ref_losses = list()
    all_ref_rmses = list()
    all_ref_times = list()

    num_resample_iters = 0
    plot_every = 10
    save_obj_every = args['save_obj_every']
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)

        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0

            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()

            ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(rmse_path, "a") as f:
            f.write(f"{curr_rmse}\n")
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
        all_rmses.append(curr_rmse)

        if i % plot_every == 0 and i:

            plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")

            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_rmses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_rmses, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Expected RMSE")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"rmses_iter{i}.png")
            plt.clf()

        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)
            onp.save(obj_dir / f"ref_rmses_i{i}.npy", onp.array(all_ref_rmses), allow_pickle=False)
            onp.save(obj_dir / f"rmses_i{i}.npy", onp.array(all_rmses), allow_pickle=False)


        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_rmses.npy", onp.array(all_ref_rmses), allow_pickle=False)
    onp.save(obj_dir / f"fin_rmses.npy", onp.array(all_rmses), allow_pickle=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Optimize RMSE of target structure using differentiable trajectory reweighting")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps for sampling reference states per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
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
