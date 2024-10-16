import pdb
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import pandas as pd
import random
from copy import deepcopy
import seaborn as sns
import functools
import pprint

import jax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad
import optax

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import tm

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args):
    # Load parameters
    device = args['device']
    if device == "cpu":
        backend = "CPU"
    elif device == "gpu":
        backend = "CUDA"
    else:
        raise RuntimeError(f"Invalid device: {device}")
    n_threads = args['n_threads']
    key = args['key']
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
    t_kelvin = args['temp']
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    n_iters = args['n_iters']
    lr = args['lr']
    target_pdist = args['target_pdist']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']

    # FIXME: should read in maybe?
    force_per_nuc = args['force_per_nuc']
    ext_force_bps1 = [5, 214] # should each experience force_per_nuc
    ext_force_bps2 = [104, 115] # should each experience -force_per_nuc
    dir_force_axis = jnp.array([0, 0, 1])


    def compute_dist(state):
        end1_com = (state.center[ext_force_bps1[0]] + state.center[ext_force_bps1[1]]) / 2
        end2_com = (state.center[ext_force_bps2[0]] + state.center[ext_force_bps2[1]]) / 2

        midp_disp = end1_com - end2_com
        projected_dist = jnp.dot(midp_disp, dir_force_axis)
        return jnp.linalg.norm(projected_dist) # Note: incase it's negative


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

    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    neff_path = log_dir / "neff.txt"
    dist_path = log_dir / "dist.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/force-ext")
    externals_basedir = sys_basedir / "externals"
    external_path = externals_basedir / f"external_{force_per_nuc}.conf"
    if not external_path.exists():
        raise RuntimeError(f"No external forces file at location: {external_path}")
    input_template_path = sys_basedir / "input"



    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] == 220) # 110 bp duplex
    n = seq_oh.shape[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=True
        # reverse_direction=False
    )

    box_size = conf_info.box_size # Only for writing the trajectory
    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    ext_energy_arr = onp.zeros((n, 3))
    ext_energy_arr[ext_force_bps1[0], 2] = -force_per_nuc # to experience force_per_nuc
    ext_energy_arr[ext_force_bps1[1], 2] = -force_per_nuc # to experience force_per_nuc
    ext_energy_arr[ext_force_bps2[0], 2] = force_per_nuc # to experience -force_per_nuc
    ext_energy_arr[ext_force_bps2[1], 2] = force_per_nuc # to experience -force_per_nuc
    ext_energy_arr = jnp.array(ext_energy_arr)
    def ext_force_energy_fn(body):
        center = body.center
        return jnp.multiply(center, ext_energy_arr).sum()
    ext_force_energy_fn = jit(ext_force_energy_fn)

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
            shutil.copy(external_path, repeat_dir / "external.conf")

            if prev_basedir is None:
                init_conf_info = deepcopy(conf_info)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    reverse_direction=True
                    # reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            oxdna_utils.rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(100),
                equilibration_steps=n_eq_steps, dt=dt,
                no_stdout_energy=0, backend=backend,
                cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
                log_file=str(repeat_dir / "sim.log"),
                external_forces_file=str(repeat_dir / "external.conf")
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
        elif device == "gpu":
            mps_cmd = f"./scripts/mps.sh {oxdna_path} {iter_dir} {n_sims}"
            mps_proc = subprocess.run(mps_cmd, shell=True)
            if mps_proc.returncode != 0:
                raise RuntimeError(f"Generating states via MPS failed with error code: {mps_proc.returncode}")

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

        # FIXME: check that this does not include the external force
        energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
        energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # Check energies
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")

        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        atol_places = 3
        tol = 10**(-atol_places)
        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        # Compute the distances
        start = time.time()
        pdists = vmap(compute_dist)(traj_states)
        end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- took {end - start} seconds\n")

        ## Plot the running average
        running_averages = jnp.cumsum(pdists) / jnp.arange(1, pdists.shape[0]+1)
        plt.plot(running_averages, label=f"{force_per_nuc*2}")
        plt.xlabel("Sample")
        plt.ylabel("Avg. Distance (oxDNA units)")
        plt.title(f"Cumulative average")
        plt.legend()
        plt.savefig(iter_dir / "dist_running_avg.png")
        plt.clf()

        # Plot the running averages (truncated)
        min_running_avg_idx = 50
        plt.plot(running_averages[min_running_avg_idx:], label=f"{force_per_nuc*2}")
        plt.xlabel("Sample")
        plt.ylabel("Avg. Distance (oxDNA units)")
        plt.title(f"Cumulative average")
        plt.legend()
        plt.savefig(iter_dir / "dist_running_avg_truncated.png")
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


        # Add contribution of external force to energies
        ext_force_energy_scan_fn = lambda state, ts: (None, ext_force_energy_fn(ts))
        _, calc_ext_force_energies = scan(ext_force_energy_scan_fn, None, traj_states)
        combined_energies = calc_energies + calc_ext_force_energies

        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(combined_energies, label="Calculated w. ext. force", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"combined_energies.png")
        plt.clf()


        # Record the loss
        mean_pdist = jnp.mean(pdists)
        mse = (target_pdist - mean_pdist)**2
        rmse = jnp.sqrt(mse)
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")
            f.write(f"MSE: {mse}\n")
            f.write(f"RMSE: {rmse}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        return traj_states, combined_energies, pdists, iter_dir


    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_combined_energies, pdists):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        def combined_energy_fn(body):
            base_en = em.energy_fn(body,
                                   seq=seq_oh,
                                   bonded_nbrs=top_info.bonded_nbrs,
                                   unbonded_nbrs=top_info.unbonded_nbrs.T)
            ext_force_en = ext_force_energy_fn(body)
            return base_en + ext_force_en
        combined_energy_fn = jit(combined_energy_fn)

        combined_energy_scan_fn = lambda state, rs: (None, combined_energy_fn(rs))
        _, new_combined_energies = scan(combined_energy_scan_fn, None, ref_states)

        diffs = new_combined_energies - ref_combined_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        expected_pdist = jnp.dot(weights, pdists)
        mse = (target_pdist - expected_pdist)**2
        rmse = jnp.sqrt(mse)

        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return rmse, (n_eff, expected_pdist)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_pdists = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_pdists = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_combined_energies, pdists, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_eff, curr_pdist)), grads = grad_fn(params, ref_states, ref_combined_energies, pdists)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_pdists.append(curr_pdist)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()
            ref_states, ref_combined_energies, pdists, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff, curr_pdist)), grads = grad_fn(params, ref_states, ref_combined_energies, pdists)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_pdists.append(curr_pdist)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(dist_path, "a") as f:
            f.write(f"{curr_pdist}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")

        all_losses.append(loss)
        all_n_effs.append(n_eff)
        all_pdists.append(curr_pdist)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_pdists, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_pdists, marker='o', label="Resample points", color="blue")
        plt.axhline(y=target_pdist, linestyle='--', label="Target pdist", color='red')
        plt.legend()
        plt.ylabel("Expected pdist")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"pdists_iter{i}.png")
        plt.clf()


def get_parser():

    # Simulation arguments
    parser = argparse.ArgumentParser(description="Optimize a single force extension")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int,
                        # default=10000,
                        default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='Run name')
    parser.add_argument('--temp', type=float, default=utils.DEFAULT_TEMP,
                        help="Simulation temperature in Kelvin")
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")
    parser.add_argument('--force-per-nuc', type=float,
                        # Note: Tom considers the following forces: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
                        choices=[0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375],
                        help="Force per nucleotide. E.g. for 0.75 total on each side, put 0.375")


    # Optimization arguments
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--target-pdist', type=float, required=True,
                        help="Target avg. projected end to end distance under the force")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    assert(args['n_eq_steps'] == 0)

    run(args)
