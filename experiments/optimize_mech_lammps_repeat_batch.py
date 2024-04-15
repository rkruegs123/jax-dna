from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import subprocess
import pdb
from copy import deepcopy
import time
import functools
import numpy as onp
import pprint
import random
import pandas as pd
import socket
from collections import Counter

from jax import jit, vmap, lax, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body
import optax

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna2 import model, lammps_utils




checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def single_pitch(quartet, base_sites, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    # get normalized base-base vectors for each base pair, 1 and 2
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    bb1 = bb1[:2]
    bb2 = bb2[:2]

    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = bb2 / jnp.linalg.norm(bb2)

    theta = jnp.arccos(utils.clamp(jnp.dot(bb1, bb2)))

    return theta


def compute_pitches(body, quartets, displacement_fn, com_to_hb):
    # Construct the base site position in the body frame
    base_site_bf = jnp.array([com_to_hb, 0.0, 0.0])

    # Compute the space-frame base sites
    base_sites = body.center + rigid_body.quaternion_rotate(
        body.orientation, base_site_bf)

    # Compute the pitches for all quartets
    all_pitches = vmap(single_pitch, (0, None, None))(
        quartets, base_sites, displacement_fn)

    return all_pitches


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

def run(args):
    lammps_basedir = Path(args['lammps_basedir'])
    assert(lammps_basedir.exists())
    lammps_exec_path = lammps_basedir / "build/lmp"
    assert(lammps_exec_path.exists())

    tacoxdna_basedir = Path(args['tacoxdna_basedir'])
    assert(tacoxdna_basedir.exists())

    sample_every = args['sample_every']
    n_sims = args['n_sims']

    n_eq_steps = args['n_eq_steps']
    assert(n_eq_steps % sample_every == 0)
    n_eq_states = n_eq_steps // sample_every

    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_sample_states = n_sample_steps // sample_every

    n_total_steps = n_eq_steps + n_sample_steps
    n_total_states = n_total_steps // sample_every
    assert(n_total_states == n_sample_states + n_eq_states)

    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    rmse_coeff = args['rmse_coeff']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    seq_avg = not args['seq_dep']
    assert(seq_avg)

    # forces_pn = jnp.array([0.0, 2.0, 10.0, 15.0])
    # target_deltas = jnp.array([0.0, 0.1, 0.5, 0.75]) # per kb
    # force_colors = ["blue", "red", "green", "purple"]

    # forces_pn = jnp.array([0.0, 2.0, 10.0])
    # target_deltas = jnp.array([0.0, 0.1, 0.5]) # per kb
    # force_colors = ["blue", "red", "green"]

    forces_pn = jnp.array([0.0, 2.0, 4.0, 6.0, 10.0])
    target_deltas = jnp.array([0.0, 0.1, 0.2, 0.3, 0.5])
    force_colors = ["blue", "red", "green", "purple", "orange"]

    target_slope = 0.05


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
    neffs_path = log_dir / "neffs.txt"
    slope_path = log_dir / "slope.txt"
    diffs_path = log_dir / "diffs.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_sample_states: {n_sample_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/templates/lammps-stretch-tors")
    lammps_data_rel_path = sys_basedir / "data"
    lammps_data_abs_path = os.getcwd() / lammps_data_rel_path

    p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", lammps_data_abs_path], cwd=run_dir)
    p.wait()
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

    init_conf_fpath = run_dir / "data.oxdna"
    assert(init_conf_fpath.exists())
    os.rename(init_conf_fpath, run_dir / "init.conf")

    top_fpath = run_dir / "data.top"
    assert(top_fpath.exists())
    os.rename(top_fpath, run_dir / "sys.top")
    top_fpath = run_dir / "sys.top"

    top_info = topology.TopologyInfo(top_fpath, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n = seq_oh.shape[0]
    assert(n % 2 == 0)
    n_bp = n // 2

    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    ## The region for which theta and distance are measured
    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[4:n_bp-5]

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]

    rise_per_bp = 3.4 / utils.ang_per_oxdna_length # oxDNA length units
    contour_length = quartets.shape[0] * rise_per_bp # oxDNA length units

    displacement_fn, shift_fn = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.

    @jit
    def compute_theta(body):
        pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
        return pitches.sum()

    t_kelvin = 300.0
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    salt_conc = 0.15
    q_eff = 0.815


    def get_ref_states(params, i, seed):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        # Run the simulations
        procs = list()
        all_sim_dirs = list()
        repeat_seeds = [random.randrange(100) for _ in range(n_sims)] # This is important!
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                repeat_seed = repeat_seeds[r]

                shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                    force_pn=force_pn, torque_pnnm=0,
                    save_every=sample_every, n_steps=n_total_steps,
                    seq_avg=seq_avg, seed=repeat_seed)

                procs.append(subprocess.Popen([lammps_exec_path, "-in", "in"], cwd=repeat_dir))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"LAMMPS simulation failed with error code: {rc}")

        # Convert via TacoxDNA
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=repeat_dir)
                p.wait()
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")


        # Analyze

        ## Generate an energy function
        if seq_avg:
            ss_hb_weights = utils.HB_WEIGHTS_SA
            ss_stack_weights = utils.STACK_WEIGHTS_SA
        else:
            ss_path = "data/seq-specific/seq_oxdna2.txt"
            ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                ss_path,
                model.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                model.default_base_params_seq_dep['stacking']['eps_stack_base'],
                model.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                enforce_symmetry=False,
                t_kelvin=t_kelvin
            )
        em = model.EnergyModel(displacement_fn,
                               params,
                               t_kelvin=t_kelvin,
                               ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                               salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                               ignore_exc_vol_bonded=True # Because we're in LAMMPS
        )
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))

        ## Populate our states, energies, and thetas
        all_traj_states = list()
        all_calc_energies = list()
        all_gt_energies = list()
        all_thetas = list()
        for r in tqdm(range(n_sims), desc="Processing repeats"):
            repeat_traj_states = list()
            repeat_calc_energies = list()
            repeat_gt_energies = list()
            repeat_thetas = list()
            for force_pn in forces_pn:
                repeat_dir = iter_dir / f"sim-f{force_pn}" / f"r{r}"
                traj_info = trajectory.TrajectoryInfo(
                    top_info, read_from_file=True, reindex=True,
                    traj_path=repeat_dir / "data.oxdna",
                    # reverse_direction=True)
                    reverse_direction=False)

                full_traj_states = traj_info.get_states()
                assert(len(full_traj_states) == 1+n_total_states)
                sampled_sim_states = full_traj_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states)

                sampled_sim_states = utils.tree_stack(sampled_sim_states)

                log_path = repeat_dir / "log.lammps"
                rpt_log_df = lammps_utils.read_log(log_path)
                assert(rpt_log_df.shape[0] == n_total_states+1)
                rpt_log_df = rpt_log_df[1+n_eq_states:]

                ## Compute the energies via our energy function
                _, calc_energies = scan(energy_scan_fn, None, sampled_sim_states)

                ## Check energies
                gt_energies = (rpt_log_df.PotEng * seq_oh.shape[0]).to_numpy()
                energy_diffs = list()
                for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                    diff = onp.abs(calc - gt)
                    energy_diffs.append(diff)

                sns.distplot(calc_energies, label="Calculated", color="red")
                sns.distplot(gt_energies, label="Reference", color="green")
                plt.legend()
                plt.savefig(repeat_dir / f"energies.png")
                plt.clf()

                sns.histplot(energy_diffs)
                plt.savefig(repeat_dir / f"energy_diffs.png")
                plt.clf()

                ## Compute the thetas
                traj_thetas = list()
                for rs_idx in range(n_sample_states):
                    ref_state = sampled_sim_states[rs_idx]
                    theta = compute_theta(ref_state)
                    traj_thetas.append(theta)

                traj_thetas = onp.array(traj_thetas)

                repeat_traj_states.append(sampled_sim_states)
                repeat_calc_energies.append(calc_energies)
                repeat_gt_energies.append(gt_energies)
                repeat_thetas.append(traj_thetas)

            all_traj_states.append(repeat_traj_states)
            all_calc_energies.append(repeat_calc_energies)
            all_gt_energies.append(repeat_gt_energies)
            all_thetas.append(repeat_thetas)

        all_traj_states = utils.tree_stack([utils.tree_stack(r_states) for r_states in all_traj_states])
        all_calc_energies = jnp.array(all_calc_energies)
        all_gt_energies = jnp.array(all_gt_energies)
        all_thetas = jnp.array(all_thetas)

        sns.distplot(all_calc_energies.flatten(), label="Calculated", color="red")
        sns.distplot(all_gt_energies.flatten(), label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        ## For each repeat, get the running average of the slope
        all_running_slopes = list()
        all_running_slope_offset0s = list()
        for r in range(n_sims):
            repeat_thetas = all_thetas[r]
            running_avgs = onp.cumsum(repeat_thetas, axis=1) / onp.arange(1, n_sample_states+1)
            assert(running_avgs.shape[1] == n_sample_states)
            assert(running_avgs.shape[0] == len(forces_pn))

            running_avg_freq = 10
            running_avg_idxs = onp.arange(0, n_sample_states, running_avg_freq)
            repeat_slopes = list()
            repeat_slope_offset0s = list()
            for idx in tqdm(running_avg_idxs, desc="Running avg. indices"):
                force_avgs = running_avgs[:, idx]
                thetas_per_kb = force_avgs * (1000 / 36)
                theta_diffs = thetas_per_kb - thetas_per_kb[0]

                # Fit with a flexible offset
                xs_to_fit = jnp.stack([jnp.ones_like(forces_pn), forces_pn], axis=1)
                fit_ = jnp.linalg.lstsq(xs_to_fit, theta_diffs)
                slope = fit_[0][1]
                offset = fit_[0][0]

                repeat_slopes.append(slope)

                # Fit with offset=0
                # xs_to_fit = forces_pn[:, onp.newaxis]
                xs_to_fit = jnp.stack([jnp.zeros_like(forces_pn), forces_pn], axis=1)
                fit_ = jnp.linalg.lstsq(xs_to_fit, theta_diffs)
                slope_offset0 = fit_[0][1]
                repeat_slope_offset0s.append(slope_offset0)

            all_running_slopes.append(repeat_slopes)
            all_running_slope_offset0s.append(repeat_slope_offset0s)

        for r in range(n_sims):
            plt.plot(all_running_slopes[r], label=f"r{r}")
        plt.legend()
        plt.savefig(iter_dir / "running_avg_slope.png")
        plt.clf()

        for r in range(n_sims):
            plt.plot(all_running_slope_offset0s[r], label=f"r{r}")
        plt.legend()
        plt.savefig(iter_dir / "running_avg_slope_offset0.png")
        plt.clf()

        ## Plot all the final scatter plots
        for r in range(n_sims):
            repeat_thetas = all_thetas[r]
            force_avgs = onp.mean(repeat_thetas, axis=1)
            thetas_per_kb = force_avgs * (1000 / 36)
            # theta_diffs = thetas_per_kb - thetas_per_kb[0]
            plt.scatter(forces_pn, thetas_per_kb, label=f"r{r}")
        plt.legend()
        plt.savefig(iter_dir / "repeat_scatter.png")
        plt.clf()

        repeat_slopes = list()
        for r in range(n_sims):
            repeat_thetas = all_thetas[r]
            force_avgs = onp.mean(repeat_thetas, axis=1)
            thetas_per_kb = force_avgs * (1000 / 36)
            theta_diffs = thetas_per_kb - thetas_per_kb[0]

            xs_to_fit = forces_pn[:, onp.newaxis]
            fit_ = jnp.linalg.lstsq(xs_to_fit, theta_diffs)
            slope_offset0 = fit_[0][1]

            repeat_slopes.append(slope_offset0)
            plt.scatter(forces_pn, theta_diffs, label=f"r{r}, slope={onp.round(slope_offset0, 3)}")
        plt.legend()
        plt.savefig(iter_dir / "repeat_scatter_norm.png")
        plt.clf()

        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Avg slope: {onp.mean(repeat_slopes)}\n")

        return all_traj_states, all_calc_energies, all_thetas

    @jit
    def loss_fn(params, all_ref_states, all_ref_energies, all_ref_thetas):
        # Setup energy function
        em = model.EnergyModel(displacement_fn,
                               params,
                               t_kelvin=t_kelvin,
                               salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                               ignore_exc_vol_bonded=True # Because we're in LAMMPS
        )
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

        def get_expected_force_theta(ref_states, ref_energies, ref_thetas):
            _, new_energies = scan(energy_scan_fn, None, ref_states)

            diffs = new_energies - ref_energies
            boltzs = jnp.exp(-beta * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom

            expected_theta = jnp.dot(weights, ref_thetas)
            n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
            return expected_theta, n_eff

        def get_expected_repeat_slope(r_idx):
            r_ref_states = all_ref_states[r_idx]
            r_ref_energies = all_ref_energies[r_idx]
            r_ref_thetas = all_ref_thetas[r_idx]
            thetas, n_effs = vmap(get_expected_force_theta, (0, 0, 0))(r_ref_states, r_ref_energies, r_ref_thetas)

            theta_diffs = (thetas - thetas[0]) * (1000 / 36)

            """
            xs_to_fit = forces_pn[:, jnp.newaxis]
            fit_ = jnp.linalg.lstsq(xs_to_fit, theta_diffs)
            expected_slope = fit_[0][1]
            """

            # xs_to_fit = jnp.stack([jnp.ones_like(forces_pn), forces_pn], axis=1)
            xs_to_fit = jnp.stack([jnp.zeros_like(forces_pn), forces_pn], axis=1)
            fit_ = jnp.linalg.lstsq(xs_to_fit, theta_diffs)
            expected_slope = fit_[0][1]
            expected_offset = fit_[0][0]

            return expected_slope, n_effs, theta_diffs

        expected_slopes, all_n_effs, all_theta_diffs = vmap(get_expected_repeat_slope)(jnp.arange(n_sims))

        n_effs = jnp.sum(all_n_effs, axis=0)
        theta_diffs = jnp.sum(all_theta_diffs, axis=0)
        expected_slope = jnp.mean(expected_slopes)

        rmse = jnp.sqrt((target_slope - expected_slope)**2)
        return rmse_coeff*rmse, (n_effs, expected_slope, theta_diffs)


    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    if seq_avg:
        params["stacking"] = deepcopy(model.default_base_params_seq_avg["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_avg["hydrogen_bonding"])
        # params["cross_stacking"] = deepcopy(model.default_base_params_seq_avg["cross_stacking"])
        # del params["cross_stacking"]["dr_c_cross"]
        # del params["cross_stacking"]["dr_low_cross"]
        # del params["cross_stacking"]["dr_high_cross"]
    else:
        params["stacking"] = deepcopy(model.default_base_params_seq_dep["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_dep["hydrogen_bonding"])
        # params["cross_stacking"] = deepcopy(model.default_base_params_seq_dep["cross_stacking"])
        # del params["cross_stacking"]["dr_c_cross"]
        # del params["cross_stacking"]["dr_low_cross"]
        # del params["cross_stacking"]["dr_high_cross"]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # min_n_eff = int(2*n_sample_states * min_neff_factor) # 2x because we simulate at two different forces
    min_n_eff = int(n_sample_states * min_neff_factor) # 2x because we simulate at two different forces

    all_losses = list()
    all_slopes = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_slopes = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    all_ref_states, all_ref_energies, all_ref_thetas = get_ref_states(params, i=0, seed=30362)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_effs, curr_slope, curr_diffs)), grads = grad_fn(
            params, all_ref_states, all_ref_energies, all_ref_thetas)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_slopes.append(curr_slope)

        resample = False
        for n_eff in n_effs:
            if n_eff < min_n_eff:
                resample = True
                break

        if resample or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- min n_eff was {n_effs.min()}...")

            start = time.time()
            all_ref_states, all_ref_energies, all_ref_thetas = get_ref_states(params, i=i, seed=i)
            end = time.time()
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_effs, curr_slope, curr_diffs)), grads = grad_fn(
                params, all_ref_states, all_ref_energies, all_ref_thetas)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_slopes.append(curr_slope)


        iter_end = time.time()

        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(diffs_path, "a") as f:
            f.write(f"{curr_diffs}\n")
        with open(neffs_path, "a") as f:
            f.write(f"{n_effs}\n")
        with open(slope_path, "a") as f:
            f.write(f"{curr_slope}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(iter_params_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")

        all_losses.append(loss)
        all_n_effs.append(n_effs)

        all_slopes.append(curr_slope)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


        plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()


        plt.plot(onp.arange(i+1), all_slopes, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_slopes, marker='o', label="Resample points", color="blue")
        plt.axhline(y=target_slope, linestyle='--', label="Target Slope", color='red')
        plt.legend()
        plt.ylabel("Expected Slope")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"slopes_iter{i}.png")
        plt.clf()


def get_parser():

    parser = argparse.ArgumentParser(description="Optimize the length under a pulling force via LAMMPS")

    parser.add_argument('--n-sample-steps', type=int, default=3000000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=100000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=500,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of simulations per force")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--lammps-basedir', type=str,
                        default="/n/brenner_lab/Lab/software/lammps-stable_29Sep2021",
                        help='LAMMPS base directory')
    parser.add_argument('--tacoxdna-basedir', type=str,
                        default="/n/brenner_lab/User/rkrueger/tacoxDNA",
                        help='tacoxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--rmse-coeff', type=float, default=100.0,
                        help="Coefficient for the RMSE")
    parser.add_argument('--seq-dep', action='store_true')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
