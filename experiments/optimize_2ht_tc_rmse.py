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
import random
import functools

from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs
from oxDNA_analysis_tools.deviations import deviations

import optax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, lax, tree_util
from jax_md import space, rigid_body

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.rna2 import model, oxrna_utils
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.rna2.load_params import read_seq_specific, DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS
import jax_dna.input.trajectory as jdt


from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 50
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


default_observables_str = """
data_output_1 = {
  name = split_energy.dat
  print_every = PRINT_EVERY
  col_1 = {
          type = step
          units = MD
  }
  col_2 = {
          type = potential_energy
          # type = pair_energy
          split = true
  }
}
"""

def run(args):
    # Load parameters
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
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    no_delete = args['no_delete']
    n_threads = args['n_threads']

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

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 3e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    def get_ref_states(params, i, seed, prev_basedir):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        procs = list()
        sim_start = time.time()

        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")

            seq_dep_path = repeat_dir / "rna_sequence_dependent_parameters.txt"
            # shutil.copy(ss_path, seq_dep_path)
            if "stacking" in params["seq_dep"]:
                curr_stack_weights = params["seq_dep"]["stacking"]
            else:
                curr_stack_weights = ss_stack_weights
            oxrna_utils.write_seq_specific(seq_dep_path, params["seq_avg"], ss_hb_weights, curr_stack_weights, ss_cross_weights)

            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            external_model_fpath = repeat_dir / "external_model.txt"
            oxrna_utils.write_external_model(params["seq_avg"], t_kelvin, salt_conc, external_model_fpath)

            split_energy_path = str(repeat_dir / "split_energy.dat")
            observables_str = default_observables_str.replace("split_energy.dat", split_energy_path)
            observables_str = observables_str.replace("PRINT_EVERY", str(sample_every))

            rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(1000),
                equilibration_steps=n_eq_steps, dt=dt,
                no_stdout_energy=0,
                log_file=str(repeat_dir / "sim.log"),
                external_model=str(external_model_fpath),
                seq_dep_file=str(seq_dep_path),
                seq_dep_file_RNA=str(seq_dep_path),
                observables_str=observables_str
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

        split_energy_df_columns = ["t", "fene", "b_exc", "stack", "n_exc", "hb",
                                   "cr_stack", "cx_stack", "debye"]
        split_energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "split_energy.dat", names=split_energy_df_columns,
                                        delim_whitespace=True)[1:] for r in range(n_sims)]
        split_energy_df = pd.concat(split_energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            # ss_stack_weights=ss_stack_weights)
            ss_stack_weights=curr_stack_weights)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        subterms_fn = lambda body: em.compute_subterms(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        subterms_fn = jit(subterms_fn)

        ## Check energies
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

        ## Check energy subterms
        calc_start = time.time()
        subterms_scan_fn = lambda state, ts: (None, subterms_fn(ts))
        _, subterms = scan(subterms_scan_fn, None, traj_states)
        fene_dgs, exc_vol_bonded_dgs, stack_dgs, exc_vol_unbonded_dgs, hb_dgs, cr_stack_dgs, cx_stack_dgs, debye_dgs = subterms
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating subterms took {calc_end - calc_start} seconds\n")

        fene_diffs, exc_vol_bonded_diffs, stack_diffs, exc_vol_unbonded_diffs, hb_diffs, cr_stack_diffs, cx_stack_diffs, debye_diffs = list(), list(), list(), list(), list(), list(), list(), list()

        gt_fenes = split_energy_df["fene"].to_numpy() * seq_oh.shape[0]
        gt_bexcs = split_energy_df["b_exc"].to_numpy() * seq_oh.shape[0]
        gt_stacks = split_energy_df["stack"].to_numpy() * seq_oh.shape[0]
        gt_nexcs = split_energy_df["n_exc"].to_numpy() * seq_oh.shape[0]
        gt_hbs = split_energy_df["hb"].to_numpy() * seq_oh.shape[0]
        gt_cr_stacks = split_energy_df["cr_stack"].to_numpy() * seq_oh.shape[0]
        gt_cx_stacks = split_energy_df["cx_stack"].to_numpy() * seq_oh.shape[0]
        gt_debyes = split_energy_df["debye"].to_numpy() * seq_oh.shape[0]
        for i in range(n_ref_states):
            fene_diffs.append(onp.abs(fene_dgs[i] - gt_fenes[i]))
            exc_vol_bonded_diffs.append(onp.abs(exc_vol_bonded_dgs[i] - gt_bexcs[i]))
            stack_diffs.append(onp.abs(stack_dgs[i] - gt_stacks[i]))
            exc_vol_unbonded_diffs.append(onp.abs(exc_vol_unbonded_dgs[i] - gt_nexcs[i]))
            hb_diffs.append(onp.abs(hb_dgs[i] - gt_hbs[i]))
            cr_stack_diffs.append(onp.abs(cr_stack_dgs[i] - gt_cr_stacks[i]))
            cx_stack_diffs.append(onp.abs(cx_stack_dgs[i] - gt_cx_stacks[i]))
            debye_diffs.append(onp.abs(debye_dgs[i] - gt_debyes[i]))


        plt.hist(fene_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"fene_diffs_hist.png")
        plt.clf()

        plt.hist(exc_vol_bonded_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"bexc_diffs_hist.png")
        plt.clf()

        plt.hist(stack_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"stack_diffs_hist.png")
        plt.clf()

        plt.hist(exc_vol_unbonded_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"nexc_diffs_hist.png")
        plt.clf()

        plt.hist(hb_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"hb_diffs_hist.png")
        plt.clf()

        plt.hist(cr_stack_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"cr_stack_diffs_hist.png")
        plt.clf()

        plt.hist(cx_stack_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"cx_stack_diffs_hist.png")
        plt.clf()

        plt.hist(debye_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"debye_diffs_hist.png")
        plt.clf()

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
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        # sns.histplot(energy_diffs)
        plt.hist(energy_diffs, 25, histtype='bar', facecolor='blue')
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.clf()

        mean_rmsd = onp.mean(RMSDs)
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")
            f.write(f"Mean RMSD: {mean_rmsd}\n")

            f.write(f"\nFENE diff, mean: {onp.mean(fene_diffs)}\n")
            f.write(f"FENE diff, var: {onp.var(fene_diffs)}\n")

            f.write(f"\nBonded Exc. Vol diff, mean: {onp.mean(exc_vol_bonded_diffs)}\n")
            f.write(f"Bonded Exc. Vol diff, var: {onp.var(exc_vol_bonded_diffs)}\n")

            f.write(f"\nStack diff, mean: {onp.mean(stack_diffs)}\n")
            f.write(f"Stack diff, var: {onp.var(stack_diffs)}\n")

            f.write(f"\nNonbonded Exc. Vol diff, mean: {onp.mean(exc_vol_unbonded_diffs)}\n")
            f.write(f"Nonbonded Exc. Vol diff, var: {onp.var(exc_vol_unbonded_diffs)}\n")

            f.write(f"\nHB diff, mean: {onp.mean(hb_diffs)}\n")
            f.write(f"HB diff, var: {onp.var(hb_diffs)}\n")

            f.write(f"\nCr. Stack diff, mean: {onp.mean(cr_stack_diffs)}\n")
            f.write(f"Cr. Stack diff, var: {onp.var(cr_stack_diffs)}\n")

            f.write(f"\nCx. Stack diff, mean: {onp.mean(cx_stack_diffs)}\n")
            f.write(f"Cx. Stack diff, var: {onp.var(cx_stack_diffs)}\n")

            f.write(f"\nDebye diff, mean: {onp.mean(debye_diffs)}\n")
            f.write(f"Debye diff, var: {onp.var(debye_diffs)}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        # Spring cleaning
        if not no_delete:
            for r in range(n_sims):
                repeat_dir = iter_dir / f"r{r}"
                file_to_rem = repeat_dir / "output.dat"
                file_to_rem.unlink()
            file_to_rem = iter_dir / "output.dat"
            file_to_rem.unlink()

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
            ss_stack_weights=curr_stack_weights)
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

    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
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

            start = time.time()
            ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
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
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--no-delete', action='store_true')

    parser.add_argument('--full-system', action='store_true')

    parser.add_argument(
        '--seq-avg-opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "cross_stacking", "coaxial_stacking"],
        help='Parameter keys to optimize'
    )
    parser.add_argument('--opt-seq-dep-stacking', action='store_true')

    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for trajectory reading")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
