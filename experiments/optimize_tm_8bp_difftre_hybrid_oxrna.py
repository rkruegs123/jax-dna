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
import zipfile
import os
import pprint

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad
import optax

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.rna2 import model, oxrna_utils
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.rna2.load_params import read_seq_specific, DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS
from jax_dna.loss import tm

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))

def run(args):
    # Load parameters
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
    extrapolate_temps = jnp.array([float(et) for et in args['extrapolate_temps']]) # in Kelvin
    assert(jnp.all(extrapolate_temps[:-1] <= extrapolate_temps[1:])) # check that temps. are sorted
    n_extrap_temps = len(extrapolate_temps)
    extrapolate_kts = vmap(utils.get_kt)(extrapolate_temps)
    extrapolate_temp_str = ', '.join([f"{tc}K" for tc in extrapolate_temps])
    salt_concentration = args['salt_concentration']

    n_iters = args['n_iters']
    lr = args['lr']
    target_tm = args['target_tm']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    optimizer_type = args['optimizer_type']

    opt_keys = args['opt_keys']
    no_delete = args['no_delete']
    no_archive = args['no_archive']
    plot_every = args['plot_every']
    save_obj_every = args['save_obj_every']


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
    tm_path = log_dir / "tm.txt"
    width_path = log_dir / "width.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/tm-8bp-rna")
    input_template_path = sys_basedir / "input"

    weight_path = sys_basedir / "wfile.txt"
    weight_df = pd.read_csv(weight_path, delim_whitespace=True, names=["op", "weight"])
    weight_mapper = dict(zip(weight_df.op, weight_df.weight))

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path,
                                     reverse_direction=False
                                     # reverse_direction=True
    )
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] % 2 == 0)
    n_bp = seq_oh.shape[0] // 2

    conf_path_bound = sys_basedir / "init_bound.conf"
    conf_info_bound = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path_bound,
        # reverse_direction=True
        reverse_direction=False
    )
    conf_path_unbound = sys_basedir / "init_unbound.conf"
    conf_info_unbound = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path_unbound,
        # reverse_direction=True
        reverse_direction=False
    )
    box_size = conf_info_bound.box_size

    weights_path = sys_basedir / "wfile.txt"
    op_path = sys_basedir / "op.txt"

    displacement_fn, shift_fn = space.periodic(box_size)
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    max_seed_tries = 5
    seed_check_sample_freq = 10
    seed_check_steps = 100

    def get_ref_states(params, i, seed, prev_basedir):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        procs = list()
        start = time.time()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")
            shutil.copy(weights_path, repeat_dir / "wfile.txt")
            shutil.copy(op_path, repeat_dir / "op.txt")

            if prev_basedir is None or True: # FIXME: just doing this every time
                if r % 2 == 0:
                    conf_info_copy = deepcopy(conf_info_bound)
                else:
                    conf_info_copy = deepcopy(conf_info_unbound)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                # conf_info_copy = center_configuration.center_conf(top_info, prev_lastconf_info)
            conf_info_copy.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)

            conf_info_copy.write(repeat_dir / "init.conf",
                                 reverse=False,
                                 # reverse=True,
                                 write_topology=False)

            external_model_fpath = repeat_dir / "external_model.txt"
            oxrna_utils.write_external_model(params, t_kelvin, salt_concentration, external_model_fpath)

            check_seed_dir = repeat_dir / "check_seed"
            check_seed_dir.mkdir(parents=False, exist_ok=False)

            s_idx = 0
            valid_seed = None
            while s_idx < max_seed_tries and valid_seed is None:
                seed_try = random.randrange(10000)
                seed_dir = check_seed_dir / f"s{seed_try}"
                seed_dir.mkdir(parents=False, exist_ok=False)

                rewrite_input_file(
                    input_template_path, seed_dir,
                    temp=f"{t_kelvin}K", steps=seed_check_steps,
                    init_conf_path=str(repeat_dir / "init.conf"),
                    top_path=str(repeat_dir / "sys.top"),
                    save_interval=seed_check_sample_freq, seed=seed_try,
                    equilibration_steps=0,
                    no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str,
                    weights_file=str(repeat_dir / "wfile.txt"), op_file=str(repeat_dir / "op.txt"),
                    log_file=str(seed_dir / "sim.log"),
                    interaction_type="RNA2",
                    salt_concentration=salt_concentration,
                    external_model=str(external_model_fpath)
                )

                seed_proc = subprocess.Popen([oxdna_exec_path, seed_dir / "input"])
                seed_proc.wait()
                seed_rc = seed_proc.returncode

                if seed_rc == 0:
                    valid_seed = seed_try

                s_idx += 1

            if valid_seed is None:
                raise RuntimeError(f"Could not find valid seed.")


            rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=valid_seed,
                equilibration_steps=n_eq_steps,
                no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str,
                weights_file=str(repeat_dir / "wfile.txt"), op_file=str(repeat_dir / "op.txt"),
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="RNA2",
                salt_concentration=salt_concentration,
                external_model=str(external_model_fpath)
            )

            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

        end = time.time()
        sim_time = end - start

        start = time.time()

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

        ## Compute running avg. from all `traj_hist.dat`
        all_traj_hist_fpaths = list()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")
        all_running_tms, all_running_widths = tm.traj_hist_running_avg_1d(all_traj_hist_fpaths)

        plt.plot(all_running_tms)
        plt.xlabel("Iteration")
        plt.ylabel("Tm (C)")
        plt.savefig(iter_dir / "traj_hist_running_tm.png")
        # plt.show()
        plt.clf()

        plt.plot(all_running_widths)
        plt.xlabel("Iteration")
        plt.ylabel("Width (C)")
        plt.savefig(iter_dir / "traj_hist_running_width.png")
        # plt.show()
        plt.clf()

        ## Load states from oxDNA simulation
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / "output.dat",
            reverse_direction=False
            # reverse_direction=True
        )
        ref_states = traj_info.get_states()
        assert(len(ref_states) == n_ref_states)
        ref_states = utils.tree_stack(ref_states)

        ## Load the oxDNA energies
        """
        energy_df_columns = [
            "time", "potential_energy", "kinetic_energy", "total_energy",
            "op_idx", "op", "op_weight"
        ]
        """
        energy_df_columns = [
            "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot",
            "acc_ratio_vol", "op", "op_weight"
        ]
        energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Load all last_hist.dat and combine
        last_hist_columns = ["num_bp", "count_biased", "count_unbiased"] \
                            + [str(et) for et in extrapolate_temps]
        last_hist_df = None
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            last_hist_fpath = repeat_dir / "last_hist.dat"
            repeat_last_hist_df = pd.read_csv(
                last_hist_fpath, delim_whitespace=True, skiprows=[0], header=None,
                names=last_hist_columns)

            if last_hist_df is None:
                last_hist_df = repeat_last_hist_df
            else:
                last_hist_df = last_hist_df.add(repeat_last_hist_df)

        ## Check energies
        em_base = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_concentration)
        energy_fn = lambda body: em_base.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T
        )
        energy_fn = jit(energy_fn)

        ref_energies = list()
        for rs_idx in tqdm(range(n_ref_states), desc="Calculating energies"):
            rs = ref_states[rs_idx]
            ref_energies.append(energy_fn(rs))
        ref_energies = jnp.array(ref_energies)

        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        atol_places = 3
        tol = 10**(-atol_places)
        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(ref_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            if diff > tol:
                print(f"WARNING: energy difference of {diff}")
                # pdb.set_trace() # note: in practice, we wouldn't set a trace
            energy_diffs.append(diff)

        sns.distplot(ref_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        # plt.show()
        plt.savefig(iter_dir / "energies.png")
        plt.clf()

        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / "energy_diffs.png")
        # plt.show()
        plt.clf()


        ## Check uniformity across biased counts
        count_df = energy_df.groupby(['op', 'op_weight']).size().reset_index().rename(columns={0: "count"})
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=count_df['op'], y=count_df['count'], ax=ax[0])
        ax[0].set_title("Periodic Counts")
        ax[0].set_xlabel("num_bp")
        sns.barplot(x=last_hist_df['num_bp'], y=last_hist_df['count_biased'], ax=ax[1])
        ax[1].set_title("Frequent Counts")

        plt.savefig(iter_dir / "biased_counts.png")
        plt.clf()


        ## Unbias reference counts
        ref_unbiased_counts = onp.zeros(n_bp+1)
        all_ops = energy_df.op.to_numpy()
        all_op_weights = onp.array([weight_mapper[op] for op in all_ops])
        for rs_idx in tqdm(range(n_ref_states)):
            op = all_ops[rs_idx]
            op_weight = all_op_weights[rs_idx]
            ref_unbiased_counts[op] += 1/op_weight

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=jnp.arange(9), y=ref_unbiased_counts, ax=ax[0])
        ax[0].set_title(f"Periodic Counts, Reference T={t_kelvin}K")
        ax[0].set_xlabel("num_bp")
        sns.barplot(x=last_hist_df['num_bp'], y=last_hist_df['count_unbiased'], ax=ax[1])
        ax[1].set_title(f"Frequent Counts, Reference T={t_kelvin}K")
        # fig.show()
        # plt.show()
        plt.savefig(iter_dir / "unbiased_counts.png")
        plt.clf()

        sim_finf = tm.compute_finf(ref_unbiased_counts)

        ## Unbias counts for each temperature
        all_unbiased_counts = list()
        running_avg_min = 250
        running_avg_freq = 50
        running_avg_mapper = dict()
        for extrap_t_kelvin, extrap_kt in zip(extrapolate_temps, extrapolate_kts):
            em_temp = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_concentration)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T
            )
            energy_fn_temp = jit(energy_fn_temp)

            temp_unbiased_counts = onp.zeros(9)
            for rs_idx in tqdm(range(n_ref_states), desc=f"Extrapolating to {extrap_t_kelvin}K"):
                rs = ref_states[rs_idx]
                op = all_ops[rs_idx]
                op_weight = weight_mapper[op]

                calc_energy = ref_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT - calc_energy_temp/extrap_kt)
                temp_unbiased_counts[op] += 1/op_weight * boltz_diff


                if rs_idx >= running_avg_min and rs_idx % running_avg_freq == 0:
                    if rs_idx not in running_avg_mapper:
                        running_avg_mapper[rs_idx] = [deepcopy(temp_unbiased_counts)]
                    else:
                        running_avg_mapper[rs_idx].append(deepcopy(temp_unbiased_counts))


            all_unbiased_counts.append(temp_unbiased_counts)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(x=jnp.arange(9), y=temp_unbiased_counts, ax=ax[0])
            ax[0].set_title(f"Periodic Counts, T={extrap_t_kelvin}K")
            ax[0].set_xlabel("num_bp")
            sns.barplot(x=last_hist_df['num_bp'], y=last_hist_df[str(extrap_t_kelvin)], ax=ax[1])
            ax[1].set_title(f"Frequent Counts, T={extrap_t_kelvin}K")
            # plt.show()
            plt.savefig(iter_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.clf()

        ## Compute discrete running avg. Tm
        running_avg_iters = list()
        running_avg_tms = list()
        for running_avg_iter, running_avg_unb_counts in running_avg_mapper.items():
            iter_unb_counts = onp.array(running_avg_unb_counts)
            iter_discrete_finfs = vmap(tm.compute_finf)(iter_unb_counts)
            iter_tm = tm.compute_tm(extrapolate_temps, iter_discrete_finfs)

            running_avg_iters.append(running_avg_iter)
            running_avg_tms.append(iter_tm)

        plt.plot(running_avg_iters, running_avg_tms, '-o')
        plt.savefig(iter_dir / f"discrete_running_avg_tm.png")
        plt.clf()

        ## Compute final Tms and widths, log all, as well as time
        all_unbiased_counts = onp.array(all_unbiased_counts) # (ntemps, n_bp+1)
        discrete_finfs = vmap(tm.compute_finf)(all_unbiased_counts)

        calc_tm = tm.compute_tm(extrapolate_temps, discrete_finfs)
        calc_width = tm.compute_width(extrapolate_temps, discrete_finfs)

        last_hist_extrap_counts = last_hist_df.to_numpy()[:, -n_extrap_temps:].T # (ntemps, n_bp+1)
        ref_finfs = vmap(tm.compute_finf)(last_hist_extrap_counts)

        ref_tm = tm.compute_tm(extrapolate_temps, ref_finfs)
        ref_width = tm.compute_width(extrapolate_temps, ref_finfs)

        ## Plot melting temperature
        rev_finfs = jnp.flip(ref_finfs)
        rev_temps = jnp.flip(extrapolate_temps)
        finfs_extrap = jnp.arange(0.1, 1., 0.05)
        temps_extrap = jnp.interp(finfs_extrap, rev_finfs, rev_temps)
        plt.plot(temps_extrap, finfs_extrap)
        plt.xlabel("T/K")
        plt.ylabel("Duplex Yield")
        plt.title(f"Tm={ref_tm}, width={ref_width}")
        plt.savefig(iter_dir / "melting_curve.png")
        plt.clf()


        end = time.time()
        analyze_time = end - start

        summary_str = ""
        summary_str += f"Simulation time: {sim_time}\n"
        summary_str += f"Analyze time: {analyze_time}\n"
        summary_str += f"Reference Tm: {ref_tm}\n"
        summary_str += f"Reference width: {ref_width}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        summary_str += f"Simulation finf: {sim_finf}\n"
        summary_str += f"Mean energy diff: {onp.mean(energy_diffs)}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        if not no_archive:
            zip_file(str(iter_dir / "output.dat"), str(iter_dir / "output.dat.zip"))
            os.remove(str(iter_dir / "output.dat"))

        return ref_states, ref_energies, jnp.array(all_ops), jnp.array(all_op_weights), iter_dir


    def loss_fn(params, ref_states, ref_energies, all_ops, all_op_weights):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_concentration)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        # FIXME: use weights appropriately below

        def compute_extrap_temp_finfs(t_kelvin_extrap):
            extrap_kt = utils.get_kt(t_kelvin_extrap)
            em_temp = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin_extrap, salt_conc=salt_concentration)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energy_fn_temp = jit(energy_fn_temp)

            def unbias_scan_fn(unb_counts, rs_idx):
                rs = ref_states[rs_idx]
                op = all_ops[rs_idx]
                op_weight = all_op_weights[rs_idx]

                # calc_energy = ref_energies[rs_idx] # this is wrong
                calc_energy = new_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT - calc_energy_temp/extrap_kt)

                difftre_weight = weights[rs_idx]
                # weighted_add_term = n_ref_states/difftre_weight * 1/op_weight * boltz_diff
                weighted_add_term = n_ref_states*difftre_weight * 1/op_weight * boltz_diff
                return unb_counts.at[op].add(weighted_add_term), None

            temp_unbiased_counts, _ = scan(unbias_scan_fn, jnp.zeros(n_bp+1), jnp.arange(n_ref_states))
            temp_finfs = tm.compute_finf(temp_unbiased_counts)
            return temp_finfs

        finfs = vmap(compute_extrap_temp_finfs)(extrapolate_temps)
        curr_tm = tm.compute_tm(extrapolate_temps, finfs)
        curr_width = tm.compute_width(extrapolate_temps, finfs)

        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
        aux = (curr_tm, curr_width, n_eff)
        rmse = jnp.sqrt((target_tm - curr_tm)**2)
        return rmse, aux
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    params = deepcopy(EMPTY_BASE_PARAMS)
    for opt_key in opt_keys:
        params[opt_key] = deepcopy(DEFAULT_BASE_PARAMS[opt_key])

    if optimizer_type == "adam":
        optimizer = optax.adam(learning_rate=lr)
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(learning_rate=lr)
    elif optimizer_type == "rmsprop":
        optimizer = optax.rmsprop(learning_rate=lr)
    else:
        raise RuntimeError(f"Invalid optimizer type: {optimizer_type}")
    opt_state = optimizer.init(params)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_tms = list()
    all_widths = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_tms = list()
    all_ref_widths = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, ref_ops, ref_op_weights, ref_iter_dir = get_ref_states(params, i=0, seed=key, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (loss, (curr_tm, curr_width, n_eff)), grads = grad_fn(params, ref_states, ref_energies, ref_ops, ref_op_weights)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_tms.append(curr_tm)
            all_ref_widths.append(curr_width)

        if n_eff < min_n_eff or num_resample_iters > max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()
            ref_states, ref_energies, ref_ops, ref_op_weights, ref_iter_dir = get_ref_states(params, i=i, seed=key+1+i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (curr_tm, curr_width, n_eff)), grads = grad_fn(params, ref_states, ref_energies, ref_ops, ref_op_weights)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_tms.append(curr_tm)
            all_ref_widths.append(curr_width)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(tm_path, "a") as f:
            f.write(f"{curr_tm}\n")
        with open(width_path, "a") as f:
            f.write(f"{curr_width}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(iter_params_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")

        all_losses.append(loss)
        all_n_effs.append(n_eff)
        all_tms.append(curr_tm)
        all_widths.append(curr_width)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0 and i:

            plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_tms, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_tms, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_tm, linestyle='--', label="Target Tm", color='red')
            plt.legend()
            plt.ylabel("Expected Tm")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"tms_iter{i}.png")
            plt.clf()

        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)
            onp.save(obj_dir / f"ref_losses_i{i}.npy", onp.array(all_ref_losses), allow_pickle=False)
            onp.save(obj_dir / f"ref_tms_i{i}.npy", onp.array(all_ref_tms), allow_pickle=False)
            onp.save(obj_dir / f"ref_widths_i{i}.npy", onp.array(all_ref_widths), allow_pickle=False)
            onp.save(obj_dir / f"tms_i{i}.npy", onp.array(all_tms), allow_pickle=False)
            onp.save(obj_dir / f"widths_i{i}.npy", onp.array(all_widths), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_losses.npy", onp.array(all_ref_losses), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_tms.npy", onp.array(all_ref_tms), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_widths.npy", onp.array(all_ref_widths), allow_pickle=False)
    onp.save(obj_dir / f"fin_tms.npy", onp.array(all_tms), allow_pickle=False)
    onp.save(obj_dir / f"fin_widths.npy", onp.array(all_widths), allow_pickle=False)


def get_parser():

    # Simulation arguments
    parser = argparse.ArgumentParser(description="Calculate melting temperature for an 8bp duplex")

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
    parser.add_argument('--temp', type=float, default=312.15,
                        help="Simulation temperature in Kelvin")
    parser.add_argument('--extrapolate-temps', nargs='+',
                        help='Temperatures for extrapolation in Kelvin in ascending order',
                        required=True)

    parser.add_argument('--salt-concentration', type=float, default=1.0,
                        help="Salt concentration in molar (M)")

    # Optimization arguments
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--target-tm', type=float, required=True,
                        help="Target melting temperature in Kelvin")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--optimizer-type', type=str,
                        default="adam", choices=["adam", "sgd", "rmsprop"])

    # Misc.
    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--no-archive', action='store_true')

    parser.add_argument(
        '--opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "hydrogen_bonding"],
        help='Parameter keys to optimize'
    )
    parser.add_argument('--save-obj-every', type=int, default=5,
                        help="Frequency of saving numpy files")
    parser.add_argument('--plot-every', type=int, default=1,
                        help="Frequency of plotting data from gradient descent epochs")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    assert(args['n_eq_steps'] == 0)

    run(args)
