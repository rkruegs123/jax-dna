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
import pprint
import functools

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad
import optax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import tm

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def hairpin_tm_running_avg(traj_hist_files, n_stem_bp, n_dist_thresholds):
    n_files = len(traj_hist_files)
    num_ops = 2
    n_skip_lines = 2 + num_ops

    # Open the first file to read relevant statistics
    with open(traj_hist_files[0], 'r') as f:
        repr_lines = f.readlines()

    assert(repr_lines[0][0] == "#")
    lines_per_hist = 1
    for l in repr_lines[1:]:
        if l[0] != '#':
            lines_per_hist += 1
        else:
            break

    n_lines = len(repr_lines)
    assert(n_lines % lines_per_hist == 0)
    n_hists = n_lines // lines_per_hist

    nvalues = lines_per_hist - 1
    ntemps = len(repr_lines[1].split()) - n_skip_lines # number of *extrapolated* temps

    ## extrapolated temperatures in celsius
    extrapolated_temps = [float(x) * 3000. - 273.15 for x in repr_lines[0].split()[-ntemps:]]
    extrapolated_temps = onp.array(extrapolated_temps)

    # Load contents of all files
    all_flines = list()
    for fname in traj_hist_files:
        with open(fname, 'r') as f:
            all_flines.append(f.readlines())

    # Compute running averages
    all_tms = list()
    all_widths = list()
    start_hist_idx = 50
    # start_hist_idx = 5
    assert(n_hists > start_hist_idx)
    for hist_idx in tqdm(range(start_hist_idx, n_hists), desc="Traj. histogram running avg."):
        start_line = hist_idx * lines_per_hist
        end_line = start_line + lines_per_hist

        # Construct a matrix of unbiased counts for each temperature and order parameter
        unbiased_counts = onp.zeros((ntemps, nvalues))
        for f_idx in range(n_files):
            f_hist_lines = all_flines[f_idx][start_line:end_line]
            for op_idx, op_line in enumerate(f_hist_lines[1:]): # ignore the header
                tokens = op_line.split()
                op_unbiased_temp_counts = onp.array([float(t) for t in tokens[n_skip_lines:]])
                unbiased_counts[:, op_idx] += op_unbiased_temp_counts


        unbound_op_idxs_extended = onp.array([n_stem_bp*d_idx for d_idx in range(n_dist_thresholds)])
        bound_op_idxs_extended = onp.array(list(range(1, 1+n_stem_bp)))

        unbound_unbiased_counts = unbiased_counts[:, unbound_op_idxs_extended]
        bound_unbiased_counts = unbiased_counts[:, bound_op_idxs_extended]

        ratios = list()
        for t_idx in range(len(extrapolated_temps)):
            unbound_count = unbound_unbiased_counts[t_idx].sum()
            bound_count = bound_unbiased_counts[t_idx].sum()

            ratio = bound_count / unbound_count
            ratios.append(ratio)
        ratios = onp.array(ratios)

        tm_ = tm.compute_tm(extrapolated_temps, ratios)
        width_ = tm.compute_width(extrapolated_temps, ratios)

        all_tms.append(tm_)
        all_widths.append(width_)

    return all_tms, all_widths


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

    n_iters = args['n_iters']
    lr = args['lr']
    target_tm = args['target_tm']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    optimizer_type = args['optimizer_type']

    stem_bp = args['stem_bp']
    loop_nt = args['loop_nt']

    # Load the system
    hairpin_basedir = Path("data/templates/hairpins")
    sys_basedir = hairpin_basedir / f"{stem_bp}bp_stem_{loop_nt}nt_loop"
    assert(sys_basedir.exists())
    conf_path_unbound = sys_basedir / "init_unbound.conf"
    conf_path_bound = sys_basedir / "init_bound.conf"
    top_path = sys_basedir / "sys.top"
    input_template_path = sys_basedir / "input"
    op_path = sys_basedir / "op.txt"
    wfile_path = sys_basedir / "wfile.txt"

    top_info = topology.TopologyInfo(top_path,
                                     # reverse_direction=False
                                     reverse_direction=True
    )
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n_nt = seq_oh.shape[0]
    assert(n_nt == 2*stem_bp + loop_nt)

    conf_info_unbound = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=conf_path_unbound,
        reverse_direction=True
    )
    conf_info_bound = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=conf_path_bound,
        reverse_direction=True
    )
    box_size = conf_info_bound.box_size

    displacement_fn, shift_fn = space.free()

    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    dt = 5e-3

    ## Process the weights information
    weights_df = pd.read_fwf(wfile_path, names=["op1", "op2", "weight"])
    num_ops = len(weights_df)
    n_stem_bp = len(weights_df.op1.unique())
    n_dist_thresholds = len(weights_df.op2.unique())
    pair2idx = dict()
    idx2pair = dict()
    idx2weight = dict()
    unbound_op_idxs = list()
    bound_op_idxs = list()
    for row_idx, row in weights_df.iterrows():
        op1 = int(row.op1)
        op2 = int(row.op2)
        pair2idx[(op1, op2)] = row_idx
        idx2pair[row_idx] = (op1, op2)
        idx2weight[row_idx] = row.weight

        if op1 == 0:
            unbound_op_idxs.append(row_idx)
        else:
            bound_op_idxs.append(row_idx)
    bound_op_idxs = onp.array(bound_op_idxs)
    unbound_op_idxs = onp.array(unbound_op_idxs)

    def compute_ratio(ub_counts):
        ub_unbiased_counts = ub_counts[unbound_op_idxs]
        ub_biased_counts = ub_counts[bound_op_idxs]

        return ub_biased_counts.sum() / ub_unbiased_counts.sum()


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

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

    max_seed_tries = 5
    seed_check_sample_freq = 10
    seed_check_steps = 100

    def get_ref_states(params, i, seed, prev_basedir):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

        procs = list()

        start = time.time()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")
            shutil.copy(wfile_path, repeat_dir / "wfile.txt")
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
                    reverse_direction=True
                    # reverse_direction=False
                )
                # conf_info_copy = center_configuration.center_conf(top_info, prev_lastconf_info)

            conf_info_copy.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)

            conf_info_copy.write(repeat_dir / "init.conf",
                                 # reverse=False,
                                 reverse=True,
                                 write_topology=False)


            check_seed_dir = repeat_dir / "check_seed"
            check_seed_dir.mkdir(parents=False, exist_ok=False)
            s_idx = 0
            valid_seed = None
            while s_idx < max_seed_tries and valid_seed is None:
                seed_try = random.randrange(10000)
                seed_dir = check_seed_dir / f"s{seed_try}"
                seed_dir.mkdir(parents=False, exist_ok=False)

                oxdna_utils.rewrite_input_file(
                    input_template_path, seed_dir,
                    temp=f"{t_kelvin}K", steps=seed_check_steps,
                    init_conf_path=str(repeat_dir / "init.conf"),
                    top_path=str(repeat_dir / "sys.top"),
                    save_interval=seed_check_sample_freq, seed=seed_try,
                    equilibration_steps=0,
                    no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str,
                    weights_file=str(repeat_dir / "wfile.txt"),
                    op_file=str(repeat_dir / "op.txt"),
                    log_file=str(repeat_dir / "sim.log"),
                )

                seed_proc = subprocess.Popen([oxdna_exec_path, seed_dir / "input"])
                seed_proc.wait()
                seed_rc = seed_proc.returncode

                if seed_rc == 0:
                    valid_seed = seed_try

                s_idx += 1

            if valid_seed is None:
                raise RuntimeError(f"Could not find valid seed.")


            oxdna_utils.rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=valid_seed,
                equilibration_steps=n_eq_steps,
                no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str,
                weights_file=str(repeat_dir / "wfile.txt"),
                op_file=str(repeat_dir / "op.txt"),
                log_file=str(repeat_dir / "sim.log"),
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

        # Analyze
        start = time.time()

        ## Combine the output files
        combine_cmd = "cat "
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {iter_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        ## Compute running avg. from `traj_hist.dat`
        all_traj_hist_fpaths = list()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")

        ### Note: the below should really go in its own loss file
        all_running_tms, all_running_widths = hairpin_tm_running_avg(
            all_traj_hist_fpaths, n_stem_bp, n_dist_thresholds)

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
            # reverse_direction=False)
            reverse_direction=True
        )
        ref_states = traj_info.get_states()
        assert(len(ref_states) == n_ref_states)
        ref_states = utils.tree_stack(ref_states)

        ## Load the oxDNA energies
        energy_df_columns = [
            "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot",
            "acc_ratio_vol", "op1", "op2", "op_weight"
        ]
        energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Load all last_hist.dat and combine
        last_hist_columns = ["num_bp", "dist_threshold_idx", "count_biased", "count_unbiased"] \
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
                orig_num_bp_col = deepcopy(last_hist_df['num_bp'])
                orig_dist_threshold_idx_col = deepcopy(last_hist_df['dist_threshold_idx'])
            else:
                last_hist_df = last_hist_df.add(repeat_last_hist_df)
        last_hist_df['num_bp'] = orig_num_bp_col
        last_hist_df['dist_threshold_idx'] = orig_dist_threshold_idx_col

        ## Check energies
        em_base = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em_base.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
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
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ### First, the periodic counts derived from the energy file(s)
        count_df = energy_df.groupby(['op1', 'op2', 'op_weight']).size().reset_index().rename(columns={0: "count"})
        op_names = list()
        op_weights = list()
        op_counts_periodic = list()
        for row_idx, row in weights_df.iterrows():
            op1 = int(row.op1)
            op2 = int(row.op2)
            op_name = f"({op1}, {op2})"

            count_row = count_df[(count_df.op1 == op1) & (count_df.op2 == op2)]
            if count_row.empty:
                op_count = 0
            else:
                op_count = count_row['count'].to_numpy()[0]

            op_names.append(op_name)
            op_counts_periodic.append(op_count)
            op_weights.append(row.weight)
        op_counts_periodic = onp.array(op_counts_periodic)
        op_weights = onp.array(op_weights)

        sns.barplot(x=op_names, y=op_counts_periodic, ax=ax[0])
        ax[0].set_title("Periodic Counts")
        ax[0].set_xlabel("O.P.")

        ### Then, the frequent counts from the histogram
        op_counts_frequent = list()
        for row_idx, row in weights_df.iterrows():
            op1 = int(row.op1)
            op2 = int(row.op2)

            last_hist_row = last_hist_df[(last_hist_df.num_bp == op1) \
                                         & (last_hist_df.dist_threshold_idx == op2)]
            assert(not last_hist_row.empty)
            op_count = last_hist_row['count_biased'].to_numpy()[0]

            op_counts_frequent.append(op_count)
        op_counts_frequent = onp.array(op_counts_frequent)

        sns.barplot(x=op_names, y=op_counts_frequent, ax=ax[1])
        ax[1].set_title("Frequent Counts")

        plt.savefig(iter_dir / "biased_counts.png")
        plt.clf()



        ## Unbias reference counts
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        op_counts_periodic_unbiased = op_counts_periodic / op_weights
        sns.barplot(x=op_names, y=op_counts_periodic_unbiased, ax=ax[0])
        ax[0].set_title(f"Periodic Counts, Reference T={t_kelvin}K")
        ax[0].set_xlabel("O.P.")

        op_counts_frequent_unbiased = op_counts_frequent / op_weights

        sns.barplot(x=op_names, y=op_counts_frequent_unbiased, ax=ax[1])
        ax[1].set_title(f"Frequent Counts, Reference T={t_kelvin}K")

        plt.savefig(iter_dir / "unbiased_counts.png")
        plt.clf()


        ## Unbias counts for each temperature
        all_ops = list(zip(energy_df.op1.to_numpy(), energy_df.op2.to_numpy()))
        all_unbiased_counts = list()
        all_unbiased_counts_ref = list()
        for extrap_t_kelvin, extrap_kt in zip(extrapolate_temps, extrapolate_kts):
            em_temp = model.EnergyModel(displacement_fn, params, t_kelvin=extrap_t_kelvin)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energy_fn_temp = jit(energy_fn_temp)

            temp_unbiased_counts = onp.zeros(num_ops)
            for rs_idx in tqdm(range(n_ref_states), desc=f"Extrapolating to {extrap_t_kelvin}K"):
                rs = ref_states[rs_idx]
                op1, op2 = all_ops[rs_idx]
                op_idx = pair2idx[(op1, op2)]
                op_weight = idx2weight[int(op_idx)]

                calc_energy = ref_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT - calc_energy_temp/extrap_kt)
                temp_unbiased_counts[op_idx] += 1/op_weight * boltz_diff

            all_unbiased_counts.append(temp_unbiased_counts)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(x=op_names, y=temp_unbiased_counts, ax=ax[0])
            ax[0].set_title(f"Periodic Counts, T={extrap_t_kelvin}K")
            ax[0].set_xlabel("O.P.")

            # Get the frequent counts at the extrapolated temp. from the oxDNA histogram
            frequent_extrap_counts = list()
            for row_idx, row in weights_df.iterrows():
                op1 = int(row.op1)
                op2 = int(row.op2)

                last_hist_row = last_hist_df[(last_hist_df.num_bp == op1) \
                                             & (last_hist_df.dist_threshold_idx == op2)]
                frequent_extrap_counts.append(last_hist_row[str(extrap_t_kelvin)].to_numpy()[0])
            frequent_extrap_counts = onp.array(frequent_extrap_counts)
            all_unbiased_counts_ref.append(frequent_extrap_counts)

            sns.barplot(x=op_names, y=frequent_extrap_counts, ax=ax[1])
            ax[1].set_title(f"Frequent Counts, T={extrap_t_kelvin}K")

            plt.savefig(iter_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.clf()

        all_unbiased_counts = onp.array(all_unbiased_counts)
        all_unbiased_counts_ref = onp.array(all_unbiased_counts_ref)


        # Compute the final Tms and widths

        unbound_unbiased_counts = all_unbiased_counts[:, unbound_op_idxs]
        bound_unbiased_counts = all_unbiased_counts[:, bound_op_idxs]

        ratios = list()
        for t_idx in range(len(extrapolate_temps)):
            unbound_count = unbound_unbiased_counts[t_idx].sum()
            bound_count = bound_unbiased_counts[t_idx].sum()

            ratio = bound_count / unbound_count
            ratios.append(ratio)
        ratios = onp.array(ratios)

        calc_tm = tm.compute_tm(extrapolate_temps, ratios)
        calc_width = tm.compute_width(extrapolate_temps, ratios)



        unbound_unbiased_counts_ref = all_unbiased_counts_ref[:, unbound_op_idxs]
        bound_unbiased_counts_ref = all_unbiased_counts_ref[:, bound_op_idxs]
        ratios_ref = list()
        for t_idx in range(len(extrapolate_temps)):
            unbound_count_ref = unbound_unbiased_counts_ref[t_idx].sum()
            bound_count_ref = bound_unbiased_counts_ref[t_idx].sum()

            ratio_ref = bound_count_ref / unbound_count_ref
            ratios_ref.append(ratio_ref)
        ratios_ref = onp.array(ratios_ref)

        calc_tm_ref = tm.compute_tm(extrapolate_temps, ratios_ref)
        calc_width_ref = tm.compute_width(extrapolate_temps, ratios_ref)

        end = time.time()
        analyze_time = end - start

        summary_str = ""
        summary_str += f"Simulation time: {sim_time}\n"
        summary_str += f"Analyze time: {analyze_time}\n"
        summary_str += f"Reference Tm: {calc_tm_ref}\n"
        summary_str += f"Reference width: {calc_width_ref}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        all_op_weights = list()
        all_op_idxs = list()
        for op1, op2 in all_ops:
            op_idx = pair2idx[(op1, op2)]
            op_weight = idx2weight[int(op_idx)]
            all_op_weights.append(op_weight)
            all_op_idxs.append(op_idx)

        all_ops = jnp.array(all_ops).astype(jnp.int32)
        all_op_weights = jnp.array(all_op_weights)
        all_op_idxs = jnp.array(all_op_idxs)

        return ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs, iter_dir


    def loss_fn(params, ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

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


        def compute_extrap_temp_ratios(t_kelvin_extrap):
            extrap_kt = utils.get_kt(t_kelvin_extrap)
            em_temp = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin_extrap)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energy_fn_temp = jit(energy_fn_temp)

            def unbias_scan_fn(unb_counts, rs_idx):
                rs = ref_states[rs_idx]
                op1, op2 = all_ops[rs_idx]
                op_idx = all_op_idxs[rs_idx]
                op_weight = all_op_weights[rs_idx]

                # calc_energy = ref_energies[rs_idx] # this is wrong
                calc_energy = new_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT - calc_energy_temp/extrap_kt)

                difftre_weight = weights[rs_idx]
                # weighted_add_term = n_ref_states/difftre_weight * 1/op_weight * boltz_diff
                weighted_add_term = n_ref_states*difftre_weight * 1/op_weight * boltz_diff
                return unb_counts.at[op_idx].add(weighted_add_term), None

            temp_unbiased_counts, _ = scan(unbias_scan_fn, jnp.zeros(num_ops), jnp.arange(n_ref_states))
            temp_ratios = compute_ratio(temp_unbiased_counts)
            return temp_ratios

        ratios = vmap(compute_extrap_temp_ratios)(extrapolate_temps)
        curr_tm = tm.compute_tm(extrapolate_temps, ratios)
        curr_width = tm.compute_width(extrapolate_temps, ratios)

        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
        aux = (curr_tm, curr_width, n_eff)
        rmse = jnp.sqrt((target_tm - curr_tm)**2)
        return rmse, aux
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    # params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]
    params["hydrogen_bonding"] = model.DEFAULT_BASE_PARAMS["hydrogen_bonding"]


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
    ref_states, ref_energies, ref_ops, ref_op_weights, ref_op_idxs, ref_iter_dir = get_ref_states(params, i=0, seed=key, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")


    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (loss, (curr_tm, curr_width, n_eff)), grads = grad_fn(params, ref_states, ref_energies, ref_ops, ref_op_weights, ref_op_idxs)
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
            ref_states, ref_energies, ref_ops, ref_op_weights, ref_op_idxs, ref_iter_dir = get_ref_states(params, i=i, seed=key+1+i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (curr_tm, curr_width, n_eff)), grads = grad_fn(params, ref_states, ref_energies, ref_ops, ref_op_weights, ref_op_idxs)

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


def get_parser():

    parser = argparse.ArgumentParser(description="Calculate melting temperature for a given hairpin")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu"])
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-threads', type=int, default=2,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=int(5e6),
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=int(1e3),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--temp', type=float, default=330.15,
                        help="Temperature in kelvin")
    parser.add_argument('--extrapolate-temps', nargs='+',
                        help='Temperatures for extrapolation in Kelvin in ascending order',
                        required=True)

    parser.add_argument('--stem-bp', type=int, default=6,
                        help="Number of base pairs comprising the stem")
    parser.add_argument('--loop-nt', type=int, default=6,
                        help="Number of nucleotides comprising the loop")


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

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
