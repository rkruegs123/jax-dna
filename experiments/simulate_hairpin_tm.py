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

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import tm

from jax.config import config
config.update("jax_enable_x64", True)


def hairpin_tm_running_avg(traj_hist_fpaths, n_stem_bp, n_dist_thresholds):
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
    # start_hist_idx = 50
    start_hist_idx = 5
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


        unbound_op_idxs = onp.array([n_stem_bp*d_idx for d_idx in range(n_dist_thresholds)])
        bound_op_idxs = onp.array(list(range(1, 1+n_stem_bp)))

        unbound_unbiased_counts = unbiased_counts[:, unbound_op_idxs]
        bound_unbiased_counts = unbiased_counts[:, bound_op_idxs]

        ratios = list()
        for t_idx in range(len(extrapolated_temps)):
            unbound_count = unbound_unbiased_counts[t_idx].sum()
            bound_count = bound_unbiased_counts[t_idx].sum()

            ratio = bound_count / unbound_count
            ratios.append(ratio)
        ratios = onp.array(ratios)

        tm = tm.compute_tm(extrapolated_temps, ratios)
        width = tm.compute_width(extrapolated_temps, ratios)

        all_tms.append(tm)
        all_widths.append(width)

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
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    t_kelvin = args['temp']
    extrapolate_temps = jnp.array([float(et) for et in args['extrapolate_temps']]) # in Kelvin
    assert(jnp.all(extrapolate_temps[:-1] <= extrapolate_temps[1:])) # check that temps. are sorted
    n_extrap_temps = len(extrapolate_temps)
    extrapolate_kts = vmap(utils.get_kt)(extrapolate_temps)
    extrapolate_temp_str = ', '.join([f"{tc}K" for tc in extrapolate_temps])


    stem_bp = args['stem_bp']
    loop_nt = args['loop_nt']

    # Load the system
    hairpin_basedir = Path("data/templates/hairpins")
    sys_basedir = hairpin_basedir / f"{stem_bp}bp_stem_{loop_nt}nt_loop"
    assert(sys_dir.exists())
    init_conf_path = sys_dir / "init.conf"
    top_path = sys_dir / "sys.top"
    input_template_path = sys_dir / "input"
    op_path = sys_dir / "op.txt"
    wfile_path = sys_dir / "wfile.txt"

    top_info = topology.TopologyInfo(top_path,
                                     # reverse_direction=False
                                     reverse_direction=True
    )
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n_nt = seq_oh.shape[0]
    assert(n_nt == 2*stem_bp + loop_nt)

    conf_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=init_conf_path,
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
    unbound_op_idxs = onp.array([n_stem_bp*d_idx for d_idx in range(n_dist_thresholds)])
    bound_op_idxs = onp.array(list(range(1, 1+n_stem_bp)))
    bins = np.arange(num_ops + 1) - 0.5
    pair2idx = dict()
    idx2pair = dict()
    idx2weight = dict()
    for row_idx, row in weights_df.iterrows():
        op1 = int(row.op1)
        op2 = int(row.op2)
        pair2idx[(op1, op2)] = row_idx
        idx2pair[row_idx] = (op1, op2)
        idx2weight[row_idx] = row.weight

    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    def get_ref_states(params, i, seed):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        # oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

        procs = list()

        start = time.time()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")
            shutil.copy(weights_path, repeat_dir / "wfile.txt")
            shutil.copy(op_path, repeat_dir / "op.txt")

            conf_info_copy = deepcopy(conf_info)
            conf_info_copy.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)

            conf_info_copy.write(repeat_dir / "init.conf",
                                 # reverse=False,
                                 reverse=True,
                                 write_topology=False)

            oxdna_utils.rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(100),
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

        pdb.set_trace()

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

        pdb.set_trace()

        ## Load states from oxDNA simulation
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / "output.dat",
            # reverse_direction=False)
            reverse_direction=True
        )
        ref_states = traj_info.get_states()
        n_ref_states = len(ref_states)
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
            else:
                last_hist_df = last_hist_df.add(repeat_last_hist_df)

        pdb.set_trace()

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

        pdb.set_trace()

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
            op_name = f"({row.op1}, {row.op2})"

            count_row = count_df[(count_df.op1 == op1) & (count_df.op2 == op2)]
            if not count_row:
                op_count = 0
            else:
                op_count = count_row.count

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
            op_name = f"({row.op1}, {row.op2})"

            last_hist_row = last_hist_df[(last_hist_df.num_bp == op1) \
                                         & (last_hist_df.dist_threshold_idx == op2)]
            assert(last_hist_row)
            op_count = last_hist_row.count

            op_names.append(op_name)
            op_counts_frequent.append(op_count)
        op_counts_frequent = onp.array(op_counts_frequent)

        sns.barplot(x=op_names, y=op_counts_frequent, ax=ax[1])
        ax[1].set_title("Frequent Counts")

        plt.savefig(iter_dir / "biased_counts.png")
        plt.clf()


        pdb.set_trace()


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

        pdb.set_trace()


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
                op_weight = idx2weight[op_idx]

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
                frequent_extrap_counts.append(last_hist_row[str(extrap_t_kelvin)])
            frequent_extrap_counts = onp.array(frequent_extrap_counts)
            all_unbiased_counts_ref.append(frequent_extrap_counts)

            sns.barplot(x=op_names, y=frequent_extrap_counts, ax=ax[1])
            ax[1].set_title(f"Frequent Counts, T={extrap_t_kelvin}K")

            plt.savefig(iter_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.clf()

        all_unbiased_counts = onp.array(all_unbiased_counts)
        all_unbiased_counts_ref = onp.array(all_unbiased_counts_ref)

        pdb.set_trace()

        # Compute the final Tms and widths
        unbound_unbiased_counts = all_unbiased_counts[:, unbound_op_idxs]
        bound_unbiased_counts = all_unbiased_counts[:, bound_op_idxs]

        ratios = list()
        for t_idx in range(len(extrapolated_temps)):
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
        for t_idx in range(len(extrapolated_temps)):
            unbound_count_ref = unbound_unbiased_counts_ref[t_idx].sum()
            bound_count_ref = bound_unbiased_counts_ref[t_idx].sum()

            ratio_ref = bound_count_ref / unbound_count_ref
            ratios_ref.append(ratio_ref)
        ratios_ref = onp.array(ratios_ref)

        calc_tm_ref = tm.compute_tm(extrapolate_temps, ratios_ref)
        calc_width_ref = tm.compute_width(extrapolate_temps, ratios_ref)

        end = time.time()
        analyze_time = end - start

        pdb.set_trace()

        summary_str = ""
        summary_str += f"Simulation time: {sim_time}\n"
        summary_str += f"Analyze time: {analyze_time}\n"
        summary_str += f"Reference Tm: {calc_tm_ref}\n"
        summary_str += f"Reference width: {calc_width_ref}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        return ref_states, ref_energies, all_ops


    # Initialize parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    get_ref_states(params, i=0, seed=key)



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
    parser.add_argument('--n-eq-steps', type=int, default=int(1e5),
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

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
