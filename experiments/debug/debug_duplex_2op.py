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

import jax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad
import optax

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model as model2
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.dna2.oxdna_utils import recompile_oxdna
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


def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))

def run(args):

    # Load parameters
    key = 0
    n_sims = 34
    n_steps_per_sim = 50000000
    n_eq_steps = 0
    sample_every = 250000
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = 6800
    t_kelvin = 312.15
    extrapolate_temps = jnp.array([300.15, 303.15, 306.15, 309.15, 315.15, 318.15, 321.15, 324.15, 327.15, 330.15, 333.15, 336.15, 339.15])
    assert(jnp.all(extrapolate_temps[:-1] <= extrapolate_temps[1:])) # check that temps. are sorted
    n_extrap_temps = len(extrapolate_temps)
    extrapolate_kts = vmap(utils.get_kt)(extrapolate_temps)
    extrapolate_temp_str = ', '.join([f"{tc}K" for tc in extrapolate_temps])
    salt_concentration = 0.25

    target_tm = 317.0



    # Load the system
    sys_basedir = Path("data/templates/tm-8bp-2op")
    input_template_path = sys_basedir / "input"
    wfile_path = sys_basedir / "wfile.txt"

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

    ## Process the weights information
    weights_df = pd.read_fwf(wfile_path, names=["op1", "op2", "weight"])
    num_ops = len(weights_df)
    assert(len(weights_df.op1.unique()) == n_bp+1)
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

    unbound_op_idxs_extended = onp.array([(1+n_bp)*d_idx for d_idx in range(n_dist_thresholds)])
    bound_op_idxs_extended = onp.array(list(range(1, 1+n_bp)))

    """
    # unbound_op_idxs_extended = onp.array([(1+n_bp)*d_idx for d_idx in range(n_dist_thresholds)])
    # bound_op_idxs_extended = onp.array(list(range(1, 1+n_bp)))

    unbound_op_idxs_extended = list()
    bound_op_idxs_extended = list()
    for (op1, op2), op_idx in pair2idx.items():
        if op1 == 0:
            unbound_op_idxs_extended.append(op_idx)
        else:
            bound_op_idxs_extended.append(op_idx)

    unbound_op_idxs_extended = onp.array(unbound_op_idxs_extended)
    bound_op_idxs_extended = onp.array(bound_op_idxs_extended)
    """

    def process_ops(unbiased_counts, extended=False):

        # Note: below wasn't working
        # unbiased_unbound_counts = unbiased_counts[:, unbound_op_idxs_extended]
        # unbiased_bound_counts = unbiased_counts[:, bound_op_idxs_extended]
        # return jnp.concatenate([jnp.array([unbiased_unbound_counts.sum(axis=0)]), unbiased_bound_counts])

        # unbiased_unbound_count = unbiased_counts[unbound_op_idxs_extended].sum()
        # unbiased_bound_counts = unbiased_counts[bound_op_idxs_extended]
        if extended:
            unbiased_unbound_count = unbiased_counts[unbound_op_idxs_extended].sum()
            unbiased_bound_counts = unbiased_counts[bound_op_idxs_extended]
        else:
            unbiased_unbound_count = unbiased_counts[unbound_op_idxs].sum()
            unbiased_bound_counts = unbiased_counts[bound_op_idxs]
        return jnp.concatenate([jnp.array([unbiased_unbound_count]), unbiased_bound_counts])

    def get_ref_states(params, i, seed, prev_basedir):
        random.seed(seed)

        iter_dir = Path("/home/ryan/Downloads/test-tm-op2/opt-tm-hb-adam-oxdna2-2op-test/ref_traj/iter0")


        # Analyze

        ## Compute running avg. from all `traj_hist.dat`
        all_traj_hist_fpaths = list()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")
        all_running_tms, all_running_widths = tm.traj_hist_running_avg_2d(all_traj_hist_fpaths, n_bp, n_dist_thresholds,
                                                                          start_hist_idx=10) # for debugging

        plt.plot(all_running_tms)
        plt.xlabel("Iteration")
        plt.ylabel("Tm (C)")
        # plt.savefig(iter_dir / "traj_hist_running_tm.png")
        # plt.show()
        plt.close()

        plt.plot(all_running_widths)
        plt.xlabel("Iteration")
        plt.ylabel("Width (C)")
        # plt.savefig(iter_dir / "traj_hist_running_width.png")
        # plt.show()
        plt.close()

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

        pdb.set_trace()

        ## Check energies
        em_base = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_concentration)
        energy_fn = lambda body: em_base.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T,
            is_end=top_info.is_end
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
        # plt.savefig(iter_dir / "energies.png")
        plt.close()

        sns.histplot(energy_diffs)
        # plt.savefig(iter_dir / "energy_diffs.png")
        # plt.show()
        plt.close()


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

        # plt.show()
        # plt.savefig(iter_dir / "biased_counts.png")
        plt.close()


        ## Unbias reference counts
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        op_counts_periodic_unbiased = op_counts_periodic / op_weights
        sns.barplot(x=op_names, y=op_counts_periodic_unbiased, ax=ax[0])
        ax[0].set_title(f"Periodic Counts, Reference T={t_kelvin}K")
        ax[0].set_xlabel("O.P.")

        op_counts_frequent_unbiased = op_counts_frequent / op_weights

        sns.barplot(x=op_names, y=op_counts_frequent_unbiased, ax=ax[1])
        ax[1].set_title(f"Frequent Counts, Reference T={t_kelvin}K")

        # plt.show()
        # plt.savefig(iter_dir / "unbiased_counts.png")
        plt.close()







        ## Unbias counts for each temperature
        all_unbiased_counts = list()
        all_unbiased_counts_ref = list()
        all_ops = list(zip(energy_df.op1.to_numpy(), energy_df.op2.to_numpy()))
        running_avg_min = 250
        running_avg_freq = 50
        running_avg_mapper = dict()
        for extrap_t_kelvin, extrap_kt in zip(extrapolate_temps, extrapolate_kts):
            em_temp = model2.EnergyModel(displacement_fn, params, t_kelvin=extrap_t_kelvin, salt_conc=salt_concentration)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T,
                is_end=top_info.is_end
            )
            energy_fn_temp = jit(energy_fn_temp)

            temp_unbiased_counts = onp.zeros(10)
            for rs_idx in tqdm(range(n_ref_states), desc=f"Extrapolating to {extrap_t_kelvin}K"):
                rs = ref_states[rs_idx]
                op1, op2 = all_ops[rs_idx]
                op_idx = pair2idx[(op1, op2)]
                op_weight = idx2weight[int(op_idx)]

                calc_energy = ref_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT - calc_energy_temp/extrap_kt)
                temp_unbiased_counts[op_idx] += 1/op_weight * boltz_diff


                if rs_idx >= running_avg_min and rs_idx % running_avg_freq == 0:
                    if rs_idx not in running_avg_mapper:
                        running_avg_mapper[rs_idx] = [deepcopy(temp_unbiased_counts)]
                    else:
                        running_avg_mapper[rs_idx].append(deepcopy(temp_unbiased_counts))


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

            # plt.show()
            # plt.savefig(iter_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.close()

        ## Compute discrete running avg. Tm
        """
        running_avg_iters = list()
        running_avg_tms = list()
        for running_avg_iter, running_avg_unb_counts in running_avg_mapper.items():
            # iter_unb_counts = process_ops(onp.array(running_avg_unb_counts))
            iter_unb_counts = vmap(process_ops)(jnp.array(running_avg_unb_counts))

            iter_discrete_finfs = vmap(tm.compute_finf)(iter_unb_counts)
            iter_tm = tm.compute_tm(extrapolate_temps, iter_discrete_finfs)

            running_avg_iters.append(running_avg_iter)
            running_avg_tms.append(iter_tm)

        plt.plot(running_avg_iters, running_avg_tms, '-o')
        plt.savefig(iter_dir / f"discrete_running_avg_tm.png")
        plt.close()
        """

        ## Compute final Tms and widths, log all, as well as time
        pdb.set_trace()
        # all_unbiased_counts = process_ops(onp.array(all_unbiased_counts))
        all_unbiased_counts = vmap(process_ops)(jnp.array(all_unbiased_counts))
        discrete_finfs = vmap(tm.compute_finf)(all_unbiased_counts)

        calc_tm = tm.compute_tm(extrapolate_temps, discrete_finfs)
        calc_width = tm.compute_width(extrapolate_temps, discrete_finfs)

        last_hist_extrap_counts = last_hist_df.to_numpy()[:, -n_extrap_temps:].T
        last_hist_extrap_counts_processed = vmap(process_ops, (0, None))(last_hist_extrap_counts, True)
        ref_finfs = vmap(tm.compute_finf)(last_hist_extrap_counts_processed)

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
        # plt.savefig(iter_dir / "melting_curve.png")
        plt.close()


        end = time.time()
        analyze_time = end - start

        summary_str = ""
        summary_str += f"Simulation time: {sim_time}\n"
        summary_str += f"Analyze time: {analyze_time}\n"
        summary_str += f"Reference Tm: {ref_tm}\n"
        summary_str += f"Reference width: {ref_width}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        # with open(iter_dir / "summary.txt", "w+") as f:
        #     f.write(summary_str)


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

        plt.plot(all_op_idxs)
        for i in range(n_sims):
            plt.axvline(x=i*n_ref_states_per_sim, linestyle="--", color="red")
        # plt.savefig(iter_dir / "op_trajectory.png")
        plt.close()

        return ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs, iter_dir


    def loss_fn(params, ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs):
        em = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin, salt_conc=salt_concentration)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T,
                                              is_end=top_info.is_end)
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
            em_temp = model2.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin_extrap, salt_conc=salt_concentration)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T,
                is_end=top_info.is_end)
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
            temp_unbiased_counts_processed = process_ops(temp_unbiased_counts)
            temp_finf = tm.compute_finf(temp_unbiased_counts_processed)
            return temp_finf

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
    params = deepcopy(model2.EMPTY_BASE_PARAMS)
    ref_states, ref_energies, ref_ops, ref_op_weights, ref_op_idxs, ref_iter_dir = get_ref_states(params, i=0, seed=key, prev_basedir=None)


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


    parser.add_argument('--salt-concentration', type=float, default=0.25,
                        help="Salt concentration in molar (M)")

    # Optimization arguments
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
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
