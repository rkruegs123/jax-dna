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
    n_sims = 28
    n_steps_per_sim = 50000000
    n_eq_steps = 0
    sample_every = 250000
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    t_kelvin = 312.15
    extrapolate_temps = jnp.array([300.15, 303.15, 306.15, 309.15, 315.15, 318.15, 321.15, 324.15, 327.15, 330.15, 333.15, 336.15, 339.15])
    assert(jnp.all(extrapolate_temps[:-1] <= extrapolate_temps[1:])) # check that temps. are sorted
    n_extrap_temps = len(extrapolate_temps)
    extrapolate_kts = vmap(utils.get_kt)(extrapolate_temps)
    extrapolate_temp_str = ', '.join([f"{tc}K" for tc in extrapolate_temps])
    salt_concentration = 0.5




    # Setup the logging directory


    # Load the system
    sys_basedir = Path("data/templates/tm-8bp")
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

    def analyze(params, i, seed, prev_basedir):
        iter_dir = Path("/home/ryan/Downloads/test-tm/iter0")

        ## Compute running avg. from all `traj_hist.dat`
        all_traj_hist_fpaths = list()
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")
        all_running_tms, all_running_widths = tm.traj_hist_running_avg_1d(all_traj_hist_fpaths)

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
        count_df = energy_df.groupby(['op', 'op_weight']).size().reset_index().rename(columns={0: "count"})
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x=count_df['op'], y=count_df['count'], ax=ax[0])
        ax[0].set_title("Periodic Counts")
        ax[0].set_xlabel("num_bp")
        sns.barplot(x=last_hist_df['num_bp'], y=last_hist_df['count_biased'], ax=ax[1])
        ax[1].set_title("Frequent Counts")

        plt.savefig(iter_dir / "biased_counts.png")
        plt.close()


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
        plt.close()

        sim_finf = tm.compute_finf(ref_unbiased_counts)

        ## Unbias counts for each temperature
        all_unbiased_counts = list()
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
            # plt.savefig(iter_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.close()

        ## Compute discrete running avg. Tm
        """
        running_avg_iters = list()
        running_avg_tms = list()
        for running_avg_iter, running_avg_unb_counts in running_avg_mapper.items():
            iter_unb_counts = onp.array(running_avg_unb_counts)
            iter_discrete_finfs = vmap(tm.compute_finf)(iter_unb_counts)
            iter_tm = tm.compute_tm(extrapolate_temps, iter_discrete_finfs)

            running_avg_iters.append(running_avg_iter)
            running_avg_tms.append(iter_tm)

        plt.plot(running_avg_iters, running_avg_tms, '-o')
        plt.show()
        # plt.savefig(iter_dir / f"discrete_running_avg_tm.png")
        plt.close()
        """

        ## Compute final Tms and widths, log all, as well as time
        all_unbiased_counts = onp.array(all_unbiased_counts) # (ntemps, n_bp+1)
        discrete_finfs = vmap(tm.compute_finf)(all_unbiased_counts)

        calc_tm = tm.compute_tm(extrapolate_temps, discrete_finfs)
        calc_width = tm.compute_width(extrapolate_temps, discrete_finfs)

        last_hist_extrap_counts = last_hist_df.to_numpy()[:, -n_extrap_temps:].T # (ntemps, n_bp+1)
        ref_finfs = vmap(tm.compute_finf)(last_hist_extrap_counts)

        pdb.set_trace()

        ref_tm = tm.compute_tm(extrapolate_temps, ref_finfs)
        ref_width = tm.compute_width(extrapolate_temps, ref_finfs)

        pdb.set_trace()

        ## Plot melting temperature
        rev_finfs = jnp.flip(ref_finfs)
        rev_temps = jnp.flip(extrapolate_temps)
        finfs_extrap = jnp.arange(0.1, 1., 0.05)
        temps_extrap = jnp.interp(finfs_extrap, rev_finfs, rev_temps)
        plt.plot(temps_extrap, finfs_extrap, label="Reference")

        rev_finfs = jnp.flip(discrete_finfs)
        temps_extrap = jnp.interp(finfs_extrap, rev_finfs, rev_temps)
        plt.plot(temps_extrap, finfs_extrap, label="Discrete")


        plt.xlabel("T/K")
        plt.ylabel("Duplex Yield")
        plt.title(f"Tm={ref_tm}, width={ref_width}")
        plt.legend()
        # plt.savefig(iter_dir / "melting_curve.png")
        plt.show()
        plt.close()


        pdb.set_trace()

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

        return

    # Initialize parameters
    params = deepcopy(model2.EMPTY_BASE_PARAMS)
    analyze(params, i=0, seed=0, prev_basedir=None)


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

    parser.add_argument('--opt-width', action='store_true')
    parser.add_argument('--target-width', type=float, default=14.0,
                        help="Target width of melting temperature curve in Kelvin")
    parser.add_argument('--no-opt-tm', action='store_true')


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    assert(args['n_eq_steps'] == 0)

    run(args)
