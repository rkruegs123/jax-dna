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

import jax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import tm

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


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

    # Load the system
    sys_basedir = Path("data/templates/tm-8bp")
    input_template_path = sys_basedir / "input"

    weight_path = sys_basedir / "wfile.txt"
    weight_df = pd.read_csv(weight_path, delim_whitespace=True, names=["op", "weight"])
    weight_mapper = dict(zip(weight_df.op, weight_df.weight))

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path,
                                     # reverse_direction=False
                                     reverse_direction=True
    )
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] % 2 == 0)
    n_bp = seq_oh.shape[0] // 2

    conf_path_bound = sys_basedir / "init_bound.conf"
    conf_info_bound = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path_bound,
        reverse_direction=True
        # reverse_direction=False
    )
    conf_path_unbound = sys_basedir / "init_unbound.conf"
    conf_info_unbound = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path_unbound,
        reverse_direction=True
        # reverse_direction=False
    )
    box_size = conf_info_bound.box_size

    weights_path = sys_basedir / "wfile.txt"
    op_path = sys_basedir / "op.txt"

    displacement_fn, shift_fn = space.periodic(box_size)
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT

    def get_ref_states(params, i, seed):
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
            shutil.copy(weights_path, repeat_dir / "wfile.txt")
            shutil.copy(op_path, repeat_dir / "op.txt")

            if r % 2 == 0:
                conf_info_copy = deepcopy(conf_info_bound)
            else:
                conf_info_copy = deepcopy(conf_info_unbound)
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
                weights_file=str(repeat_dir / "wfile.txt"), op_file=str(repeat_dir / "op.txt")
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
            # reverse_direction=False)
            reverse_direction=True
        )
        ref_states = traj_info.get_states()
        n_ref_states = len(ref_states)
        ref_states = utils.tree_stack(ref_states)

        ## Load the oxDNA energies
        energy_df_columns = [
            "time", "potential_energy", "kinetic_energy", "total_energy",
            "op_idx", "op", "op_weight"
        ] # NOTE: this is wrong. the columns are different for VMMC than for MD! And op_idx is not a thing!
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
        for rs_idx in tqdm(range(n_ref_states)):
            op = all_ops[rs_idx]
            op_weight = weight_mapper[op]
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

        ## Unbias counts for each temperature
        all_unbiased_counts = list()
        for extrap_t_kelvin, extrap_kt in zip(extrapolate_temps, extrapolate_kts):
            em_temp = model.EnergyModel(displacement_fn, params, t_kelvin=extrap_t_kelvin)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
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


        ## FIXME: compute final Tms and widths, log all, as well as time
        all_unbiased_counts = onp.array(all_unbiased_counts) # (ntemps, n_bp+1)
        discrete_finfs = vmap(tm.compute_finf)(all_unbiased_counts)

        calc_tm = tm.compute_tm(extrapolate_temps, discrete_finfs)
        calc_width = tm.compute_width(extrapolate_temps, discrete_finfs)

        last_hist_extrap_counts = last_hist_df.to_numpy()[:, -n_extrap_temps:].T # (ntemps, n_bp+1)
        ref_finfs = vmap(tm.compute_finf)(last_hist_extrap_counts)

        ref_tm = tm.compute_tm(extrapolate_temps, ref_finfs)
        ref_width = tm.compute_width(extrapolate_temps, ref_finfs)

        end = time.time()
        analyze_time = end - start

        summary_str = ""
        summary_str += f"Simulation time: {sim_time}\n"
        summary_str += f"Analyze time: {analyze_time}\n"
        summary_str += f"Reference Tm: {ref_tm}\n"
        summary_str += f"Reference width: {ref_width}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        return ref_states, ref_energies, all_ops


    # Initialize parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

    get_ref_states(params, i=0, seed=key)


def get_parser():
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

    return parser


if __name__ == "__main__":
    # Example: python3 -m experiments.simulate_tm_8bp_hybrid --run-name tm-sim-test --oxdna-path /home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA --temp 312.15 --extrapolate-temps 300.15 303.15 306.15 309.15 315.15 318.15 321.15 324.15 327.15 330.15 333.15 336.15 339.15
    parser = get_parser()
    args = vars(parser.parse_args())

    # FIXME: otherwise, restart_step_counter must be true. Could do, but annoying. Maybe not an issue if we sample long enough. Note that we think we're equilibrating for Lp when we're not, but maybe not an issue because we reuse old states from reweighting. Could do same thing
    assert(args['n_eq_steps'] == 0)

    run(args)
