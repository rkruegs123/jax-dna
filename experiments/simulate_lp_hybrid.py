import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import shutil
import shlex
import argparse
import pandas as pd
import random

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model, oxdna_utils

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 5
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))

# Compute the average persistence length

def run(args):
    # Load parameters
    device = args['device']
    n_threads = args['n_threads']
    key = args['key']
    n_sims = args['n_sims']
    # if device == "cpu":
    #     assert(n_sims == 1)
    if device == "gpu":
        raise NotImplementedError(f"Still need to implement GPU version...")
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    n_skipped_quartets = args['n_skipped_quartets']
    truncation = args['truncation']

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
    # FIXME: will have to add template options for GPU/CUDA-related things
    sys_basedir = Path("data/templates/persistence-length")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    # top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    quartets = utils.get_all_quartets(n_nucs_per_strand=seq_oh.shape[0] // 2)
    quartets = quartets[n_skipped_quartets:]
    quartets = quartets[:-n_skipped_quartets]
    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False

    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT


    # Do the simulation
    # FIXME: take care of CPU vs. GPU
    def get_ref_states(params, i, seed):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

        procs = list()

        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")

            conf_info_copy = deepcopy(centered_conf_info)
            conf_info_copy.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)

            conf_info_copy.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            oxdna_utils.rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(100),
                equilibration_steps=n_eq_steps, dt=dt,
                no_stdout_energy=0
            )

            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

        combine_cmd = "cat "
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {iter_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        """
        input_path = iter_dir / "input"

        oxdna_process = subprocess.run([oxdna_exec_path, input_path])
        rc = oxdna_process.returncode
        if rc != 0:
            raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
        """

        # Analyze

        ## Load states from oxDNA simulation
        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / "output.dat",
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        n_traj_states = len(traj_states)
        traj_states = utils.tree_stack(traj_states)

        ## Load the oxDNA energies

        # FIXME: take out first row when loading them in
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

        # Method 1: For loop
        """
        calc_energies = list()
        for ts_idx in tqdm(range(n_traj_states), desc="Calculating energies"):
            ts = traj_states[ts_idx]
            calc_energies.append(energy_fn(ts))
        calc_energies = jnp.array(calc_energies)
        """

        # Method 2: vmap
        # calc_energies = vmap(energy_fn)(traj_states)

        # Method 3: scan
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)


        # gt_energies = energy_df.iloc[1:, :].potential_energy.to_numpy() * seq_oh.shape[0]
        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        # atol_places = 3
        # tol = 10**(-atol_places)
        tol = 2.0
        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            print(f"State {i}:")
            print(f"\t- Calc. Energy: {calc}")
            print(f"\t- Reference. Energy: {gt}")
            diff = onp.abs(calc - gt)
            print(f"\t- Difference: {diff}")

            if diff >= tol:
                # pdb.set_trace()
                print(f"WARNING: energy difference of {diff}")


        # Compute the persistence lengths

        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(traj_states, quartets, base_site)
        mean_corr_curve = jnp.mean(unweighted_corr_curves, axis=0)
        mean_l0 = jnp.mean(unweighted_l0_avgs)
        # mean_Lp, _ = persistence_length.persistence_length_fit(mean_corr_curve, mean_l0)

        mean_Lp_truncated, offset = persistence_length.persistence_length_fit(mean_corr_curve[:truncation], mean_l0)

        compute_every = 10
        n_curves = unweighted_corr_curves.shape[0]
        all_inter_lps = list()
        all_inter_lps_truncated = list()
        for i in range(0, n_curves, compute_every):
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], mean_l0)
            all_inter_lps_truncated.append(inter_mean_Lp_truncated)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, mean_l0)
            all_inter_lps.append(inter_mean_Lp)

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps)
        plt.ylabel("Lp")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(iter_dir / "running_avg.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated)
        plt.ylabel("Lp")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(iter_dir / "running_avg_truncated.png")
        plt.clf()

        plt.plot(mean_corr_curve)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(iter_dir / "full_corr_curve.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "full_log_corr_curve.png")
        plt.clf()

        fit_fn = lambda n: -n * mean_l0 / mean_Lp_truncated + offset
        plt.plot(jnp.log(mean_corr_curve)[:truncation])
        neg_inverse_slope = mean_Lp_truncated / mean_l0 # in nucleotides
        rounded_offset = onp.round(offset, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation)), linestyle='--', label=fit_str)

        plt.title(f"Log-Correlation Curve, Truncated.")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "log_corr_curve.png")
        plt.clf()


        centered_conf_info.write(iter_dir / "init.conf", reverse=False, write_topology=False)
        shutil.copy(top_path, iter_dir / "sys.top")
        oxdna_utils.rewrite_input_file(
            input_template_path, iter_dir,
            temp=f"{t_kelvin}K", steps=n_steps_per_sim*n_sims,
            init_conf_path=str(iter_dir / "init.conf"),
            top_path=str(iter_dir / "sys.top"),
            save_interval=sample_every, seed=seed,
            equilibration_steps=n_eq_steps, dt=dt,
            no_stdout_energy=0
        )

        """
        subprocess_cmd = f"oat persistence_length -p {n_threads} -n {iter_dir}/oat_corr.png {iter_dir}/output.dat {iter_dir}/input 5 45"
        oat_result = subprocess.run(shlex.split(subprocess_cmd), capture_output=True, text=True)
        rc = oat_result.returncode
        if rc != 0:
            raise RuntimeError(f"oat analysis failed with error code: {rc}")
        oat_stdout = oat_result.stdout.split('\n')
        oat_lp = float(oat_stdout[-4].split()[2]) # in nucleotides
        oat_contour_length = float(oat_stdout[-3].split()[-2])
        """

        return traj_states, calc_energies, unweighted_corr_curves, unweighted_l0_avgs



    # Initialize parameters
    # FIXME: take care of CPU vs. GPU
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]



    params['stacking']['a_stack'] = 5.999
    params['stacking']['a_stack_1'] = 2.001
    params['stacking']['a_stack_2'] = 1.999
    params['stacking']['a_stack_4'] = 1.299
    params['stacking']['a_stack_5'] = 0.899
    params['stacking']['a_stack_6'] = 0.9
    params['stacking']['delta_theta_star_stack_4'] = 0.799
    params['stacking']['delta_theta_star_stack_5'] = 0.951
    params['stacking']['delta_theta_star_stack_6'] = 0.95
    params['stacking']['dr0_stack'] = 0.399
    params['stacking']['dr_c_stack'] = 0.899
    params['stacking']['dr_high_stack'] = 0.74900015
    params['stacking']['dr_low_stack'] = 0.32
    params['stacking']['eps_stack_base'] = 1.3438
    params['stacking']['eps_stack_kt_coeff'] = 2.6558
    params['stacking']['neg_cos_phi1_star_stack'] = -0.651
    params['stacking']['neg_cos_phi2_star_stack'] = -0.65
    params['stacking']['theta0_stack_4'] = 0.001
    params['stacking']['theta0_stack_5'] = 0.001
    params['stacking']['theta0_stack_6'] = 0.001
    params['stacking']['theta0_stack_6'] = 0.0


    start = time.time()
    ref_states, ref_energies, unweighted_corr_curves, unweighted_l0s = get_ref_states(params, i=0, seed=0)
    end = time.time()
    print(f"Took {end - start} seconds.")


def get_parser():

    parser = argparse.ArgumentParser(description="Calculate persistence length via standalone oxDNA package")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--n-skipped-quartets', type=int, default=5,
                        help="Number of quartets to skip on either side to account for fraying")
    parser.add_argument('--truncation', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='Run name')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
