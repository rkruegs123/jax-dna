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
import os
import socket
import ray
from collections import Counter

import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad
import optax
from jaxopt import GaussNewton

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna1 import model, oxdna_utils
from jax_dna.loss import persistence_length

from jax.config import config
config.update("jax_enable_x64", True)


if "ip_head" in os.environ:
    ray.init(address=os.environ["ip_head"])
else:
    ray.init()

@ray.remote
def run_oxdna_ray(oxdna_exec_path, input_path):
    time.sleep(1)

    hostname = socket.gethostbyname(socket.gethostname())

    start = time.time()
    p = subprocess.Popen([oxdna_exec_path, input_path])
    p.wait()
    end = time.time()

    rc = p.returncode
    return rc, end-start, hostname


checkpoint_every = 25
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)

energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
atol_places = 3
tol = 10**(-atol_places)

compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))

PER_NUC_FORCES = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375]
num_forces = len(PER_NUC_FORCES)
TOTAL_FORCES = jnp.array(PER_NUC_FORCES) * 2.0
TOTAL_FORCES_SI = TOTAL_FORCES * utils.oxdna_force_to_pn # pN
ext_force_bps1 = [5, 214] # should each experience force_per_nuc
ext_force_bps2 = [104, 115] # should each experience -force_per_nuc
dir_force_axis = jnp.array([0, 0, 1])

TOTAL_FORCES_SI = TOTAL_FORCES * utils.oxdna_force_to_pn # pN
test_forces = onp.linspace(0.05, 0.8, 20) # in simulation units
test_forces_si = test_forces * utils.oxdna_force_to_pn # in pN
kT_si = 4.08846006711 # in pN*nm
min_running_avg_idx_wlc = 10
min_running_avg_idx_pdist = 250

x_init = jnp.array([39.87, 50.60, 44.54]) # initialize to the true values
x_init_si = jnp.array([x_init[0] * utils.nm_per_oxdna_length,
                       x_init[1] * utils.nm_per_oxdna_length,
                       x_init[2] * utils.oxdna_force_to_pn])
x_init_lp_fixed = jnp.array([39.87, 44.54])
x_init_lp_fixed_si = jnp.array([x_init_lp_fixed[0] * utils.nm_per_oxdna_length,
                                x_init_lp_fixed[1] * utils.oxdna_force_to_pn])


def compute_dist(state):
    end1_com = (state.center[ext_force_bps1[0]] + state.center[ext_force_bps1[1]]) / 2
    end2_com = (state.center[ext_force_bps2[0]] + state.center[ext_force_bps2[1]]) / 2

    midp_disp = end1_com - end2_com
    projected_dist = jnp.dot(midp_disp, dir_force_axis)
    return jnp.linalg.norm(projected_dist) # Note: incase it's negative

def coth(x):
    # return 1 / jnp.tanh(x)
    return (jnp.exp(2*x) + 1) / (jnp.exp(2*x) - 1)

def calculate_x(force, l0, lps, k, kT):
    y = ((force * l0**2)/(lps*kT))**(1/2)
    x = l0 * (1 + force/k - kT/(2*force*l0) * (1 + y*coth(y)))
    return x

def WLC(coeffs, x_data, force_data, kT):
    # coefficients ordering: [L0, Lp, K]
    l0 = coeffs[0]
    lps = coeffs[1]
    k = coeffs[2]

    x_calc = calculate_x(force_data, l0, lps, k, kT)
    residual = x_data - x_calc
    return residual


def WLC_lp_fixed(coeffs, x_data, force_data, kT, lp):
    # coefficients ordering: [L0, K]
    l0 = coeffs[0]
    k = coeffs[1]

    x_calc = calculate_x(force_data, l0, lp, k, kT)
    residual = x_data - x_calc
    return residual


# Compute the average persistence length
def get_all_quartets(n_nucs_per_strand):
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    bps = list(zip(s1_nucs, s2_nucs))
    n_bps = len(s1_nucs)
    all_quartets = list()
    for i in range(n_bps-1):
        bp1 = bps[i]
        bp2 = bps[i+1]
        all_quartets.append(bp1 + bp2)
    return jnp.array(all_quartets, dtype=jnp.int32)


def run(args):
    # Load parameters

    ## General arguments
    device = args['device']
    if device == "cpu-single-node" or device == "cpu-ray":
        backend = "CPU"
    elif device == "gpu":
        backend = "CUDA"
        raise NotImplementedError(f"TODO: support simultaneous reference state calculations with MPS")
    else:
        raise RuntimeError(f"Invalid device: {device}")

    n_threads = args['n_threads']
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    ## Optimization arguments
    n_iters = args['n_iters']
    lr = args['lr']
    target_ext_mod = args['target_ext_mod']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']


    ## Force extension arguments
    n_sims_per_force = args['n_sims_per_force']
    n_steps_per_sim_fe = args['n_steps_per_sim_fe']
    sample_every_fe = args['sample_every_fe']
    assert(n_steps_per_sim_fe % sample_every_fe == 0)
    n_ref_states_per_sim_fe = n_steps_per_sim_fe // sample_every_fe
    low_forces = args['low_forces']
    for lf in low_forces:
        assert(lf in PER_NUC_FORCES)
    low_force_multiplier = args['low_force_multiplier']
    assert(low_force_multiplier >= 1)

    is_low_force = list()
    low_force_idxs = list()
    normal_force_idxs = list()
    for f_idx, f in enumerate(PER_NUC_FORCES):
        if f in low_forces:
            is_low_force.append(1)
            low_force_idxs.append(f_idx)
        else:
            is_low_force.append(0)
            normal_force_idxs.append(f_idx)
    is_low_force = jnp.array(is_low_force, dtype=jnp.int32)
    normal_force_idxs = jnp.array(normal_force_idxs, dtype=jnp.int32)
    low_force_idxs = jnp.array(low_force_idxs, dtype=jnp.int32)


    ## Setup metadata for set of force ext. simulations
    sim_forces_fe = list()
    sim_idxs_fe = list()
    sim_start_steps_fe = list()

    sim_idx_fe = 0
    n_ref_states_fe = 0
    for force in PER_NUC_FORCES:

        n_force_states = n_sims_per_force * n_ref_states_per_sim_fe
        if force in low_forces:
            n_force_states *= low_force_multiplier
        n_ref_states_fe += n_force_states

        sim_start_step = 0

        sims_to_add = n_sims_per_force
        if force in low_forces:
            sims_to_add *= low_force_multiplier

        for f_sim_idx in range(sims_to_add):
            sim_forces_fe.append(force)
            sim_idxs_fe.append(sim_idx_fe)
            sim_start_steps_fe.append(sim_start_step)

            sim_idx_fe += 1
            sim_start_step += n_steps_per_sim_fe
    n_sims_fe = len(sim_idxs_fe)
    n_sims_total = n_sims_fe


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
    lp_path = log_dir / "lp.txt"
    l0_avg_path = log_dir / "l0_avg.txt"
    ext_mod_path = log_dir / "ext_mod.txt"
    resample_log_path = log_dir / "resample_log.txt"

    params_str = ""
    params_str += f"n_sims_total: {n_sims_total}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # General simulation definitions
    displacement_fn, shift_fn = space.free()
    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT


    @jit
    def get_wlc_params(f_lens):
        gn = GaussNewton(residual_fun=WLC, implicit_diff=True)
        gn_sol = gn.run(x_init, f_lens, TOTAL_FORCES, kT).params
        return gn_sol

    @jit
    def get_wlc_params_lp_fixed(f_lens, lp_fixed):

        gn = GaussNewton(residual_fun=WLC_lp_fixed, implicit_diff=True)
        gn_sol = gn.run(x_init_lp_fixed, f_lens,
                        TOTAL_FORCES, kT, lp_fixed).params

        return gn_sol


    # Load the systems

    ## Force extension
    sys_basedir_fe = Path("data/templates/force-ext")
    externals_basedir = sys_basedir_fe / "externals"
    input_template_path_fe = sys_basedir_fe / "input"

    top_path_fe = sys_basedir_fe / "sys.top"
    top_info_fe = topology.TopologyInfo(top_path_fe, reverse_direction=True)
    seq_oh_fe = jnp.array(utils.get_one_hot(top_info_fe.seq), dtype=jnp.float64)
    assert(seq_oh_fe.shape[0] == 220) # 110 bp duplex
    n_fe = seq_oh_fe.shape[0]

    conf_path_fe = sys_basedir_fe / "init.conf"
    conf_info_fe = trajectory.TrajectoryInfo(
        top_info_fe,
        read_from_file=True, traj_path=conf_path_fe,
        reverse_direction=True
        # reverse_direction=False
    )

    box_size_fe = conf_info_fe.box_size # Only for writing the trajectory


    # Do the simulations
    def get_ref_states(params, i, seed, prev_basedir):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        recompile_start = time.time()
        oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)
        recompile_end = time.time()

        with open(resample_log_path, "a") as f:
            f.write(f"- Recompiling took {recompile_end - recompile_start} seconds\n")

        # Make the simulation directories

        ## Force extension
        fe_dir = iter_dir / "fe"
        fe_dir.mkdir(parents=False, exist_ok=False)
        fe_input_paths = list()
        for sim_force, sim_idx, start_step in zip(sim_forces_fe, sim_idxs_fe, sim_start_steps_fe):
            repeat_dir = fe_dir / f"r{sim_idx}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            external_path = externals_basedir / f"external_{sim_force}.conf"

            shutil.copy(top_path_fe, repeat_dir / "sys.top")
            shutil.copy(external_path, repeat_dir / "external.conf")

            if prev_basedir is None:
                init_conf_info = deepcopy(conf_info_fe)
            else:
                prev_repeat_dir = prev_basedir / "fe" / f"r{sim_idx}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info_fe,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info_fe, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh_fe.shape[0], start_step)
            init_conf_info.write(repeat_dir / "init.conf", reverse=True, write_topology=False)

            oxdna_utils.rewrite_input_file(
                input_template_path_fe, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim_fe,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_fe, seed=random.randrange(100),
                equilibration_steps=0, dt=dt,
                no_stdout_energy=0, backend=backend,
                cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
                log_file=str(repeat_dir / "sim.log"),
                external_forces_file=str(repeat_dir / "external.conf")
            )

            fe_input_paths.append(repeat_dir / "input")

        all_input_paths = fe_input_paths
        sim_start = time.time()
        if device == "cpu-single-node":
            procs = list()

            for ipath in all_input_paths:
                procs.append(subprocess.Popen([oxdna_exec_path, ipath]))

            for p in procs:
                p.wait()

            for p in procs:
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
        elif device == "cpu-ray":
            all_ret_info = ray.get([run_oxdna_ray.remote(oxdna_exec_path, ipath) for ipath in all_input_paths])
            all_rcs = [ret_info[0] for ret_info in all_ret_info]
            all_times = [ret_info[1] for ret_info in all_ret_info]
            all_hostnames = [ret_info[2] for ret_info in all_ret_info]


            sns.histplot(all_times)
            plt.savefig(iter_dir / f"sim_times.png")
            plt.clf()


            with open(resample_log_path, "a") as f:
                f.write(f"Performed {len(all_input_paths)} simulations with Ray...\n")
                f.write(f"Hostname distribution:\n{pprint.pformat(Counter(all_hostnames))}\n")
                f.write(f"Min. time: {onp.min(all_times)}\n")
                f.write(f"Max. time: {onp.max(all_times)}\n")

            for ipath, rc in zip(all_input_paths, all_rcs):
                if rc != 0:
                    raise RuntimeError(f"oxDNA simulation at path {ipath} failed with error code: {rc}")

        else:
            raise NotImplementedError(f"Invalid device: {device}")

        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulation took {sim_end - sim_start} seconds\n")


        # Analyze


        ## Force extension
        analyze_start = time.time()
        fe_analyze_dir = fe_dir / "analyze"
        fe_analyze_dir.mkdir(parents=False, exist_ok=False)

        single_force_dir = fe_analyze_dir / f"single_force"
        single_force_dir.mkdir(parents=False, exist_ok=False)


        ### Combine force trajectories into master output files
        combine_cmds = {force: "cat " for force in PER_NUC_FORCES}
        force_to_idxs = {force: list() for force in PER_NUC_FORCES}
        for idx in range(len(sim_idxs_fe)):
            sim_force = sim_forces_fe[idx]
            assert(sim_force in PER_NUC_FORCES)

            repeat_dir = fe_dir / f"r{sim_idxs_fe[idx]}"
            combine_cmds[sim_force] += f"{repeat_dir}/output.dat "
            force_to_idxs[sim_force].append(idx)

        force_to_energy_df = dict()
        for force in PER_NUC_FORCES:
            energy_dfs = [pd.read_csv(fe_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                      delim_whitespace=True)[1:] for r in force_to_idxs[force]]
            energy_df = pd.concat(energy_dfs, ignore_index=True)
            force_to_energy_df[force] = energy_df

        trajectories_fe = dict()
        for force in PER_NUC_FORCES:
            combine_cmds[force] += f"> {fe_analyze_dir}/output_{force}.dat"
            combine_proc = subprocess.run(combine_cmds[force], shell=True)
            if combine_proc.returncode != 0:
                raise RuntimeError(f"Combining trajectories for force {force} failed with error code: {combine_proc.returncode}")

            traj_info = trajectory.TrajectoryInfo(
                top_info_fe, read_from_file=True,
                traj_path=fe_analyze_dir / f"output_{force}.dat",
                reverse_direction=True)
                # reverse_direction=False)
            traj_states = traj_info.get_states()
            traj_states = utils.tree_stack(traj_states)
            trajectories_fe[force] = traj_states


        ### Generate an energy function
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_fe,
            bonded_nbrs=top_info_fe.bonded_nbrs,
            unbonded_nbrs=top_info_fe.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        ### Calculate energies and check energy differences
        energy_diffs = list()
        calc_energies_fe = dict()
        for force in PER_NUC_FORCES:
            energy_df = force_to_energy_df[force]
            traj_states = trajectories_fe[force]
            n_traj_states = traj_states.center.shape[0]

            calc_energies = list()
            for ts_idx in tqdm(range(n_traj_states), desc="Calculating energies"):
                ts = traj_states[ts_idx]
                calc_energies.append(energy_fn(ts))
            calc_energies = jnp.array(calc_energies)
            calc_energies_fe[force] = calc_energies

            gt_energies = energy_df.potential_energy.to_numpy() * seq_oh_fe.shape[0]

            for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                print(f"State {i}:")
                print(f"\t- Calc. Energy: {calc}")
                print(f"\t- Reference. Energy: {gt}")
                diff = onp.abs(calc - gt)
                energy_diffs.append(diff)
                print(f"\t- Difference: {diff}")

        sns.histplot(energy_diffs)
        plt.savefig(fe_analyze_dir / f"energy_diffs.png")
        plt.clf()

        ### Compute projected distances
        pdists = dict()
        for force in tqdm(PER_NUC_FORCES, desc="Computing projected distances"):
            traj = trajectories_fe[force]
            force_dists = vmap(compute_dist)(traj)
            pdists[force] = force_dists
            sns.distplot(force_dists, label=f"{force*2}")
        plt.legend()
        plt.savefig(fe_analyze_dir / f"pdist_dists.png")
        plt.clf()

        final_f_lens = list() # oxDNA units
        for f in PER_NUC_FORCES:
            final_f_lens.append(jnp.mean(pdists[f]))
        final_f_lens = jnp.array(final_f_lens)


        ### Compute running averages

        all_running_avg_pdists = dict()
        running_avg_interval = 10

        for force, forcex2 in zip(PER_NUC_FORCES, TOTAL_FORCES):
            f_pdists = pdists[force]
            f_running_averages = jnp.cumsum(f_pdists) / jnp.arange(1, f_pdists.shape[0]+1)

            plt.plot(f_running_averages[min_running_avg_idx_pdist:])
            plt.xlabel("Sample")
            plt.ylabel("Avg. Distance (oxDNA units)")
            plt.title(f"Cumulative average, force={force*2}")
            plt.savefig(single_force_dir / f"pdist_running_avg_{force*2}_trunc.png")
            plt.clf()

            if force in low_forces:
                f_running_averages_sampled = f_running_averages[::(running_avg_interval*low_force_multiplier)]
            else:
                f_running_averages_sampled = f_running_averages[::running_avg_interval]

            all_running_avg_pdists[force] = f_running_averages_sampled

        ### Plot the running averages
        for force in PER_NUC_FORCES:
            plt.plot(all_running_avg_pdists[force][min_running_avg_idx_pdist:], label=f"{force}")
        plt.xlabel("Sample")
        plt.ylabel("Avg. Distance (oxDNA units)")
        plt.title(f"Cumulative average")
        plt.legend()
        plt.savefig(fe_analyze_dir / "pdist_running_avg_trunc.png")
        plt.clf()


        num_running_avg_points = len(all_running_avg_pdists[PER_NUC_FORCES[-1]])

        ### Check WLC fit convergence *without* fixed Lp
        all_l0s = list()
        all_lps = list()
        all_ks = list()

        for i in tqdm(range(num_running_avg_points), desc="Computing running avg."):
            f_lens = list()
            for f in PER_NUC_FORCES:
                f_lens.append(all_running_avg_pdists[f][i])
            f_lens = jnp.array(f_lens)

            gn_sol = get_wlc_params(f_lens)
            all_l0s.append(gn_sol[0])
            all_lps.append(gn_sol[1])
            all_ks.append(gn_sol[2])

        plt.plot(all_l0s[min_running_avg_idx_wlc:], label="l0", color="green")
        plt.plot(all_lps[min_running_avg_idx_wlc:], label="lp", color="blue")
        plt.plot(all_ks[min_running_avg_idx_wlc:], label="k", color="red")
        plt.legend()
        plt.title(f"WLC Fit Running Avg. (Truncated)")
        plt.xlabel("Time")
        plt.savefig(fe_analyze_dir / "wlc_fit_running_avg_truncanted.png")
        plt.clf()


        ### Plot the final fit (Lp not fixed)
        gn_sol = get_wlc_params(final_f_lens)
        computed_extensions = [calculate_x(force, gn_sol[0], gn_sol[1], gn_sol[2], kT) for force in test_forces]
        plt.plot(computed_extensions, test_forces, label="fit")
        plt.scatter(final_f_lens, TOTAL_FORCES, label="samples")
        plt.xlabel("Extension (oxDNA units)")
        plt.ylabel("Force (oxDNA units)")
        plt.title("WLC Fit, oxDNA Units")
        plt.legend()
        plt.savefig(fe_analyze_dir / "fit_evaluation_oxdna.png")
        plt.clf()



        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining force ext. analysis took {analyze_end - analyze_start} seconds\n")

        with open(fe_analyze_dir / "summary.txt", "w+") as f:
            f.write(f"Ext. modulus (oxDNA units): {gn_sol[2]}\n")
            f.write(f"Lp (oxDNA units): {gn_sol[1]}\n")
            f.write(f"Contour length (oxDNA units): {gn_sol[0]}\n")
            f.write(f"Force lengths (oxDNA units): {pprint.pformat(final_f_lens)}\n")


        # Return all information

        ## Postprocess force ext. reference data

        trajectories_fe_arr = list()
        calc_energies_fe_arr = list()
        pdists_arr = list()

        trajectories_fe_low_forces_arr = list()
        calc_energies_fe_low_forces_arr = list()
        pdists_low_forces_arr = list()

        for force in PER_NUC_FORCES:

            if force in low_forces:
                trajectories_fe_low_forces_arr.append(trajectories_fe[force])
                calc_energies_fe_low_forces_arr.append(calc_energies_fe[force])
                pdists_low_forces_arr.append(pdists[force])
            else:
                trajectories_fe_arr.append(trajectories_fe[force])
                calc_energies_fe_arr.append(calc_energies_fe[force])
                pdists_arr.append(pdists[force])

        trajectories_fe_arr = utils.tree_stack(trajectories_fe_arr)
        calc_energies_fe_arr = jnp.array(calc_energies_fe_arr)
        pdists_arr = jnp.array(pdists_arr)

        trajectories_fe_low_forces_arr = utils.tree_stack(trajectories_fe_low_forces_arr)
        calc_energies_fe_low_forces_arr = jnp.array(calc_energies_fe_low_forces_arr)
        pdists_low_forces_arr = jnp.array(pdists_low_forces_arr)

        fe_info = [
            trajectories_fe_arr, calc_energies_fe_arr, pdists_arr,
            trajectories_fe_low_forces_arr, calc_energies_fe_low_forces_arr, pdists_low_forces_arr
        ]

        return tuple(fe_info + [iter_dir])

    # Construct the loss function -- FIXME: include force ext. reference data
    @jit
    def loss_fn(params,

                # Force ext. metadata
                ref_states_fe, ref_energies_fe, ref_pdists,
                ref_states_fe_lf, ref_energies_fe_lf, ref_pdists_lf

    ):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)


        # Compute the WLC fit
        energy_fn_fe = lambda body: em.energy_fn(
            body,
            seq=seq_oh_fe,
            bonded_nbrs=top_info_fe.bonded_nbrs,
            unbonded_nbrs=top_info_fe.unbonded_nbrs.T)
        energy_fn_fe = jit(energy_fn_fe)

        energy_scan_fn_fe = lambda state, rs: (None, energy_fn_fe(rs))
        def calc_energies_fe(force_ref_states):
            _, calc_energies_fe = scan(energy_scan_fn_fe, None, force_ref_states)
            return calc_energies_fe
        all_calc_energies_fe_nf = vmap(calc_energies_fe)(ref_states_fe) # `nf` denotes "normal" force
        all_calc_energies_fe_lf = vmap(calc_energies_fe)(ref_states_fe_lf)

        def calc_weights_fe(ref_energies, new_energies):
            diffs = new_energies - ref_energies
            boltzs = jnp.exp(-beta * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom
            return weights
        all_weights_fe_nf = vmap(calc_weights_fe)(ref_energies_fe, all_calc_energies_fe_nf)
        all_weights_fe_lf = vmap(calc_weights_fe)(ref_energies_fe_lf, all_calc_energies_fe_lf)


        def get_expected_pdist(force_pdists, force_weights):
            return jnp.dot(force_pdists, force_weights)
        all_expected_pdists_nf = vmap(get_expected_pdist)(ref_pdists, all_weights_fe_nf)
        all_expected_pdists_lf = vmap(get_expected_pdist)(ref_pdists_lf, all_weights_fe_lf)

        all_expected_pdists = jnp.zeros(num_forces)
        all_expected_pdists = all_expected_pdists.at[low_force_idxs].set(all_expected_pdists_lf)
        all_expected_pdists = all_expected_pdists.at[normal_force_idxs].set(all_expected_pdists_nf)

        gn_sol = get_wlc_params(all_expected_pdists)
        expected_l0 = gn_sol[0]
        expected_lp = gn_sol[1]
        expected_ext_mod = gn_sol[2]

        mse = (target_ext_mod - expected_ext_mod)**2
        rmse = jnp.sqrt(mse)

        all_weights = jnp.concatenate([all_weights_fe_nf.flatten(), all_weights_fe_lf.flatten()])
        n_eff = jnp.exp(-jnp.sum(all_weights * jnp.log(all_weights)))

        return rmse, (n_eff, expected_lp, expected_ext_mod, expected_l0, all_expected_pdists, gn_sol)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    n_ref_states = n_ref_states_fe
    min_n_eff = int(n_ref_states * min_neff_factor)

    all_losses = list()
    all_lps = list()
    all_l0s = list()
    all_n_effs = list()
    all_ext_mods = list()

    all_ref_losses = list()
    all_ref_lps = list()
    all_ref_ext_mods = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    prev_ref_basedir = None
    ref_info = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    end = time.time()

    fe_ref_info = ref_info[:6]
    ref_states_fe, ref_energies_fe, ref_pdists = fe_ref_info[:3]
    ref_states_fe_lf, ref_energies_fe_lf, ref_pdists_lf = fe_ref_info[3:]
    ref_iter_dir = ref_info[-1]
    prev_ref_basedir = deepcopy(ref_iter_dir)

    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, aux), grads = grad_fn(
            params,
            ref_states_fe, ref_energies_fe, ref_pdists,
            ref_states_fe_lf, ref_energies_fe_lf, ref_pdists_lf
        )
        n_eff = aux[0]
        num_resample_iters += 1

        if i == 0:
            expected_lp = aux[1]
            expected_ext_mod = aux[2]
            all_ref_losses.append(loss)
            all_ref_lps.append(expected_lp)
            all_ref_ext_mods.append(expected_ext_mod)
            all_ref_times.append(i)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()
            ref_info = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()

            fe_ref_info = ref_info[:6]
            ref_states_fe, ref_energies_fe, ref_pdists = fe_ref_info[:3]
            ref_states_fe_lf, ref_energies_fe_lf, ref_pdists_lf = fe_ref_info[3:]
            ref_iter_dir = ref_info[-1]
            prev_ref_basedir = deepcopy(ref_iter_dir)

            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, aux), grads = grad_fn(
                params,
                ref_states_fe, ref_energies_fe, ref_pdists,
                ref_states_fe_lf, ref_energies_fe_lf, ref_pdists_lf
            )

            expected_lp = aux[1]
            expected_ext_mod = aux[2]
            all_ref_losses.append(loss)
            all_ref_lps.append(expected_lp)
            all_ref_ext_mods.append(expected_ext_mod)
            all_ref_times.append(i)

        iter_end = time.time()

        n_eff = aux[0]
        expected_lp, expected_ext_mod, expected_l0 = aux[1:4]
        all_expected_pdists, expected_gn_sol = aux[4:6]

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(lp_path, "a") as f:
            f.write(f"{expected_lp}\n")
        with open(l0_avg_path, "a") as f:
            f.write(f"{expected_l0}\n")
        with open(ext_mod_path, "a") as f:
            f.write(f"{expected_ext_mod}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")

        all_losses.append(loss)
        all_n_effs.append(n_eff)
        all_lps.append(expected_lp)
        all_l0s.append(expected_l0)
        all_ext_mods.append(expected_ext_mod)



        # Plot the current fit
        computed_extensions = [calculate_x(force, expected_gn_sol[0], expected_gn_sol[1], expected_gn_sol[2], kT) for force in test_forces]
        plt.plot(computed_extensions, test_forces, label="fit")
        plt.scatter(all_expected_pdists, TOTAL_FORCES, label="samples")
        plt.xlabel("Extension (oxDNA units)")
        plt.ylabel("Force (oxDNA units)")
        plt.title("WLC Fit, oxDNA Units, Lp Fixed")
        plt.legend()
        plt.savefig(img_dir / f"fit_i{i}.png")
        plt.clf()


        # Plot observables

        plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_lps, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_lps, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("Expected Lp (nm)")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"lps_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_ext_mods, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_ext_mods, marker='o', label="Resample points", color="blue")
        plt.axhline(y=target_ext_mod, linestyle='--', label="Target Ext. Modulus", color='red')
        plt.legend()
        plt.ylabel("Expected Ext. Modulus (oxDNA units)")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"ext_mods_iter{i}.png")
        plt.clf()


        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)



def get_parser():

    parser = argparse.ArgumentParser(description="Optimize extensional modulus via standalone oxDNA package")

    # General arguments
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu-single-node",
                        choices=["cpu-single-node", "cpu-ray", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")


    # Optimization arguments
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--target-ext-mod', type=float,
                        help="Target extensional modulus in oxDNA units")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")


    # Force extension arguments
    parser.add_argument('--n-sims-per-force', type=int, default=2,
                        help="Number of individual simulations per force")
    parser.add_argument('--n-steps-per-sim-fe', type=int, default=100000,
                        help="Number of steps per simulation for WLC fit")
    parser.add_argument('--sample-every-fe', type=int, default=1000,
                        help="Frequency of sampling reference states for WLC cit")
    parser.add_argument('--low-forces', nargs='*', type=float,
                        help='Forces for which we simulate for longer')
    parser.add_argument('--low-force-multiplier', type=int, default=1,
                        help="Multiplicative factor for low force simulation length")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
