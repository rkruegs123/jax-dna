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
import seaborn as sns

import jax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax
from jaxopt import GaussNewton

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model, oxdna_utils

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


ALL_FORCES = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.375] # per nucleotide
TOTAL_FORCES = jnp.array(ALL_FORCES) * 2.0
TOTAL_FORCES_SI = TOTAL_FORCES * utils.oxdna_force_to_pn # pN
test_forces = onp.linspace(0.05, 0.8, 20) # in simulation units
test_forces_si = test_forces * utils.oxdna_force_to_pn # in pN
num_forces = len(ALL_FORCES)
ext_force_bps1 = [5, 214] # should each experience force_per_nuc
ext_force_bps2 = [104, 115] # should each experience -force_per_nuc
dir_force_axis = jnp.array([0, 0, 1])
kT_si = 4.08846006711 # in pN*nm
min_running_avg_idx = 10

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


def simulate(args):
    # Load parameters
    device = args['device']
    if device == "cpu":
        backend = "CPU"
    elif device == "gpu":
        backend = "CUDA"
    else:
        raise RuntimeError(f"Invalid device: {device}")
    n_threads = args['n_threads']
    key = args['key']
    n_sims_per_force = args['n_sims_per_force']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims_per_force
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    t_kelvin = args['temp']
    oxdna_cuda_device = args['oxdna_cuda_device']
    oxdna_cuda_list = args['oxdna_cuda_list']

    low_forces_input = args['low_forces']
    low_forces = list()
    for lf in low_forces_input:
        lf_float = float(lf)
        assert(lf_float in ALL_FORCES)
        low_forces.append(lf_float)

    low_force_multiplier = args['low_force_multiplier']
    assert(low_force_multiplier >= 1)

    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/force-ext")
    externals_basedir = sys_basedir / "externals"
    ## note: no external_path set
    # external_path = externals_basedir / f"external_{force_per_nuc}.conf"
    # if not external_path.exists():
    #     raise RuntimeError(f"No external forces file at location: {external_path}")
    input_template_path = sys_basedir / "input"


    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] == 220) # 110 bp duplex
    n = seq_oh.shape[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=True
        # reverse_direction=False
    )

    box_size = conf_info.box_size # Only for writing the trajectory
    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT


    # Do the simulation
    random.seed(key)

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    iter_dir = run_dir / f"simulation"
    iter_dir.mkdir(parents=False, exist_ok=False)

    oxdna_utils.recompile_oxdna(params, oxdna_path, t_kelvin, num_threads=n_threads)

    sim_start = time.time()
    if device == "cpu":
        procs = list()


    ## Setup the simulation information
    sim_lengths = list()
    sim_forces = list()
    sim_idxs = list()
    sim_start_steps = list()

    sim_idx = 0
    for force in ALL_FORCES:
        sim_start_step = 0

        sims_to_add = n_sims_per_force
        if force in low_forces:
            sims_to_add *= low_force_multiplier

        for f_sim_idx in range(sims_to_add):
            sim_length = n_steps_per_sim

            sim_lengths.append(sim_length)
            sim_forces.append(force)
            sim_idxs.append(sim_idx)
            sim_start_steps.append(sim_start_step)

            sim_idx += 1
            sim_start_step += sim_length

    # TODO: run all the simulation
    # note: keeping naming convention as r{idx} for comptability with mps.sh
    for sim_length, sim_force, sim_idx, start_step in zip(sim_lengths, sim_forces, sim_idxs, sim_start_steps):
        repeat_dir = iter_dir / f"r{sim_idx}"
        repeat_dir.mkdir(parents=False, exist_ok=False)

        external_path = externals_basedir / f"external_{sim_force}.conf"

        shutil.copy(top_path, repeat_dir / "sys.top")
        shutil.copy(external_path, repeat_dir / "external.conf")

        init_conf_info = deepcopy(conf_info)

        init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], start_step)
        init_conf_info.write(repeat_dir / "init.conf", reverse=True, write_topology=False)

        oxdna_utils.rewrite_input_file(
            input_template_path, repeat_dir,
            temp=f"{t_kelvin}K", steps=sim_length,
            init_conf_path=str(repeat_dir / "init.conf"),
            top_path=str(repeat_dir / "sys.top"),
            save_interval=sample_every, seed=random.randrange(100),
            equilibration_steps=n_eq_steps, dt=dt,
            no_stdout_energy=0, backend=backend,
            cuda_device=oxdna_cuda_device, cuda_list=oxdna_cuda_list,
            log_file=str(repeat_dir / "sim.log"),
            external_forces_file=str(repeat_dir / "external.conf")
        )

        if device == "cpu":
            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

    if device == "cpu":
        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")
    elif device == "gpu":
        mps_cmd = f"./scripts/mps.sh {oxdna_path} {iter_dir} {n_sims}"
        mps_proc = subprocess.run(mps_cmd, shell=True)
        if mps_proc.returncode != 0:
            raise RuntimeError(f"Generating states via MPS failed with error code: {mps_proc.returncode}")
    else:
        raise RuntimeError(f"Invalid device: {device}")


    # Save information for analysis
    onp.save(run_dir / "sim_lengths.npy", onp.array(sim_lengths), allow_pickle=False)
    onp.save(run_dir / "sim_forces.npy", onp.array(sim_forces), allow_pickle=False)
    onp.save(run_dir / "sim_idxs.npy", onp.array(sim_idxs), allow_pickle=False)
    onp.save(run_dir / "sim_start_steps.npy", onp.array(sim_start_steps), allow_pickle=False)

    return sim_lengths, sim_forces, sim_idxs, sim_start_steps


def analyze(args):

    # Process arguments
    n_sims_per_force = args['n_sims_per_force']
    n_steps_per_sim = args['n_steps_per_sim']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims_per_force
    run_name = args['run_name']
    t_kelvin = args['temp']
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    tom_extensions = {float(f): calculate_x(f, x_init[0], x_init[1], x_init[2], kT) for f in TOTAL_FORCES}

    @jit
    def get_wlc_params(f_lens):
        gn = GaussNewton(residual_fun=WLC)
        gn_sol = gn.run(x_init, x_data=f_lens, force_data=TOTAL_FORCES, kT=kT).params
        return gn_sol

    @jit
    def get_wlc_params_lp_fixed(f_lens):
        gn = GaussNewton(residual_fun=WLC_lp_fixed)
        gn_sol = gn.run(x_init_lp_fixed, x_data=f_lens,
                        force_data=TOTAL_FORCES, kT=kT, lp=x_init[1]).params
        return gn_sol

    low_forces_input = args['low_forces']
    low_forces = list()
    for lf in low_forces_input:
        lf_float = float(lf)
        assert(lf_float in ALL_FORCES)
        low_forces.append(lf_float)

    low_force_multiplier = args['low_force_multiplier']
    assert(low_force_multiplier >= 1)

    sys_basedir = Path("data/templates/force-ext")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    assert(seq_oh.shape[0] == 220) # 110 bp duplex
    n = seq_oh.shape[0]

    displacement_fn, shift_fn = space.free()

    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name

    sim_dir = run_dir / f"simulation"
    assert(sim_dir.exists())

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    iter_dir = run_dir / f"analyze"
    iter_dir.mkdir(parents=False, exist_ok=False)

    single_force_dir = iter_dir / f"single_force"
    single_force_dir.mkdir(parents=False, exist_ok=False)


    # Check for simulation information existence
    assert(run_dir.exists())

    sim_lengths_fpath = run_dir / "sim_lengths.npy"
    assert(sim_lengths_fpath.exists())
    sim_lengths = onp.load(sim_lengths_fpath)

    sim_forces_fpath = run_dir / "sim_forces.npy"
    assert(sim_forces_fpath.exists())
    sim_forces = onp.load(sim_forces_fpath)

    sim_idxs_fpath = run_dir / "sim_idxs.npy"
    assert(sim_idxs_fpath.exists())
    sim_idxs = onp.load(sim_idxs_fpath)

    sim_start_steps_fpath = run_dir / "sim_start_steps.npy"
    assert(sim_start_steps_fpath.exists())
    sim_start_steps = onp.load(sim_start_steps_fpath)


    # Combine force trajectories into master output files and load
    combine_cmds = {force: "cat " for force in ALL_FORCES}
    force_to_idxs = {force: list() for force in ALL_FORCES}
    for idx in range(len(sim_idxs)):
        sim_force = sim_forces[idx]
        assert(sim_force in ALL_FORCES)

        repeat_dir = sim_dir / f"r{sim_idxs[idx]}"
        combine_cmds[sim_force] += f"{repeat_dir}/output.dat "
        force_to_idxs[sim_force].append(idx)
    force_to_energy_df = dict()
    energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
    for force in ALL_FORCES:
        energy_dfs = [pd.read_csv(sim_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in force_to_idxs[force]]
        energy_df = pd.concat(energy_dfs, ignore_index=True)
        force_to_energy_df[force] = energy_df
    trajectories = dict()
    for force in ALL_FORCES:
        combine_cmds[force] += f"> {iter_dir}/output_{force}.dat"
        combine_proc = subprocess.run(combine_cmds[force], shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories for force {force} failed with error code: {combine_proc.returncode}")

        traj_info = trajectory.TrajectoryInfo(
            top_info, read_from_file=True,
            traj_path=iter_dir / f"output_{force}.dat",
            reverse_direction=True)
            # reverse_direction=False)
        traj_states = traj_info.get_states()
        traj_states = utils.tree_stack(traj_states)
        trajectories[force] = traj_states

    # Check energies

    ## Generate an energy function

    em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
    energy_fn = lambda body: em.energy_fn(
        body,
        seq=seq_oh,
        bonded_nbrs=top_info.bonded_nbrs,
        unbonded_nbrs=top_info.unbonded_nbrs.T)
    energy_fn = jit(energy_fn)

    ## FIXME: log energy differences how we do during optimization
    atol_places = 3
    tol = 10**(-atol_places)
    energy_diffs = list()
    for force in ALL_FORCES:
        energy_df = force_to_energy_df[force]
        traj_states = trajectories[force]
        n_traj_states = traj_states.center.shape[0]

        calc_energies = list()
        for ts_idx in tqdm(range(n_traj_states), desc="Calculating energies"):
            ts = traj_states[ts_idx]
            calc_energies.append(energy_fn(ts))
        calc_energies = jnp.array(calc_energies)

        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            print(f"State {i}:")
            print(f"\t- Calc. Energy: {calc}")
            print(f"\t- Reference. Energy: {gt}")
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)
            print(f"\t- Difference: {diff}")

    sns.histplot(energy_diffs)
    plt.savefig(iter_dir / f"energy_diffs.png")
    plt.clf()

    # Compute projected distances
    pdists = dict()
    for force in tqdm(ALL_FORCES, desc="Computing projected distances"):
        traj = trajectories[force]
        force_dists = vmap(compute_dist)(traj)
        pdists[force] = force_dists
        sns.distplot(force_dists, label=f"{force*2}")
    plt.legend()
    plt.savefig(iter_dir / f"pdist_dists.png")
    plt.clf()

    final_f_lens = list() # oxDNA units
    for f in ALL_FORCES:
        final_f_lens.append(jnp.mean(pdists[f]))
    final_f_lens = jnp.array(final_f_lens)
    final_f_lens_si = final_f_lens * utils.nm_per_oxdna_length # in nm


    # Compute the running average of forces
    all_running_avg_pdists = dict()
    running_avg_interval = 10
    for force, forcex2 in zip(ALL_FORCES, TOTAL_FORCES):
        f_pdists = pdists[force]
        f_running_averages = jnp.cumsum(f_pdists) / jnp.arange(1, f_pdists.shape[0]+1)

        plt.plot(f_running_averages)
        plt.axhline(y=tom_extensions[float(forcex2)], linestyle="--")
        plt.xlabel("Sample")
        plt.ylabel("Avg. Distance (oxDNA units)")
        plt.title(f"Cumulative average, force={force*2}")
        plt.savefig(single_force_dir / f"pdist_running_avg_{force*2}.png")
        plt.clf()

        plt.plot(f_running_averages[min_running_avg_idx:])
        plt.axhline(y=tom_extensions[float(forcex2)], linestyle="--")
        plt.xlabel("Sample")
        plt.ylabel("Avg. Distance (oxDNA units)")
        plt.title(f"Cumulative average, force={force*2}")
        plt.savefig(single_force_dir / f"pdist_running_avg_{force*2}_trunc.png")
        plt.clf()

        if force in low_forces:
            f_running_averages_sampled = f_running_averages[::(running_avg_interval*low_force_multiplier)]
        else:
            f_running_averages_sampled = f_running_averages[::running_avg_interval]


        all_l0s = list()
        all_lps = list()
        all_ks = list()
        all_l0s_lp_fixed = list()
        all_ks_lp_fixed = list()
        for i in tqdm(range(len(f_running_averages_sampled)), desc=f"Computing running avg. for {forcex2}"):
            f_lens = list()
            for f, fx2 in zip(ALL_FORCES, TOTAL_FORCES):
                if f == force:
                    f_lens.append(f_running_averages_sampled[i])
                else:
                    f_lens.append(tom_extensions[float(fx2)])
            f_lens = jnp.array(f_lens)

            # gn = GaussNewton(residual_fun=WLC)
            # gn_sol = gn.run(x_init, x_data=f_lens, force_data=TOTAL_FORCES, kT=kT).params
            gn_sol = get_wlc_params(f_lens)

            all_l0s.append(gn_sol[0])
            all_lps.append(gn_sol[1])
            all_ks.append(gn_sol[2])

            # gn = GaussNewton(residual_fun=WLC_lp_fixed)
            # gn_sol = gn.run(x_init_lp_fixed, x_data=f_lens,
            #                 force_data=TOTAL_FORCES, kT=kT, lp=x_init[1]).params
            gn_sol = get_wlc_params_lp_fixed(f_lens)
            all_l0s_lp_fixed.append(gn_sol[0])
            all_ks_lp_fixed.append(gn_sol[1])


        plt.plot(all_l0s, label="l0", color="green")
        plt.axhline(y=x_init[0], label="l0, true", linestyle="--", color="green")
        plt.plot(all_lps, label="lp", color="blue")
        plt.axhline(y=x_init[1], label="lp, true", linestyle="--", color="blue")
        plt.plot(all_ks, label="k", color="red")
        plt.axhline(y=x_init[2], label="k, true", linestyle="--", color="red")
        plt.legend()
        plt.title(f"WLC Fit Running Avg.")
        plt.xlabel("Time")
        plt.savefig(single_force_dir / f"wlc_fit_running_avg_{force*2}.png")
        plt.clf()

        plt.plot(all_l0s_lp_fixed, label="l0", color="green")
        plt.axhline(y=x_init_lp_fixed[0], label="l0, true", linestyle="--", color="green")
        plt.plot(all_ks_lp_fixed, label="k", color="red")
        plt.axhline(y=x_init_lp_fixed[1], label="k, true", linestyle="--", color="red")
        plt.legend()
        plt.title(f"WLC Fit Running Avg., Lp={x_init[1]}")
        plt.xlabel("Time")
        plt.savefig(single_force_dir / f"wlc_fit_running_avg_{force*2}_lp_fixed.png")
        plt.clf()

        plt.plot(all_l0s[min_running_avg_idx:], label="l0", color="green")
        plt.axhline(y=x_init[0], label="l0, true", linestyle="--", color="green")
        plt.plot(all_lps[min_running_avg_idx:], label="lp", color="blue")
        plt.axhline(y=x_init[1], label="lp, true", linestyle="--", color="blue")
        plt.plot(all_ks[min_running_avg_idx:], label="k", color="red")
        plt.axhline(y=x_init[2], label="k, true", linestyle="--", color="red")
        plt.legend()
        plt.title(f"WLC Fit Running Avg. (Truncated)")
        plt.xlabel("Time")
        plt.savefig(single_force_dir / f"wlc_fit_running_avg_{force*2}_truncanted.png")
        plt.clf()

        plt.plot(all_l0s_lp_fixed[min_running_avg_idx:], label="l0", color="green")
        plt.axhline(y=x_init_lp_fixed[0], label="l0, true", linestyle="--", color="green")
        plt.plot(all_ks_lp_fixed[min_running_avg_idx:], label="k", color="red")
        plt.axhline(y=x_init_lp_fixed[1], label="k, true", linestyle="--", color="red")
        plt.legend()
        plt.title(f"WLC Fit Running Avg. (Truncated), Lp={x_init[1]}")
        plt.xlabel("Time")
        plt.savefig(single_force_dir / f"wlc_fit_running_avg_{force*2}_truncanted_lp_fixed.png")
        plt.clf()

        all_running_avg_pdists[force] = f_running_averages_sampled


    # Plot the running averages
    for force in ALL_FORCES:
        plt.plot(all_running_avg_pdists[force], label=f"{force}")
    plt.xlabel("Sample")
    plt.ylabel("Avg. Distance (oxDNA units)")
    plt.title(f"Cumulative average")
    plt.legend()
    plt.savefig(iter_dir / "pdist_running_avg.png")
    plt.clf()

    # Plot the running averages (truncated)
    for force in ALL_FORCES:
        plt.plot(all_running_avg_pdists[force][min_running_avg_idx:], label=f"{force}")
    plt.xlabel("Sample")
    plt.ylabel("Avg. Distance (oxDNA units)")
    plt.title(f"Cumulative average")
    plt.legend()
    plt.savefig(iter_dir / "pdist_running_avg_trunc.png")
    plt.clf()

    # Calculate the running WLC fits *with Lp unknown*
    all_l0s = list()
    all_lps = list()
    all_ks = list()
    num_running_avg_points = len(all_running_avg_pdists[ALL_FORCES[-1]])
    for i in tqdm(range(num_running_avg_points), desc="Computing running avgs"):
        f_lens = list()
        for f in ALL_FORCES:
            f_lens.append(all_running_avg_pdists[f][i])
        f_lens = jnp.array(f_lens)

        # gn = GaussNewton(residual_fun=WLC)
        # gn_sol = gn.run(x_init, x_data=f_lens, force_data=TOTAL_FORCES, kT=kT).params
        gn_sol = get_wlc_params(f_lens)

        all_l0s.append(gn_sol[0])
        all_lps.append(gn_sol[1])
        all_ks.append(gn_sol[2])

    # Plot running WLC fits
    plt.plot(all_l0s, label="l0", color="green")
    plt.axhline(y=x_init[0], label="l0, true", linestyle="--", color="green")
    plt.plot(all_lps, label="lp", color="blue")
    plt.axhline(y=x_init[1], label="lp, true", linestyle="--", color="blue")
    plt.plot(all_ks, label="k", color="red")
    plt.axhline(y=x_init[2], label="k, true", linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg.")
    plt.xlabel("Time")
    plt.savefig(iter_dir / "wlc_fit_running_avg.png")
    plt.clf()

    # Plot running WLC fits (truncated)
    plt.plot(all_l0s[min_running_avg_idx:], label="l0", color="green")
    plt.axhline(y=x_init[0], label="l0, true", linestyle="--", color="green")
    plt.plot(all_lps[min_running_avg_idx:], label="lp", color="blue")
    plt.axhline(y=x_init[1], label="lp, true", linestyle="--", color="blue")
    plt.plot(all_ks[min_running_avg_idx:], label="k", color="red")
    plt.axhline(y=x_init[2], label="k, true", linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.savefig(iter_dir / "wlc_fit_running_avg_truncanted.png")
    plt.clf()

    # Plot running WLC fits (truncated, SI units)
    plt.plot(jnp.array(all_l0s[min_running_avg_idx:])*utils.nm_per_oxdna_length,
             label="l0", color="green")
    plt.axhline(y=x_init[0]*utils.nm_per_oxdna_length, label="l0, true",
                linestyle="--", color="green")
    plt.plot(jnp.array(all_lps[min_running_avg_idx:])*utils.nm_per_oxdna_length,
             label="lp", color="blue")
    plt.axhline(y=x_init[1]*utils.nm_per_oxdna_length, label="lp, true",
                linestyle="--", color="blue")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.ylabel("Length (nm)")
    plt.savefig(iter_dir / "wlc_fit_lens_running_avg_truncanted_si.png")
    plt.clf()

    plt.plot(jnp.array(all_ks[min_running_avg_idx:])*utils.oxdna_force_to_pn,
             label="k", color="red")
    plt.axhline(y=x_init[2]*utils.oxdna_force_to_pn, label="k, true",
                linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated)")
    plt.xlabel("Time")
    plt.ylabel("Extensional Modulus (pN)")
    plt.savefig(iter_dir / "wlc_fit_ext_mod_running_avg_truncanted_si.png")
    plt.clf()


    # Test the fit against true values

    ## oxDNA units

    # gn = GaussNewton(residual_fun=WLC)
    # gn_sol = gn.run(x_init, x_data=final_f_lens, force_data=TOTAL_FORCES, kT=kT).params
    gn_sol = get_wlc_params(final_f_lens)

    computed_extensions = [calculate_x(force, gn_sol[0], gn_sol[1], gn_sol[2], kT) for force in test_forces]
    tom_extensions = [calculate_x(force, x_init[0], x_init[1], x_init[2], kT) for force in test_forces]

    plt.plot(computed_extensions, test_forces, label="fit")
    plt.scatter(final_f_lens, TOTAL_FORCES, label="samples")
    plt.plot(tom_extensions, test_forces, label="tom fit")
    plt.xlabel("Extension (oxDNA units)")
    plt.ylabel("Force (oxDNA units)")
    plt.title("Fit Evaluation, oxDNA Units")
    plt.legend()
    plt.savefig(iter_dir / "fit_evaluation_oxdna.png")
    plt.clf()


    ## SI units

    gn_si = GaussNewton(residual_fun=WLC)
    gn_sol_si = gn_si.run(x_init_si, x_data=final_f_lens_si,
                          force_data=TOTAL_FORCES_SI, kT=kT_si).params

    computed_extensions_si = [calculate_x(force, gn_sol_si[0], gn_sol_si[1], gn_sol_si[2], kT_si) for force in test_forces_si] # in nm
    tom_extensions_si = [calculate_x(force, x_init_si[0], x_init_si[1], x_init_si[2], kT_si) for force in test_forces_si] # in nm

    plt.plot(computed_extensions_si, test_forces_si, label="fit")
    plt.scatter(final_f_lens_si, TOTAL_FORCES_SI, label="samples")
    plt.plot(tom_extensions_si, test_forces_si, label="tom fit")
    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)")
    plt.title("Fit Evaluation, SI Units")
    plt.legend()
    plt.savefig(iter_dir / "fit_evaluation_si.png")
    plt.clf()


    # Repeat WLC analysis *with a fixed Lp*

    # Calculate the running WLC fits
    all_l0s = list()
    all_ks = list()
    for i in tqdm(range(num_running_avg_points), desc="Computing running avg., Lp fixed"):
        f_lens = list()
        for f in ALL_FORCES:
            f_lens.append(all_running_avg_pdists[f][i])
        f_lens = jnp.array(f_lens)

        # gn = GaussNewton(residual_fun=WLC_lp_fixed)
        # gn_sol = gn.run(x_init_lp_fixed, x_data=f_lens,
        #                 force_data=TOTAL_FORCES, kT=kT, lp=x_init[1]).params
        gn_sol = get_wlc_params_lp_fixed(f_lens)

        all_l0s.append(gn_sol[0])
        all_ks.append(gn_sol[1])

    # Plot running WLC fits
    plt.plot(all_l0s, label="l0", color="green")
    plt.axhline(y=x_init_lp_fixed[0], label="l0, true", linestyle="--", color="green")
    plt.plot(all_ks, label="k", color="red")
    plt.axhline(y=x_init_lp_fixed[1], label="k, true", linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg., Lp Fixed")
    plt.xlabel("Time")
    plt.savefig(iter_dir / "wlc_fit_running_avg_lp_fixed.png")
    plt.clf()

    # Plot running WLC fits (truncated)
    plt.plot(all_l0s[min_running_avg_idx:], label="l0", color="green")
    plt.axhline(y=x_init_lp_fixed[0], label="l0, true", linestyle="--", color="green")
    plt.plot(all_ks[min_running_avg_idx:], label="k", color="red")
    plt.axhline(y=x_init_lp_fixed[1], label="k, true", linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated), Lp Fixed")
    plt.xlabel("Time")
    plt.savefig(iter_dir / "wlc_fit_running_avg_truncanted_lp_fixed.png")
    plt.clf()


    # Plot running WLC fits (truncated, SI units)
    plt.plot(jnp.array(all_l0s[min_running_avg_idx:])*utils.nm_per_oxdna_length,
             label="l0", color="green")
    plt.axhline(y=x_init_lp_fixed[0]*utils.nm_per_oxdna_length, label="l0, true",
                linestyle="--", color="green")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated), Lp Fixed")
    plt.xlabel("Time")
    plt.ylabel("Length (nm)")
    plt.savefig(iter_dir / "wlc_fit_lens_running_avg_truncanted_si_lp_fixed.png")
    plt.clf()

    plt.plot(jnp.array(all_ks[min_running_avg_idx:])*utils.oxdna_force_to_pn,
             label="k", color="red")
    plt.axhline(y=x_init_lp_fixed[1]*utils.oxdna_force_to_pn, label="k, true",
                linestyle="--", color="red")
    plt.legend()
    plt.title(f"WLC Fit Running Avg. (Truncated), Lp Fixed")
    plt.xlabel("Time")
    plt.ylabel("Extensional Modulus (pN)")
    plt.savefig(iter_dir / "wlc_fit_ext_mod_running_avg_truncanted_si_lp_fixed.png")
    plt.clf()



    # Test the fit against true values

    # gn = GaussNewton(residual_fun=WLC_lp_fixed)
    # gn_sol = gn.run(x_init_lp_fixed, x_data=final_f_lens, force_data=TOTAL_FORCES,
    #                 kT=kT, lp=x_init[1]).params
    gn_sol = get_wlc_params_lp_fixed(final_f_lens)

    computed_extensions = [calculate_x(force, gn_sol[0], x_init[1], gn_sol[1], kT) for force in test_forces]
    tom_extensions = [calculate_x(force, x_init_lp_fixed[0], x_init[1], x_init_lp_fixed[1], kT) for force in test_forces]

    plt.plot(computed_extensions, test_forces, label="fit")
    plt.scatter(final_f_lens, TOTAL_FORCES, label="samples")
    plt.plot(tom_extensions, test_forces, label="tom fit")
    plt.xlabel("Extension (oxDNA units)")
    plt.ylabel("Force (oxDNA units)")
    plt.title("Fit Evaluation, oxDNA Units, Lp Fixed")
    plt.legend()
    plt.savefig(iter_dir / "fit_evaluation_oxdna_lp_fixed.png")
    plt.clf()


    ## SI units

    gn_si = GaussNewton(residual_fun=WLC_lp_fixed)
    gn_sol_si = gn_si.run(x_init_lp_fixed_si, x_data=final_f_lens_si,
                          force_data=TOTAL_FORCES_SI, kT=kT_si, lp=x_init_si[1]).params

    computed_extensions_si = [calculate_x(force, gn_sol_si[0], x_init_si[1], gn_sol_si[1], kT_si) for force in test_forces_si] # in nm
    tom_extensions_si = [calculate_x(force, x_init_lp_fixed_si[0], x_init_si[1], x_init_lp_fixed_si[1], kT_si) for force in test_forces_si] # in nm

    plt.plot(computed_extensions_si, test_forces_si, label="fit")
    plt.scatter(final_f_lens_si, TOTAL_FORCES_SI, label="samples")
    plt.plot(tom_extensions_si, test_forces_si, label="tom fit")
    plt.xlabel("Extension (nm)")
    plt.ylabel("Force (pN)")
    plt.title("Fit Evaluation, SI Units, Lp Fixed")
    plt.legend()
    plt.savefig(iter_dir / "fit_evaluation_si_lp_fixed.png")
    plt.clf()





def run(args):
    simulate(args)
    analyze(args)


def get_parser():

    # Simulation arguments
    parser = argparse.ArgumentParser(description="Optimize a single force extension")

    parser.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--n-sims-per-force', type=int, default=2,
                        help="Number of individual simulations per force")
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
    parser.add_argument('--temp', type=float, default=utils.DEFAULT_TEMP,
                        help="Simulation temperature in Kelvin")
    parser.add_argument('--oxdna-cuda-device', type=int, default=0,
                        help="CUDA device for running oxDNA simulations")
    parser.add_argument('--oxdna-cuda-list', type=str, default="verlet",
                        choices=["no", "verlet"],
                        help="CUDA neighbor lists")


    parser.add_argument('--low-forces', nargs='*',
                        help='Forces for which we simulate for longer')
    parser.add_argument('--low-force-multiplier', type=int, default=1,
                        help="Multiplicative factor for low force simulation length")

    parser.add_argument('--simulate-only', action='store_true')
    parser.add_argument('--analyze-only', action='store_true')


    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    assert(args['n_eq_steps'] == 0)

    if args['simulate_only']:
        simulate(args)
    elif args['analyze_only']:
        analyze(args)
    else:
        run(args)




    # FIXME: should also add a simulate-only and analyze-only flag
    # FIXME: should also maybe make the multiplier on the number of simulations rather than on the length of a given simulation... Note that I don't think this sohuld chang eth ebehavior of analyze, as the ocmbined force output will be the same length
    # FIXME: then, should add an assert that the number of total simulations i sless than or equal to the number of individual threads.
