from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import subprocess
import pdb
from copy import deepcopy
import time
import functools
import numpy as onp
import pprint
import random
import pandas as pd
import socket
import ray
from collections import Counter

from jax import jit, vmap, lax, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body
import optax

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna2 import model, lammps_utils



if "ip_head" in os.environ:
    ray.init(address=os.environ["ip_head"])
else:
    ray.init()

@ray.remote
def run_lammps_ray(lammps_exec_path, sim_dir):
    time.sleep(1)

    hostname = socket.gethostbyname(socket.gethostname())

    start = time.time()
    p = subprocess.Popen([lammps_exec_path, "-in", "in"], cwd=sim_dir)
    p.wait()
    end = time.time()

    rc = p.returncode
    return rc, end-start, hostname


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

def single_pitch(quartet, base_sites, displacement_fn):
    # a1 is h-bonded to b1, a2 is h-bonded to b2
    a1, b1, a2, b2 = quartet

    # get normalized base-base vectors for each base pair, 1 and 2
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    bb1 = bb1[:2]
    bb2 = bb2[:2]

    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = bb2 / jnp.linalg.norm(bb2)

    theta = jnp.arccos(utils.clamp(jnp.dot(bb1, bb2)))

    return theta


def compute_pitches(body, quartets, displacement_fn, com_to_hb):
    # Construct the base site position in the body frame
    base_site_bf = jnp.array([com_to_hb, 0.0, 0.0])

    # Compute the space-frame base sites
    base_sites = body.center + rigid_body.quaternion_rotate(
        body.orientation, base_site_bf)

    # Compute the pitches for all quartets
    all_pitches = vmap(single_pitch, (0, None, None))(
        quartets, base_sites, displacement_fn)

    return all_pitches


def get_bp_pos(body, bp):
    return (body.center[bp[0]] + body.center[bp[1]]) / 2

def run(args):
    lammps_basedir = Path(args['lammps_basedir'])
    assert(lammps_basedir.exists())
    lammps_exec_path = lammps_basedir / "build/lmp"
    assert(lammps_exec_path.exists())

    tacoxdna_basedir = Path(args['tacoxdna_basedir'])
    assert(tacoxdna_basedir.exists())

    sample_every = args['sample_every']
    n_sims = args['n_sims']

    n_eq_steps = args['n_eq_steps']
    assert(n_eq_steps % sample_every == 0)
    n_eq_states = n_eq_steps // sample_every

    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_sample_states = n_sample_steps // sample_every

    n_total_steps = n_eq_steps + n_sample_steps
    n_total_states = n_total_steps // sample_every
    assert(n_total_states == n_sample_states + n_eq_states)

    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    rmse_coeff = args['rmse_coeff']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    seq_avg = not args['seq_dep']
    # assert(seq_avg)


    # forces_pn = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    # torques_pnnm = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    forces_pn = jnp.array([0.0, 2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    torques_pnnm = jnp.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])



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

    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    neffs_path = log_dir / "neffs.txt"
    slope_path = log_dir / "slope.txt"
    diffs_path = log_dir / "diffs.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_sample_states: {n_sample_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/templates/lammps-stretch-tors")
    lammps_data_rel_path = sys_basedir / "data"
    lammps_data_abs_path = os.getcwd() / lammps_data_rel_path

    p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", lammps_data_abs_path], cwd=run_dir)
    p.wait()
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

    init_conf_fpath = run_dir / "data.oxdna"
    assert(init_conf_fpath.exists())
    os.rename(init_conf_fpath, run_dir / "init.conf")

    top_fpath = run_dir / "data.top"
    assert(top_fpath.exists())
    os.rename(top_fpath, run_dir / "sys.top")
    top_fpath = run_dir / "sys.top"

    top_info = topology.TopologyInfo(top_fpath, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)
    n = seq_oh.shape[0]
    assert(n % 2 == 0)
    n_bp = n // 2

    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    ## The region for which theta and distance are measured
    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[4:n_bp-5]

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]

    displacement_fn, shift_fn = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.

    def compute_distance(body):
        bp1_meas_pos = get_bp_pos(ref_state, bp1_meas)
        bp2_meas_pos = get_bp_pos(ref_state, bp2_meas)
        dist = jnp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
        return dist


    def compute_theta(body):
        pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
        return pitches.sum()

    t_kelvin = 300.0
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    salt_conc = 0.15
    q_eff = 0.815


    def get_ref_states(params, i, seed):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        # Run the simulations
        procs = list()
        all_sim_dirs = list()
        repeat_seeds = [random.randrange(100) for _ in range(n_sims)]
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                # repeat_seed = random.randrange(100)
                repeat_seed = repeat_seeds[r]

                shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                    force_pn=force_pn, torque_pnnm=0,
                    save_every=sample_every, n_steps=n_total_steps,
                    seq_avg=seq_avg, seed=repeat_seed)

        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                repeat_seed = repeat_seeds[r]

                shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                    force_pn=2.0, torque_pnnm=torque_pnnm,
                    save_every=sample_every, n_steps=n_total_steps,
                    seq_avg=seq_avg, seed=repeat_seed)


        all_ret_info = ray.get([run_lammps_ray.remote(lammps_exec_path, rdir) for rdir in all_sim_dirs])
        all_rcs = [ret_info[0] for ret_info in all_ret_info]
        all_times = [ret_info[1] for ret_info in all_ret_info]
        all_hostnames = [ret_info[2] for ret_info in all_ret_info]

        sns.distplot(all_times, color="green")
        plt.savefig(iter_dir / f"sim_times.png")
        plt.clf()

        with open(resample_log_path, "a") as f:
            f.write(f"Performed {len(all_sim_dirs)} simulations with Ray...\n")
            f.write(f"Hostname distribution:\n{pprint.pformat(Counter(all_hostnames))}\n")
            f.write(f"Min. time: {onp.min(all_times)}\n")
            f.write(f"Max. time: {onp.max(all_times)}\n")

        for rdir, rc in zip(all_sim_dirs, all_rcs):
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation at path {rdir} failed with error code: {rc}")

        # Convert via TacoxDNA
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=repeat_dir)
                p.wait()
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=repeat_dir)
                p.wait()
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

        # Combine trajectories for each force
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            combine_cmd = "cat "
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                combine_cmd += f"{repeat_dir}/data.oxdna "
            combine_cmd += f"> {sim_dir}/output.dat"
            combine_proc = subprocess.run(combine_cmd, shell=True)
            if combine_proc.returncode != 0:
                raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{force_pn}"
            combine_cmd = "cat "
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                combine_cmd += f"{repeat_dir}/data.oxdna "
            combine_cmd += f"> {sim_dir}/output.dat"
            combine_proc = subprocess.run(combine_cmd, shell=True)
            if combine_proc.returncode != 0:
                raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        # Analyze
        all_force_t0_traj_states = list()
        all_force_t0_calc_energies = list()
        all_force_t0_distances = list()
        all_force_t0_thetas = list()
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"

            ## Load states from oxDNA simulation
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, reindex=True,
                traj_path=sim_dir / "output.dat",
                # reverse_direction=True)
                reverse_direction=False)
            full_traj_states = traj_info.get_states()
            assert(len(full_traj_states) == (1+n_total_states)*n_sims)
            sim_freq = 1+n_total_states
            traj_states = list()
            for r in range(n_sims):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states)
                traj_states += sampled_sim_states
            assert(len(traj_states) == n_sample_states*n_sims)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                log_path = repeat_dir / "log.lammps"
                rpt_log_df = lammps_utils.read_log(log_path)
                assert(rpt_log_df.shape[0] == n_total_states+1)
                rpt_log_df = rpt_log_df[1+n_eq_states:]
                log_dfs.append(rpt_log_df)
            log_df = pd.concat(log_dfs, ignore_index=True)

            ## Generate an energy function
            if seq_avg:
                ss_hb_weights = utils.HB_WEIGHTS_SA
                ss_stack_weights = utils.STACK_WEIGHTS_SA
            else:
                ss_path = "data/seq-specific/seq_oxdna2.txt"
                ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                    ss_path,
                    model.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                    model.default_base_params_seq_dep['stacking']['eps_stack_base'],
                    model.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                    enforce_symmetry=False,
                    t_kelvin=t_kelvin
                )
            em = model.EnergyModel(displacement_fn,
                                   params,
                                   t_kelvin=t_kelvin,
                                   ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                                   salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                                   ignore_exc_vol_bonded=True # Because we're in LAMMPS
            )
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energy_fn = jit(energy_fn)

            ## Compute the energies via our energy function
            energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
            _, calc_energies = scan(energy_scan_fn, None, traj_states)

            ## Check energies
            gt_energies = (log_df.PotEng * seq_oh.shape[0]).to_numpy()
            energy_diffs = list()
            for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                diff = onp.abs(calc - gt)
                energy_diffs.append(diff)


            ## Compute the mean distance
            traj_distances = list()
            for rs_idx in range(n_sample_states*n_sims):
                ref_state = traj_states[rs_idx]
                dist = compute_distance(ref_state)
                traj_distances.append(dist)

            traj_distances = onp.array(traj_distances)

            ## Compute the mean theta
            traj_thetas = list()
            for rs_idx in range(n_sample_states*n_sims):
                ref_state = traj_states[rs_idx]
                theta = compute_theta(ref_state)
                traj_thetas.append(theta)

            traj_thetas = onp.array(traj_thetas)

            ## Record some plots
            plt.plot(traj_distances)
            plt.savefig(sim_dir / "dist_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_distances) / onp.arange(1, (n_sample_states*n_sims)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_dist.png")
            plt.clf()

            last_half = int((n_sample_states * n_sims) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_dist_second_half.png")
            plt.clf()


            plt.plot(traj_thetas)
            plt.savefig(sim_dir / "theta_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_thetas) / onp.arange(1, (n_sample_states*n_sims)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_theta.png")
            plt.clf()

            last_half = int((n_sample_states * n_sims) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_theta_second_half.png")
            plt.clf()

            sns.distplot(calc_energies, label="Calculated", color="red")
            sns.distplot(gt_energies, label="Reference", color="green")
            plt.legend()
            plt.savefig(sim_dir / f"energies.png")
            plt.clf()

            sns.histplot(energy_diffs)
            plt.savefig(sim_dir / f"energy_diffs.png")
            plt.clf()

            with open(sim_dir / "summary.txt", "w+") as f:
                f.write(f"Mean dist: {onp.mean(traj_distances)}\n")
                f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
                f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
                f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

            all_force_t0_traj_states.append(traj_states)
            all_force_t0_calc_energies.append(calc_energies)
            all_force_t0_distances.append(traj_distances)
            all_force_t0_thetas.append(traj_thetas)


        all_force_t0_traj_states = utils.tree_stack(all_force_t0_traj_states)
        all_force_t0_calc_energies = utils.tree_stack(all_force_t0_calc_energies)
        all_force_t0_distances = utils.tree_stack(all_force_t0_distances)
        all_force_t0_thetas = utils.tree_stack(all_force_t0_thetas)


        all_f2_torque_traj_states = list()
        all_f2_torque_calc_energies = list()
        all_f2_torque_distances = list()
        all_f2_torque_thetas = list()
        for torque_pnnm in torque_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"

            ## Load states from oxDNA simulation
            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_file=True, reindex=True,
                traj_path=sim_dir / "output.dat",
                # reverse_direction=True)
                reverse_direction=False)
            full_traj_states = traj_info.get_states()
            assert(len(full_traj_states) == (1+n_total_states)*n_sims)
            sim_freq = 1+n_total_states
            traj_states = list()
            for r in range(n_sims):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states)
                traj_states += sampled_sim_states
            assert(len(traj_states) == n_sample_states*n_sims)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                log_path = repeat_dir / "log.lammps"
                rpt_log_df = lammps_utils.read_log(log_path)
                assert(rpt_log_df.shape[0] == n_total_states+1)
                rpt_log_df = rpt_log_df[1+n_eq_states:]
                log_dfs.append(rpt_log_df)
            log_df = pd.concat(log_dfs, ignore_index=True)

            ## Generate an energy function
            if seq_avg:
                ss_hb_weights = utils.HB_WEIGHTS_SA
                ss_stack_weights = utils.STACK_WEIGHTS_SA
            else:
                ss_path = "data/seq-specific/seq_oxdna2.txt"
                ss_hb_weights, ss_stack_weights = read_ss_oxdna(
                    ss_path,
                    model.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
                    model.default_base_params_seq_dep['stacking']['eps_stack_base'],
                    model.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
                    enforce_symmetry=False,
                    t_kelvin=t_kelvin
                )
            em = model.EnergyModel(displacement_fn,
                                   params,
                                   t_kelvin=t_kelvin,
                                   ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                                   salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                                   ignore_exc_vol_bonded=True # Because we're in LAMMPS
            )
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            energy_fn = jit(energy_fn)

            ## Compute the energies via our energy function
            energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
            _, calc_energies = scan(energy_scan_fn, None, traj_states)

            ## Check energies
            gt_energies = (log_df.PotEng * seq_oh.shape[0]).to_numpy()
            energy_diffs = list()
            for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                diff = onp.abs(calc - gt)
                energy_diffs.append(diff)


            ## Compute the mean distance
            traj_distances = list()
            for rs_idx in range(n_sample_states*n_sims):
                ref_state = traj_states[rs_idx]
                dist = compute_distance(ref_state)
                traj_distances.append(dist)

            traj_distances = onp.array(traj_distances)

            ## Compute the mean theta
            traj_thetas = list()
            for rs_idx in range(n_sample_states*n_sims):
                ref_state = traj_states[rs_idx]
                theta = compute_theta(ref_state)
                traj_thetas.append(theta)

            traj_thetas = onp.array(traj_thetas)

            ## Record some plots
            plt.plot(traj_distances)
            plt.savefig(sim_dir / "dist_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_distances) / onp.arange(1, (n_sample_states*n_sims)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_dist.png")
            plt.clf()

            last_half = int((n_sample_states * n_sims) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_dist_second_half.png")
            plt.clf()


            plt.plot(traj_thetas)
            plt.savefig(sim_dir / "theta_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_thetas) / onp.arange(1, (n_sample_states*n_sims)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_theta.png")
            plt.clf()

            last_half = int((n_sample_states * n_sims) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_theta_second_half.png")
            plt.clf()

            sns.distplot(calc_energies, label="Calculated", color="red")
            sns.distplot(gt_energies, label="Reference", color="green")
            plt.legend()
            plt.savefig(sim_dir / f"energies.png")
            plt.clf()

            sns.histplot(energy_diffs)
            plt.savefig(sim_dir / f"energy_diffs.png")
            plt.clf()

            with open(sim_dir / "summary.txt", "w+") as f:
                f.write(f"Mean dist: {onp.mean(traj_distances)}\n")
                f.write(f"Mean theta: {onp.mean(traj_thetas)}\n")
                f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
                f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
                f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

            all_f2_torque_traj_states.append(traj_states)
            all_f2_torque_calc_energies.append(calc_energies)
            all_f2_torque_distances.append(traj_distances)
            all_f2_torque_thetas.append(traj_thetas)

        all_f2_torque_traj_states = utils.tree_stack(all_f2_torque_traj_states)
        all_f2_torque_calc_energies = utils.tree_stack(all_f2_torque_calc_energies)
        all_f2_torque_distances = utils.tree_stack(all_f2_torque_distances)
        all_f2_torque_thetas = utils.tree_stack(all_f2_torque_thetas)

        # Compute constants
        mean_force_t0_distances = [all_force_t0_distances[f_idx].mean() for f_idx in range(len(forces_pn))]
        mean_force_t0_distances_nm = mean_force_t0_distances * utils.nm_per_oxdna_length
        l0 = mean_force_t0_distances_nm[0]
        theta0 = all_force_t0_thetas[0].mean()
        force_t0_delta_ls = mean_force_t0_distances_nm - l0 # in nm

        ## For A1, we assume an offset of 0
        xs_to_fit = jnp.stack([jnp.zeros_like(forces_pn), forces_pn], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, force_t0_delta_ls)
        a1 = fit_[0][1]

        test_forces = onp.linspace(0, forces_pn.max(), 100)
        fit_fn = lambda val: a1*val
        plt.plot(test_forces, fit_fn(test_forces))
        plt.scatter(forces_pn, forces_t0_delta_ls)
        plt.xlabel("Force (pN)")
        plt.ylabel("deltaL (nm)")
        plt.title(f"A1={a1}")
        plt.savefig(iter_dir / "a1_fit.png")

        ## Compute A3 -- fit with an unrestricted offset
        mean_f2_torque_distances = [all_f2_torque_distances[f_idx].mean() for f_idx in range(len(forces_pn))]
        mean_f2_torque_distances_nm = mean_f2_torque_distances * utils.nm_per_oxdna_length
        f2_torque_delta_ls = mean_f2_torque_distances_nm - l0 # in nm

        xs_to_fit = jnp.stack([jnp.ones_like(torques_pnnm), torques_pnnm], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, f2_torque_delta_ls)
        a3 = fit_[0][1]
        a3_offset = fit_[0][0]

        test_torques = onp.linspace(0, torques_pnnm.max(), 100)
        fit_fn = lambda val: a3*val + a3_offset
        plt.plot(test_torques, fit_fn(test_torques))
        plt.scatter(torques_pnnm, f2_torque_delta_ls)
        plt.xlabel("Torques (pN*nm)")
        plt.ylabel("deltaL (nm)")
        plt.title(f"A3={a3}")
        plt.savefig(iter_dir / "a3_fit.png")

        ## Compute A4 -- fit with an unrestricted offset
        mean_f2_torque_thetas = [all_f2_torque_thetas[f_idx].mean() for f_idx in range(len(forces_pn))]
        f2_torque_delta_thetas = mean_f2_torque_thetas - theta0

        xs_to_fit = jnp.stack([jnp.ones_like(torques_pnnm), torques_pnnm], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, f2_torque_delta_thetas)
        a4 = fit_[0][1]
        a4_offset = fit_[0][0]

        fit_fn = lambda val: a4*val + a4_offset
        plt.plot(test_torques, fit_fn(test_torques))
        plt.scatter(torques_pnnm, f2_torque_delta_thetas)
        plt.xlabel("Torques (pN*nm)")
        plt.ylabel("deltaTheta (rad)")
        plt.title(f"A4={a4}")
        plt.savefig(iter_dir / "a4_fit.png")

        s_eff = l0 / a1
        c = a1 * l0 / (a4*a1 - a3**2)
        g = -(a3 * l0) / (a4 * a1 - a3**2)

        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"A1: {a1}\n")
            f.write(f"A3: {a3}\n")
            f.write(f"A4: {a4}\n")
            f.write(f"S_eff: {s_eff}\n")
            f.write(f"C: {c}\n")
            f.write(f"g: {g}\n")

        return all_force_t0_traj_states, all_force_t0_calc_energies, all_force_t0_distances, all_f2_torque_traj_states, all_f2_torque_calc_energies, all_f2_torque_distances, all_f2_torque_thetas



    params = deepcopy(model.EMPTY_BASE_PARAMS)
    if seq_avg:
        params["stacking"] = deepcopy(model.default_base_params_seq_avg["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_avg["hydrogen_bonding"])
        # params["cross_stacking"] = deepcopy(model.default_base_params_seq_avg["cross_stacking"])
        # del params["cross_stacking"]["dr_c_cross"]
        # del params["cross_stacking"]["dr_low_cross"]
        # del params["cross_stacking"]["dr_high_cross"]
    else:
        params["stacking"] = deepcopy(model.default_base_params_seq_dep["stacking"])
        params["hydrogen_bonding"] = deepcopy(model.default_base_params_seq_dep["hydrogen_bonding"])
        # params["cross_stacking"] = deepcopy(model.default_base_params_seq_dep["cross_stacking"])
        # del params["cross_stacking"]["dr_c_cross"]
        # del params["cross_stacking"]["dr_low_cross"]
        # del params["cross_stacking"]["dr_high_cross"]

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # min_n_eff = int(2*n_sample_states * min_neff_factor) # 2x because we simulate at two different forces
    min_n_eff = int(n_sample_states * min_neff_factor) # 2x because we simulate at two different forces

    all_losses = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    all_ref_states_f, all_ref_energies_f, all_ref_dist_f, all_ref_states_t, all_ref_energies_t, all_ref_dist_t, all_ref_thetas_t = get_ref_states(params, i=0, seed=30362)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")



def get_parser():

    parser = argparse.ArgumentParser(description="Optimize the length under a pulling force via LAMMPS")

    parser.add_argument('--n-sample-steps', type=int, default=3000000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=100000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=500,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of simulations per force")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--lammps-basedir', type=str,
                        default="/n/brenner_lab/Lab/software/lammps-stable_29Sep2021",
                        help='LAMMPS base directory')
    parser.add_argument('--tacoxdna-basedir', type=str,
                        default="/n/brenner_lab/User/rkrueger/tacoxDNA",
                        help='tacoxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--rmse-coeff', type=float, default=100.0,
                        help="Coefficient for the RMSE")
    parser.add_argument('--seq-dep', action='store_true')


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
