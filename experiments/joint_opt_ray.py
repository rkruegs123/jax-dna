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
from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna2 import model, lammps_utils
import jax_dna.input.trajectory as jdt
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.dna2.oxdna_utils import recompile_oxdna
from jax_dna.dna1 import model as model1


if "ip_head" in os.environ:
    ray.init(address=os.environ["ip_head"])
else:
    ray.init()


checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


## Helper functions for moduli calculation in LAMMPS

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

@ray.remote
def run_oxdna_ray(oxdna_exec_path, sim_dir):
    time.sleep(1)

    hostname = socket.gethostbyname(socket.gethostname())

    start = time.time()
    p = subprocess.Popen([oxdna_exec_path, sim_dir / "input"])
    p.wait()
    end = time.time()

    rc = p.returncode
    return rc, end-start, hostname


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

def rmse_uncertainty(val, lo_val, hi_val):
    mse = jnp.where(val < lo_val, (val - lo_val)**2,
                    jnp.where(val > hi_val, (val - hi_val)**2,
                              0.0))
    return jnp.sqrt(mse)



def run(args):

    # General arguments
    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    seq_avg = not args['seq_dep']
    assert(seq_avg)

    opt_keys = args['opt_keys']
    n_threads = args['n_threads']

    lammps_basedir = Path(args['lammps_basedir'])
    assert(lammps_basedir.exists())
    lammps_exec_path = lammps_basedir / "build/lmp"
    assert(lammps_exec_path.exists())

    tacoxdna_basedir = Path(args['tacoxdna_basedir'])
    assert(tacoxdna_basedir.exists())

    no_archive = args['no_archive']
    no_delete = args['no_delete']
    save_obj_every = args['save_obj_every']
    plot_every = args['plot_every']
    ignore_warnings = args['ignore_warnings']

    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"


    # Moduli/LAMMPS arguments
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

    s_eff_coeff = args['s_eff_coeff']
    c_coeff = args['c_coeff']
    g_coeff = args['g_coeff']

    target_s_eff = args['target_s_eff']
    target_c = args['target_c']
    target_g = args['target_g']

    s_eff_uncertainty = args['s_eff_uncertainty']
    s_eff_hi = target_s_eff + s_eff_uncertainty
    s_eff_lo = target_s_eff - s_eff_uncertainty

    c_uncertainty = args['c_uncertainty']
    c_hi = target_c + c_uncertainty
    c_lo = target_c - c_uncertainty

    g_uncertainty = args['g_uncertainty']
    g_hi = target_g + g_uncertainty
    g_lo = target_g - g_uncertainty

    timestep = args['timestep']

    forces_pn = jnp.array(args['forces_pn'], dtype=jnp.float64)
    torques_pnnm = jnp.array(args['torques_pnnm'], dtype=jnp.float64)


    # Structural (60 bp) arguments
    n_sims_struc = args['n_sims_struc']
    n_steps_per_sim_struc = args['n_steps_per_sim_struc']
    n_eq_steps_struc = args['n_eq_steps_struc']
    sample_every_struc = args['sample_every_struc']
    assert(n_steps_per_sim_struc % sample_every_struc == 0)
    n_ref_states_per_sim_struc = n_steps_per_sim_struc // sample_every_struc
    n_ref_states_struc = n_ref_states_per_sim_struc * n_sims_struc
    offset_struc = args['offset_struc']


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
    neffs_path = log_dir / "neffs.txt"
    a1_path = log_dir / "a1.txt"
    a3_path = log_dir / "a3.txt"
    a4_path = log_dir / "a4.txt"
    s_eff_path = log_dir / "s_eff.txt"
    c_path = log_dir / "c.txt"
    g_path = log_dir / "g.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"
    warnings_path = log_dir / "warnings.txt"

    params_str = ""
    params_str += f"n_sample_states: {n_sample_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Setup systems

    ## Setup structural system (60 bp)

    sys_basedir = Path("data/templates/simple-helix-60bp")
    input_template_path = sys_basedir / "input"

    top_path_struc = sys_basedir / "sys.top"
    top_info_struc = topology.TopologyInfo(top_path_struc, reverse_direction=False)
    seq_oh_struc = jnp.array(utils.get_one_hot(top_info_struc.seq), dtype=jnp.float64)

    quartets_struc = utils.get_all_quartets(n_nucs_per_strand=seq_oh_struc.shape[0] // 2)
    quartets_struc = quartets_struc[offset_struc:-offset_struc-1]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info_struc,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info_struc, conf_info)

    dt_struc = 5e-3
    t_kelvin_struc = utils.DEFAULT_TEMP
    kT_struc = utils.get_kt(t_kelvin_struc)
    beta_struc = 1 / kT_struc




    def get_struc_tasks(iter_dir, params, prev_basedir):
        recompile_start = time.time()
        recompile_oxdna(params, oxdna_path, t_kelvin_struc, num_threads=n_threads)
        recompile_end = time.time()

        struc_dir = iter_dir / "struc"
        struc_dir.mkdir(parents=False, exist_ok=False)

        all_repeat_dirs = list()
        for r in range(n_sims_struc):
            repeat_dir = struc_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            all_repeat_dirs.append(repeat_dir)

            shutil.copy(top_path_struc, repeat_dir / "sys.top")

            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info_struc,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info_struc, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh_struc.shape[0], r*n_steps_per_sim_struc)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin_struc}K", steps=n_steps_per_sim_struc,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_struc, seed=random.randrange(100),
                equilibration_steps=n_eq_steps_struc, dt=dt_struc,
                no_stdout_energy=0, backend="CPU",
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="DNA2_nomesh"
            )
        struc_tasks = [run_oxdna_ray.remote(oxdna_exec_path, rdir) for rdir in all_repeat_dirs]
        return struc_tasks, all_repeat_dirs



    ## Setup LAMMPS stretch/torsionn system
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
    seq = top_info.seq
    n = seq_oh.shape[0]
    assert(n % 2 == 0)
    n_bp = n // 2
    strand_length = int(seq_oh.shape[0] // 2)

    strand1_start = 0
    strand1_end = n_bp-1
    strand2_start = n_bp
    strand2_end = n_bp*2-1

    ### The region for which theta and distance are measured
    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[4:n_bp-5]

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]

    displacement_fn, shift_fn = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.

    @jit
    def compute_distance(body):
        bp1_meas_pos = get_bp_pos(body, bp1_meas)
        bp2_meas_pos = get_bp_pos(body, bp2_meas)
        dist = jnp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
        return dist

    @jit
    def compute_theta(body):
        pitches = compute_pitches(body, quartets, displacement_fn, model.com_to_hb)
        return pitches.sum()

    t_kelvin = 300.0
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    salt_conc = 0.15
    q_eff = 0.815

    def get_stretch_tors_tasks(iter_dir, params, prev_states_force, prev_states_torque):
        all_sim_dirs = list()
        repeat_seeds = [random.randrange(1, 100) for _ in range(n_sims)]
        for f_idx, force_pn in enumerate(forces_pn):
            sim_dir = iter_dir / f"sim-f{force_pn}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                # repeat_seed = random.randrange(100)
                repeat_seed = repeat_seeds[r]

                if prev_states_force is None:
                    shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                else:
                    print(type(prev_states_force))
                    print(len(prev_states_force))
                    print(type(prev_states_force[f_idx]))
                    print(len(prev_states_force[f_idx]))
                    print(type(prev_states_force[f_idx][r]))
                    lammps_utils.stretch_tors_data_constructor(prev_states_force[f_idx][r], seq, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                    force_pn=force_pn, torque_pnnm=0,
                    save_every=sample_every, n_steps=n_total_steps,
                    seq_avg=seq_avg, seed=repeat_seed, timestep=timestep)

        for t_idx, torque_pnnm in enumerate(torques_pnnm):
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                repeat_seed = repeat_seeds[r]

                if prev_states_torque is None:
                    shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                else:
                    lammps_utils.stretch_tors_data_constructor(prev_states_torque[t_idx][r], seq, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT, salt_conc=salt_conc, qeff=q_eff,
                    force_pn=2.0, torque_pnnm=torque_pnnm,
                    save_every=sample_every, n_steps=n_total_steps,
                    seq_avg=seq_avg, seed=repeat_seed)

        stretch_tors_tasks = [run_lammps_ray.remote(lammps_exec_path, rdir) for rdir in all_sim_dirs]
        return stretch_tors_tasks, all_sim_dirs


    def process_stretch_tors(iter_dir, params):
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
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            combine_cmd = "cat "
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                combine_cmd += f"{repeat_dir}/data.oxdna "
            combine_cmd += f"> {sim_dir}/output.dat"
            combine_proc = subprocess.run(combine_cmd, shell=True)
            if combine_proc.returncode != 0:
                raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        # Remove unnecessary files
        if not no_delete:
            files_to_remove = ["filename.dat", "data.oxdna", "dump.lammpstrj"]
            for force_pn in forces_pn:
                sim_dir = iter_dir / f"sim-f{force_pn}"
                for r in range(n_sims):
                    repeat_dir = sim_dir / f"r{r}"
                    for f_stem in files_to_remove:
                        file_to_rem = repeat_dir / f_stem
                        file_to_rem.unlink()

            for torque_pnnm in torques_pnnm:
                sim_dir = iter_dir / f"sim-t{torque_pnnm}"
                for r in range(n_sims):
                    repeat_dir = sim_dir / f"r{r}"
                    for f_stem in files_to_remove:
                        file_to_rem = repeat_dir / f_stem
                        file_to_rem.unlink()

        # Analyze
        all_force_t0_traj_states = list()
        all_force_t0_calc_energies = list()
        all_force_t0_distances = list()
        all_force_t0_thetas = list()
        all_force_t0_last_states = list()
        running_avgs_force_dists = list()
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"

            ## Load states from oxDNA simulation

            traj_ = jdt.from_file(
                sim_dir / "output.dat",
                [strand_length, strand_length],
                is_oxdna=False,
                n_processes=n_threads,
            )
            full_traj_states = [ns.to_rigid_body() for ns in traj_.states]

            assert(len(full_traj_states) == (1+n_total_states)*n_sims)
            sim_freq = 1+n_total_states
            traj_states = list()
            force_last_states = list()
            for r in range(n_sims):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states)
                traj_states += sampled_sim_states
                force_last_states.append(sampled_sim_states[-1])
            assert(len(traj_states) == n_sample_states*n_sims)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                log_path = repeat_dir / "log.lammps"
                rpt_log_df = lammps_utils.read_log(log_path)
                if ignore_warnings:
                    n_row_full = rpt_log_df.shape[0]
                    rpt_log_df = rpt_log_df[rpt_log_df.v_tns != "WARNING:"]
                    n_row_no_warnings = rpt_log_df.shape[0]
                    n_warnings = n_row_full - n_row_no_warnings
                    if n_warnings > 0:
                        with open(warnings_path, "a") as f:
                            f.write(f"Ignored {n_warnings} at the {i}th iteration, force {force_pn}, repeat {r}...\n")
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
            running_avgs_force_dists.append(running_avg)

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
            all_force_t0_last_states.append(force_last_states)


        all_force_t0_traj_states = utils.tree_stack(all_force_t0_traj_states)
        all_force_t0_calc_energies = utils.tree_stack(all_force_t0_calc_energies)
        all_force_t0_distances = utils.tree_stack(all_force_t0_distances)
        all_force_t0_thetas = utils.tree_stack(all_force_t0_thetas)

        # Compute running avg of a1, l0, and s_eff
        running_avgs_force_dists = onp.array(running_avgs_force_dists) # (n_forces, n_sample_states*n_sims)
        running_avg_idxs = onp.arange(n_sample_states*n_sims)
        n_running_avg_points = 100
        check_every = (n_sample_states*n_sims) // n_running_avg_points
        check_idxs = onp.arange(n_running_avg_points) * check_every
        a1_running_avgs = list()
        l0_fit_running_avgs = list()
        s_eff_running_avgs = list()
        for check_idx in check_idxs:
            curr_force_dists = running_avgs_force_dists[:, check_idx]
            curr_force_dists_nm = curr_force_dists * utils.nm_per_oxdna_length

            # Compute a1 and l0
            xs_to_fit = jnp.stack([jnp.ones_like(forces_pn), forces_pn], axis=1)
            fit_ = jnp.linalg.lstsq(xs_to_fit, curr_force_dists_nm)

            curr_a1 = fit_[0][1]
            a1_running_avgs.append(curr_a1)

            curr_l0_fit = fit_[0][0]
            l0_fit_running_avgs.append(curr_l0_fit)

            curr_s_eff = curr_l0_fit / curr_a1
            s_eff_running_avgs.append(curr_s_eff)

        plt.plot(check_idxs, a1_running_avgs)
        plt.scatter(check_idxs, a1_running_avgs)
        plt.savefig(iter_dir / "a1_running_avg.png")
        plt.close()

        plt.plot(check_idxs, l0_fit_running_avgs)
        plt.scatter(check_idxs, l0_fit_running_avgs)
        plt.savefig(iter_dir / "l0_fit_running_avg.png")
        plt.close()

        plt.plot(check_idxs, s_eff_running_avgs)
        plt.scatter(check_idxs, s_eff_running_avgs)
        plt.savefig(iter_dir / "s_eff_running_avg.png")
        plt.close()


        all_f2_torque_traj_states = list()
        all_f2_torque_calc_energies = list()
        all_f2_torque_distances = list()
        all_f2_torque_thetas = list()
        all_f2_torque_last_states = list()
        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"

            ## Load states from oxDNA simulation

            traj_ = jdt.from_file(
                sim_dir / "output.dat",
                [strand_length, strand_length],
                is_oxdna=False,
                n_processes=n_threads,
            )
            full_traj_states = [ns.to_rigid_body() for ns in traj_.states]


            assert(len(full_traj_states) == (1+n_total_states)*n_sims)
            sim_freq = 1+n_total_states
            traj_states = list()
            torque_last_states = list()
            for r in range(n_sims):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states)
                traj_states += sampled_sim_states
                torque_last_states.append(sampled_sim_states[-1])
            assert(len(traj_states) == n_sample_states*n_sims)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims):
                repeat_dir = sim_dir / f"r{r}"
                log_path = repeat_dir / "log.lammps"
                rpt_log_df = lammps_utils.read_log(log_path)
                if ignore_warnings:
                    n_row_full = rpt_log_df.shape[0]
                    rpt_log_df = rpt_log_df[rpt_log_df.v_tns != "WARNING:"]
                    n_row_no_warnings = rpt_log_df.shape[0]
                    n_warnings = n_row_full - n_row_no_warnings
                    if n_warnings > 0:
                        with open(warnings_path, "a") as f:
                            f.write(f"Ignored {n_warnings} at the {i}th iteration, torque {torque_pnnm}, repeat {r}...\n")
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
            all_f2_torque_last_states.append(torque_last_states)

        all_f2_torque_traj_states = utils.tree_stack(all_f2_torque_traj_states)
        all_f2_torque_calc_energies = utils.tree_stack(all_f2_torque_calc_energies)
        all_f2_torque_distances = utils.tree_stack(all_f2_torque_distances)
        all_f2_torque_thetas = utils.tree_stack(all_f2_torque_thetas)

        # Compute constants
        mean_force_t0_distances = onp.array([all_force_t0_distances[f_idx].mean() for f_idx in range(len(forces_pn))])
        mean_force_t0_distances_nm = mean_force_t0_distances * utils.nm_per_oxdna_length

        ## For A1, we do not assume and offset of 0 and *fit* l0 (rather than take distance under 0 force)
        xs_to_fit = jnp.stack([jnp.ones_like(forces_pn), forces_pn], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, mean_force_t0_distances_nm)
        a1 = fit_[0][1]
        l0_fit = fit_[0][0]
        a1_fit_residual = fit_[1][0]

        test_forces = onp.linspace(0, forces_pn.max(), 100)
        fit_fn = lambda val: a1*val + l0_fit
        plt.plot(test_forces, fit_fn(test_forces))
        plt.scatter(forces_pn, mean_force_t0_distances_nm)
        plt.xlabel("Force (pN)")
        plt.ylabel("L (nm)")
        plt.title(f"A1={a1}, L0={l0_fit}")
        plt.savefig(iter_dir / "a1_fit.png")
        plt.clf()

        ## Compute A3 -- fit with an unrestricted offset
        mean_f2_torque_distances = onp.array([all_f2_torque_distances[t_idx].mean() for t_idx in range(len(torques_pnnm))])
        mean_f2_torque_distances_nm = mean_f2_torque_distances * utils.nm_per_oxdna_length

        xs_to_fit = jnp.stack([jnp.ones_like(torques_pnnm), torques_pnnm], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, mean_f2_torque_distances_nm)
        a3 = fit_[0][1]
        a3_offset = fit_[0][0]
        a3_fit_residual = fit_[1][0]

        test_torques = onp.linspace(0, torques_pnnm.max(), 100)
        fit_fn = lambda val: a3*val + a3_offset
        plt.plot(test_torques, fit_fn(test_torques))
        plt.scatter(torques_pnnm, mean_f2_torque_distances_nm)
        plt.xlabel("Torques (pN*nm)")
        plt.ylabel("L (nm)")
        plt.title(f"A3={a3}")
        plt.savefig(iter_dir / "a3_fit.png")
        plt.clf()

        ## Compute A4 -- fit with an unrestricted offset
        mean_f2_torque_thetas = onp.array([all_f2_torque_thetas[t_idx].mean() for t_idx in range(len(torques_pnnm))])

        xs_to_fit = jnp.stack([jnp.ones_like(torques_pnnm), torques_pnnm], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, mean_f2_torque_thetas)
        a4 = fit_[0][1]
        a4_offset = fit_[0][0]
        a4_fit_residual = fit_[1][0]

        fit_fn = lambda val: a4*val + a4_offset
        plt.plot(test_torques, fit_fn(test_torques))
        # plt.scatter(torques_pnnm, f2_torque_delta_thetas)
        plt.scatter(torques_pnnm, mean_f2_torque_thetas)
        plt.xlabel("Torques (pN*nm)")
        plt.ylabel("Theta (rad)")
        plt.title(f"A4={a4}")
        plt.savefig(iter_dir / "a4_fit.png")
        plt.clf()

        s_eff = l0_fit / a1
        c = a1 * l0_fit / (a4*a1 - a3**2)
        g = -(a3 * l0_fit) / (a4 * a1 - a3**2)

        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"A1: {a1}\n")
            f.write(f"A3: {a3}\n")
            f.write(f"A4: {a4}\n")
            f.write(f"S_eff: {s_eff}\n")
            f.write(f"C: {c}\n")
            f.write(f"g: {g}\n")
            f.write(f"A1 residual: {a1_fit_residual}\n")
            f.write(f"A3 residual: {a3_fit_residual}\n")
            f.write(f"A4 residual: {a4_fit_residual}\n")

        ref_info = (all_force_t0_traj_states, all_force_t0_calc_energies, all_force_t0_distances,
                    all_force_t0_thetas, all_f2_torque_traj_states, all_f2_torque_calc_energies,
                    all_f2_torque_distances, all_f2_torque_thetas)
        return ref_info, all_force_t0_last_states, all_f2_torque_last_states



    def get_ref_states(params, i, seed, prev_states_force, prev_states_torque, prev_basedir):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        # Run the simulations
        stretch_tors_tasks, all_sim_dirs = get_stretch_tors_tasks(iter_dir, params, prev_states_force, prev_states_torque)
        struc_tasks, all_sim_dirs_struc = get_struc_tasks(iter_dir, params, prev_basedir)

        ## Archive the previous basedir now that we've loaded states from it
        if not no_archive:
            shutil.make_archive(prev_basedir, 'zip', prev_basedir)
            shutil.rmtree(prev_basedir)

        all_ret_info = ray.get(stretch_tors_tasks)
        all_ret_info_struc = ray.get(struc_tasks) # FIXME: for now, not doing anything with this! Just want to run the simulations and see them. Then, we do analysis and what not.

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


        stretch_tors_ref_info, all_force_t0_last_states, all_f2_torque_last_states = process_stretch_tors(iter_dir, params)


        return stretch_tors_ref_info, all_force_t0_last_states, all_f2_torque_last_states, iter_dir

    @jit
    def loss_fn(params, stretch_tors_ref_info):

        all_ref_states_f, all_ref_energies_f, all_ref_dists_f, all_ref_thetas_f, all_ref_states_t, all_ref_energies_t, all_ref_dists_t, all_ref_thetas_t = stretch_tors_ref_info

        # Setup energy function
        em = model.EnergyModel(displacement_fn,
                               params,
                               t_kelvin=t_kelvin,
                               salt_conc=salt_conc, q_eff=q_eff, seq_avg=seq_avg,
                               ignore_exc_vol_bonded=True # Because we're in LAMMPS
        )
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

        def get_expected_vals(ref_states, ref_energies, ref_dists, ref_thetas):
            _, new_energies = scan(energy_scan_fn, None, ref_states)

            diffs = new_energies - ref_energies
            boltzs = jnp.exp(-beta * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom

            expected_dist = jnp.dot(weights, ref_dists)
            expected_theta = jnp.dot(weights, ref_thetas)
            n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
            return expected_dist, expected_theta, n_eff

        expected_dists_f, expected_thetas_f, n_effs_f = vmap(get_expected_vals, (0, 0, 0, 0))(all_ref_states_f, all_ref_energies_f, all_ref_dists_f, all_ref_thetas_f)
        expected_dists_f_nm = expected_dists_f * utils.nm_per_oxdna_length

        xs_to_fit = jnp.stack([jnp.ones_like(forces_pn), forces_pn], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, expected_dists_f_nm)
        a1 = fit_[0][1]
        l0_fit = fit_[0][0]


        expected_dists_t, expected_thetas_t, n_effs_t = vmap(get_expected_vals, (0, 0, 0, 0))(all_ref_states_t, all_ref_energies_t, all_ref_dists_t, all_ref_thetas_t)
        expected_dists_t_nm = expected_dists_t * utils.nm_per_oxdna_length

        xs_to_fit = jnp.stack([jnp.ones_like(torques_pnnm), torques_pnnm], axis=1)
        fit_ = jnp.linalg.lstsq(xs_to_fit, expected_dists_t_nm)
        a3 = fit_[0][1]

        fit_ = jnp.linalg.lstsq(xs_to_fit, expected_thetas_t)
        a4 = fit_[0][1]

        s_eff = l0_fit / a1
        c = a1 * l0_fit / (a4*a1 - a3**2)
        g = -(a3 * l0_fit) / (a4 * a1 - a3**2)

        rmse_s_eff = rmse_uncertainty(s_eff, s_eff_lo, s_eff_hi)
        rmse_c = rmse_uncertainty(c, c_lo, c_hi)
        rmse_g = rmse_uncertainty(g, g_lo, g_hi)

        rmse = s_eff_coeff*rmse_s_eff + c_coeff*rmse_c + g_coeff*rmse_g

        return rmse, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)


    params = deepcopy(model.EMPTY_BASE_PARAMS)
    for opt_key in opt_keys:
        if seq_avg:
            params[opt_key] = deepcopy(model.default_base_params_seq_avg[opt_key])
        else:
            params[opt_key] = deepcopy(model.default_base_params_seq_dep[opt_key])


    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    min_n_eff = int(n_sample_states*n_sims * min_neff_factor)

    all_losses = list()
    all_n_effs = list()
    all_seffs = list()
    all_cs = list()
    all_gs = list()

    all_ref_losses = list()
    all_ref_times = list()
    all_ref_seffs = list()
    all_ref_cs = list()
    all_ref_gs = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    prev_ref_basedir = None
    stretch_tors_ref_info, prev_last_states_force, prev_last_states_torque, ref_iter_dir = get_ref_states(params, i=0, seed=30362, prev_states_force=None, prev_states_torque=None, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (loss, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g)), grads = grad_fn(params, stretch_tors_ref_info)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_seffs.append(s_eff)
            all_ref_cs.append(c)
            all_ref_gs.append(g)

        resample = False
        n_effs = jnp.concatenate([n_effs_f, n_effs_t])
        for n_eff in n_effs:
            if n_eff < min_n_eff:
                resample = True
                break

        if resample or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- min n_eff was {n_effs.min()}...")

            start = time.time()
            stretch_tors_ref_info, prev_last_states_force, prev_last_states_torque, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_states_force=prev_last_states_force, prev_states_torque=prev_last_states_torque, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g)), grads = grad_fn(params, stretch_tors_ref_info)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_seffs.append(s_eff)
            all_ref_cs.append(c)
            all_ref_gs.append(g)

        iter_end = time.time()


        with open(g_path, "a") as f:
            f.write(f"{g}\n")
        with open(s_eff_path, "a") as f:
            f.write(f"{s_eff}\n")
        with open(c_path, "a") as f:
            f.write(f"{c}\n")
        with open(a1_path, "a") as f:
            f.write(f"{a1}\n")
        with open(a3_path, "a") as f:
            f.write(f"{a3}\n")
        with open(a4_path, "a") as f:
            f.write(f"{a4}\n")
        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neffs_path, "a") as f:
            f.write(f"{n_effs}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")

        iter_params_str = f"\nIteration {i}:"
        for k, v in params.items():
            iter_params_str += f"\n- {k}"
            for vk, vv in v.items():
                iter_params_str += f"\n\t- {vk}: {vv}"
        with open(iter_params_path, "a") as f:
            f.write(iter_params_str)

        grads_str = f"\nIteration {i}:"
        for k, v in grads.items():
            grads_str += f"\n- {k}"
            for vk, vv in v.items():
                grads_str += f"\n\t- {vk}: {vv}"
        with open(grads_path, "a") as f:
            f.write(grads_str)

        all_losses.append(loss)
        all_seffs.append(s_eff)
        all_cs.append(c)
        all_gs.append(g)
        all_n_effs.append(n_effs)

        if i % plot_every == 0 and i:
            plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_cs, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_cs, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_c, linestyle='--', label="Target C", color='red')
            plt.axhline(y=c_lo, linestyle='--', color='green')
            plt.axhline(y=c_hi, linestyle='--', color='green')
            plt.legend()
            plt.ylabel("C (pn*nm^2)")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"c_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_gs, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_gs, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_g, linestyle='--', label="Target g", color='red')
            plt.axhline(y=g_lo, linestyle='--', color='green')
            plt.axhline(y=g_hi, linestyle='--', color='green')
            plt.legend()
            plt.ylabel("g (pn*nm)")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"g_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_seffs, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_seffs, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_s_eff, linestyle='--', label="Target S_eff", color='red')
            plt.axhline(y=s_eff_lo, linestyle='--', color='green')
            plt.axhline(y=s_eff_hi, linestyle='--', color='green')
            plt.legend()
            plt.ylabel("S_eff pN")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"seff_iter{i}.png")
            plt.clf()


        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)

            onp.save(obj_dir / f"ref_seffs_i{i}.npy", onp.array(all_ref_seffs), allow_pickle=False)
            onp.save(obj_dir / f"seffs_i{i}.npy", onp.array(all_seffs), allow_pickle=False)

            onp.save(obj_dir / f"ref_gs_i{i}.npy", onp.array(all_ref_gs), allow_pickle=False)
            onp.save(obj_dir / f"gs_i{i}.npy", onp.array(all_gs), allow_pickle=False)

            onp.save(obj_dir / f"ref_cs_i{i}.npy", onp.array(all_ref_cs), allow_pickle=False)
            onp.save(obj_dir / f"cs_i{i}.npy", onp.array(all_cs), allow_pickle=False)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_seffs.npy", onp.array(all_ref_seffs), allow_pickle=False)
    onp.save(obj_dir / f"fin_seffs.npy", onp.array(all_seffs), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_gs.npy", onp.array(all_ref_gs), allow_pickle=False)
    onp.save(obj_dir / f"fin_gs.npy", onp.array(all_gs), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_cs.npy", onp.array(all_ref_cs), allow_pickle=False)
    onp.save(obj_dir / f"fin_cs.npy", onp.array(all_cs), allow_pickle=False)



def get_parser():

    parser = argparse.ArgumentParser(description="Joint optimization for DNA2")

    # General arguments
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--lammps-basedir', type=str,
                        default="/n/brenner_lab/Lab/software/lammps-stable_29Sep2021",
                        help='LAMMPS base directory')
    parser.add_argument('--tacoxdna-basedir', type=str,
                        default="/n/brenner_lab/User/rkrueger/tacoxDNA",
                        help='tacoxDNA base directory')
    parser.add_argument('--seq-dep', action='store_true')
    parser.add_argument('--no-archive', action='store_true')
    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--ignore-warnings', action='store_true')
    parser.add_argument('--save-obj-every', type=int, default=10,
                        help="Frequency of saving numpy files")
    parser.add_argument('--plot-every', type=int, default=1,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument(
        '--opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["fene", "stacking"],
        help='Parameter keys to optimize'
    )

    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for reading trajectories")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")




    # LAMMPS/moduli information
    parser.add_argument('--n-sample-steps', type=int, default=3000000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=100000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=500,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims', type=int, default=2,
                        help="Number of simulations per force")



    parser.add_argument('--s-eff-coeff', type=float, default=1.0,
                        help="Coefficient for S_eff component")
    parser.add_argument('--c-coeff', type=float, default=0.0,
                        help="Coefficient for C component")
    parser.add_argument('--g-coeff', type=float, default=0.0,
                        help="Coefficient for g component")

    ## Experimental values and uncertainties -- Table 2 in https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00138

    parser.add_argument('--target-s-eff', type=float, default=1045,
                        help="Target S_eff in pN")
    parser.add_argument('--s-eff-uncertainty', type=float, default=92,
                        help="Experimental uncertainty for S_eff in pN")

    parser.add_argument('--target-c', type=float, default=436,
                        help="Target C in pn*nm^2")
    parser.add_argument('--c-uncertainty', type=float, default=16,
                        help="Experimental uncertainty for C in pn*nm^2")

    parser.add_argument('--target-g', type=float, default=-90.0,
                        help="Target g in (pn*nm)")
    parser.add_argument('--g-uncertainty', type=float, default=10,
                        help="Experimental uncertainty for g in (pn*nm)")



    parser.add_argument('--timestep', type=float, default=0.01,
                        help="Timestep for nve/dotc/langevin integrator")


    parser.add_argument(
        '--forces-pn',
        type=float,
        nargs='+',
        default=[0.0, 2.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help="List of forces in pn"
    )

    parser.add_argument(
        '--torques-pnnm',
        type=float,
        nargs='+',
        default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        help="List of torques in pnnm"
    )


    # Structural information

    parser.add_argument('--n-steps-per-sim-struc', type=int, default=100000,
                        help="Number of steps for sampling reference states per simulation for structural info")
    parser.add_argument('--n-eq-steps-struc', type=int, default=0,
                        help="Number of equilibration steps for structural info")
    parser.add_argument('--sample-every-struc', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims-struc', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--offset-struc', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
