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
import zipfile

from jax import jit, vmap, lax, value_and_grad
import jax.numpy as jnp
from jax_md import space, rigid_body
import optax

import orbax.checkpoint
from flax.training import orbax_utils

from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna2 import model, lammps_utils
import jax_dna.input.trajectory as jdt
from jax_dna.dna1.oxdna_utils import rewrite_input_file
from jax_dna.dna2.oxdna_utils import recompile_oxdna
from jax_dna.dna1 import model as model1
from jax_dna.loss import pitch, pitch2, tm, persistence_length, rise


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


compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))
compute_all_rises = vmap(rise.get_avg_rises, (0, None, None, None))

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


        unbound_op_idxs_extended = onp.array([(1+n_stem_bp)*d_idx for d_idx in range(n_dist_thresholds)])
        # bound_op_idxs_extended = onp.array(list(range(1, 1+n_stem_bp)))
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


def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))


def abs_relative_diff(target, current):
    rel_diff = (current - target) / target
    return jnp.sqrt(rel_diff**2)


def abs_relative_diff_uncertainty(val, lo_val, hi_val):
    abs_rel_diff = jnp.where(val < lo_val, abs_relative_diff(lo_val, val),
                         jnp.where(val > hi_val, abs_relative_diff(hi_val, val),
                                   0.0))
    return abs_rel_diff


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
    min_neff_factor_hpin = args['min_neff_factor_hpin']
    max_approx_iters = args['max_approx_iters']
    seq_avg = not args['seq_dep']
    assert(seq_avg)
    standardize = not args['no_standardize']
    orbax_ckpt_path = args['orbax_ckpt_path']
    ckpt_freq = args['ckpt_freq']


    if standardize:
        uncertainty_loss_fn = lambda val, lo_val, hi_val: abs_relative_diff_uncertainty(val, lo_val, hi_val)
    else:
        uncertainty_loss_fn = lambda val, lo_val, hi_val: rmse_uncertainty(val, lo_val, hi_val)

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
    sample_every_st = args['sample_every_st']
    n_sims_st = args['n_sims_st']

    n_eq_steps_st = args['n_eq_steps_st']
    assert(n_eq_steps_st % sample_every_st == 0)
    n_eq_states = n_eq_steps_st // sample_every_st

    n_sample_steps_st = args['n_sample_steps_st']
    assert(n_sample_steps_st % sample_every_st == 0)
    n_sample_states_st = n_sample_steps_st // sample_every_st

    n_total_steps_st = n_eq_steps_st + n_sample_steps_st
    n_total_states_st = n_total_steps_st // sample_every_st
    assert(n_total_states_st == n_sample_states_st + n_eq_states)

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
    n_forces = forces_pn.shape[0]
    torques_pnnm = jnp.array(args['torques_pnnm'], dtype=jnp.float64)
    n_torques = torques_pnnm.shape[0]

    compute_st = not args['no_compute_st']


    # Structural (60 bp) arguments
    n_sims_60bp = args['n_sims_60bp']
    n_steps_per_sim_60bp = args['n_steps_per_sim_60bp']
    n_eq_steps_60bp = args['n_eq_steps_60bp']
    sample_every_60bp = args['sample_every_60bp']
    assert(n_steps_per_sim_60bp % sample_every_60bp == 0)
    n_ref_states_per_sim_60bp = n_steps_per_sim_60bp // sample_every_60bp
    n_ref_states_60bp = n_ref_states_per_sim_60bp * n_sims_60bp
    offset_60bp = args['offset_60bp']
    target_pitch = args['target_pitch']
    compute_60bp = not args['no_compute_60bp']
    pitch_coeff = args['pitch_coeff']
    min_n_eff_60bp = int(n_ref_states_60bp * min_neff_factor)
    pitch_uncertainty = args['pitch_uncertainty']
    pitch_lo = target_pitch - pitch_uncertainty
    pitch_hi = target_pitch + pitch_uncertainty

    ## Persistence length arguments (uses same simulation)
    target_lp = args['target_lp']
    truncation_lp = args['truncation_lp']
    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])
    lp_coeff = args['lp_coeff']
    lp_uncertainty = args['lp_uncertainty']
    lp_lo = target_lp - lp_uncertainty
    lp_hi = target_lp + lp_uncertainty


    # Hairpin Tm arguments
    n_sims_hpin = args['n_sims_hpin']
    n_steps_per_sim_hpin = args['n_steps_per_sim_hpin']
    n_eq_steps_hpin = args['n_eq_steps_hpin']
    sample_every_hpin = args['sample_every_hpin']
    assert(n_steps_per_sim_hpin % sample_every_hpin == 0)
    n_ref_states_per_sim_hpin = n_steps_per_sim_hpin // sample_every_hpin
    n_ref_states_hpin = n_ref_states_per_sim_hpin * n_sims_hpin
    min_n_eff_hpin = int(n_ref_states_hpin * min_neff_factor_hpin)

    t_kelvin_hpin = args['temp_hpin']
    extrapolate_temps_hpin = jnp.array([float(et) for et in args['extrapolate_temps_hpin']]) # in Kelvin
    assert(jnp.all(extrapolate_temps_hpin[:-1] <= extrapolate_temps_hpin[1:])) # check that temps. are sorted
    n_extrap_temps_hpin = len(extrapolate_temps_hpin)
    extrapolate_kts_hpin = vmap(utils.get_kt)(extrapolate_temps_hpin)
    extrapolate_temp_str_hpin = ', '.join([f"{tc}K" for tc in extrapolate_temps_hpin])

    salt_concentration_hpin = args['salt_concentration_hpin']

    target_tm_hpin = args['target_tm_hpin']
    tm_hpin_uncertainty = args['tm_hpin_uncertainty']
    tm_hpin_lo = target_tm_hpin - tm_hpin_uncertainty
    tm_hpin_hi = target_tm_hpin + tm_hpin_uncertainty

    stem_bp_hpin = args['stem_bp_hpin']
    loop_nt_hpin = args['loop_nt_hpin']

    compute_hpin = not args['no_compute_hpin']
    hpin_coeff = args['hpin_coeff']



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
    neffs_st_path = log_dir / "neffs_st.txt"
    neff_60bp_path = log_dir / "neff_60bp.txt"
    neff_hpin_path = log_dir / "neff_hpin.txt"
    a1_path = log_dir / "a1.txt"
    a3_path = log_dir / "a3.txt"
    a4_path = log_dir / "a4.txt"
    s_eff_path = log_dir / "s_eff.txt"
    c_path = log_dir / "c.txt"
    g_path = log_dir / "g.txt"
    pitch_path = log_dir / "pitch.txt"
    rise_path = log_dir / "rise.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"
    warnings_path = log_dir / "warnings.txt"
    tm_path = log_dir / "tm.txt"
    width_path = log_dir / "width.txt"
    lp_path = log_dir / "lp.txt"
    l0_avg_path = log_dir / "l0_avg.txt"

    s_eff_loss_path = log_dir / "s_eff_loss.txt"
    c_loss_path = log_dir / "c_loss.txt"
    g_loss_path = log_dir / "g_loss.txt"
    pitch_loss_path = log_dir / "pitch_loss.txt"
    hpin_loss_path = log_dir / "hpin_loss.txt"
    lp_loss_path = log_dir / "lp_loss.txt"

    params_str = ""
    params_str += f"n_sample_states_st: {n_sample_states_st}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Setup systems

    displacement_fn_free, shift_fn_free = space.free() # FIXME: could use box size from top_info, but not sure how the centering works.

    ## Setup structural system (60 bp)

    sys_basedir_60bp = Path("data/templates/simple-helix-60bp")
    input_template_path_60bp = sys_basedir_60bp / "input"

    top_path_60bp = sys_basedir_60bp / "sys.top"
    top_info_60bp = topology.TopologyInfo(top_path_60bp, reverse_direction=False)
    seq_oh_60bp = jnp.array(utils.get_one_hot(top_info_60bp.seq), dtype=jnp.float64)
    strand_length_60bp = int(seq_oh_60bp.shape[0] // 2)

    quartets_60bp = utils.get_all_quartets(n_nucs_per_strand=seq_oh_60bp.shape[0] // 2)
    quartets_60bp = quartets_60bp[offset_60bp:-offset_60bp-1]

    conf_path = sys_basedir_60bp / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info_60bp,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info_60bp, conf_info)

    dt_60bp = 5e-3
    t_kelvin_60bp = utils.DEFAULT_TEMP
    kT_60bp = utils.get_kt(t_kelvin_60bp)
    beta_60bp = 1 / kT_60bp




    def get_60bp_tasks(iter_dir, params, prev_basedir, recompile=True):
        if recompile:
            recompile_start = time.time()
            recompile_oxdna(params, oxdna_path, t_kelvin_60bp, num_threads=n_threads)
            recompile_end = time.time()

        bp60_dir = iter_dir / "ds60"
        bp60_dir.mkdir(parents=False, exist_ok=False)

        all_repeat_dirs = list()
        for r in range(n_sims_60bp):
            repeat_dir = bp60_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            all_repeat_dirs.append(repeat_dir)

            shutil.copy(top_path_60bp, repeat_dir / "sys.top")

            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info)
            else:
                prev_repeat_dir = prev_basedir / "ds60" / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info_60bp,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info_60bp, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh_60bp.shape[0], r*n_steps_per_sim_60bp)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            rewrite_input_file(
                input_template_path_60bp, repeat_dir,
                temp=f"{t_kelvin_60bp}K", steps=n_steps_per_sim_60bp,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_60bp, seed=random.randrange(100),
                equilibration_steps=n_eq_steps_60bp, dt=dt_60bp,
                no_stdout_energy=0, backend="CPU",
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="DNA2_nomesh"
            )
        bp60_tasks = [run_oxdna_ray.remote(oxdna_exec_path, rdir) for rdir in all_repeat_dirs]
        return bp60_tasks, all_repeat_dirs

    def process_60bp(iter_dir, params):
        bp60_dir = iter_dir / "ds60"

        pitch_dir = bp60_dir / "pitch"
        pitch_dir.mkdir(parents=False, exist_ok=False)
        lp_dir = bp60_dir / "lp"
        lp_dir.mkdir(parents=False, exist_ok=False)
        rise_dir = bp60_dir / "rise"
        rise_dir.mkdir(parents=False, exist_ok=False)

        combine_cmd = "cat "
        for r in range(n_sims_60bp):
            repeat_dir = bp60_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {bp60_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        if not no_delete:
            files_to_remove = ["output.dat"]
            for r in range(n_sims_60bp):
                repeat_dir = bp60_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()


        # Analyze

        ## Load states from oxDNA simulation
        load_start = time.time()
        """
        traj_info = trajectory.TrajectoryInfo(
            top_info_60bp, read_from_file=True,
            traj_path=bp60_dir / "output.dat",
            # reverse_direction=True)
            reverse_direction=False)
        traj_states = traj_info.get_states()
        """
        traj_ = jdt.from_file(
            bp60_dir / "output.dat",
            [strand_length_60bp, strand_length_60bp],
            is_oxdna=False,
            n_processes=n_threads,
        )
        traj_states = [ns.to_rigid_body() for ns in traj_.states]
        n_traj_states = len(traj_states)
        traj_states = utils.tree_stack(traj_states)
        load_end = time.time()

        ## Load the oxDNA energies

        energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
        energy_dfs = [pd.read_csv(bp60_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims_60bp)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(displacement_fn_free, params, t_kelvin=t_kelvin_60bp)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_60bp,
            bonded_nbrs=top_info_60bp.bonded_nbrs,
            unbonded_nbrs=top_info_60bp.unbonded_nbrs.T,
            is_end=top_info_60bp.is_end
        )
        energy_fn = jit(energy_fn)

        # Check energies

        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()

        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh_60bp.shape[0]

        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        # Plot the energies
        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(bp60_dir / f"energies.png")
        plt.clf()

        # Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(bp60_dir / f"energy_diffs.png")
        plt.clf()

        # Compute the pitches
        analyze_start = time.time()

        n_quartets = quartets_60bp.shape[0]
        ref_avg_angles = list()
        for rs_idx in range(n_traj_states):
            body = traj_states[rs_idx]
            angles = pitch2.get_all_angles(body, quartets_60bp, displacement_fn_free, model.com_to_hb, model1.com_to_backbone, 0.0)
            state_avg_angle = onp.mean(angles)
            ref_avg_angles.append(state_avg_angle)
        ref_avg_angles = onp.array(ref_avg_angles)

        running_avg_angles = onp.cumsum(ref_avg_angles) / onp.arange(1, n_traj_states + 1)
        running_avg_pitches = 2*onp.pi / running_avg_angles
        plt.plot(running_avg_pitches)
        plt.savefig(pitch_dir / f"running_avg.png")
        plt.clf()

        plt.plot(running_avg_pitches[-int(n_traj_states // 2):])
        plt.savefig(pitch_dir / f"running_avg_second_half.png")
        plt.clf()

        # Compute the rise
        all_state_rises = compute_all_rises(traj_states, quartets_60bp, displacement_fn_free, model.com_to_hb)
        avg_rise = jnp.mean(all_state_rises)

        running_avg_rises = onp.cumsum(all_state_rises) / onp.arange(1, n_traj_states + 1)
        plt.plot(running_avg_rises)
        plt.savefig(rise_dir / f"running_avg.png")
        plt.clf()

        plt.plot(running_avg_rises[-int(n_traj_states // 2):])
        plt.savefig(rise_dir / f"running_avg_second_half.png")
        plt.clf()


        # Compute the persistence lengths
        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(traj_states, quartets_60bp, base_site)
        mean_corr_curve = jnp.mean(unweighted_corr_curves, axis=0)
        mean_l0 = jnp.mean(unweighted_l0_avgs)
        mean_Lp_truncated, offset = persistence_length.persistence_length_fit(mean_corr_curve[:truncation_lp], mean_l0)


        compute_every = 10
        n_curves = unweighted_corr_curves.shape[0]
        all_inter_lps = list()
        all_inter_lps_truncated = list()
        for i in range(0, n_curves, compute_every):
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation_lp], mean_l0)
            all_inter_lps_truncated.append(inter_mean_Lp_truncated * utils.nm_per_oxdna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, mean_l0)
            all_inter_lps.append(inter_mean_Lp * utils.nm_per_oxdna_length)

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(lp_dir / "running_avg.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(lp_dir / "running_avg_truncated.png")
        plt.clf()

        plt.plot(mean_corr_curve)
        plt.axvline(x=truncation_lp, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(lp_dir / "full_corr_curve.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation_lp, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "full_log_corr_curve.png")
        plt.clf()

        # fit_fn = lambda n: -n * mean_l0 / (mean_Lp_truncated/utils.nm_per_oxdna_length) + offset
        fit_fn = lambda n: -n * (mean_l0 / mean_Lp_truncated) + offset
        plt.plot(jnp.log(mean_corr_curve)[:truncation_lp])
        # neg_inverse_slope = (mean_Lp_truncated / utils.nm_per_oxdna_length) / mean_l0 # in nucleotides
        neg_inverse_slope = mean_Lp_truncated / mean_l0 # in nucleotides
        rounded_offset = onp.round(offset, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation_lp)), linestyle='--', label=fit_str)

        plt.title(f"Log-Correlation Curve, Truncated.")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "log_corr_curve.png")
        plt.clf()





        # Record the loss
        with open(bp60_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Max energy diff: {onp.max(energy_diffs)}\n")
            f.write(f"Min energy diff: {onp.min(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")
            f.write(f"\nPitch: {2*onp.pi / onp.mean(ref_avg_angles)} bp\n")
            f.write(f"\nMean Rise (nm): {avg_rise * utils.nm_per_oxdna_length}\n")

            f.write(f"\nMean Lp truncated (oxDNA units): {mean_Lp_truncated}\n")
            f.write(f"Mean L0 (oxDNA units): {mean_l0}\n")
            f.write(f"Mean Rise (oxDNA units): {avg_rise}\n")
            f.write(f"Mean Lp truncated (num bp via oxDNA units): {mean_Lp_truncated / avg_rise}\n")

            f.write(f"\nMean Lp truncated (nm): {mean_Lp_truncated * utils.nm_per_oxdna_length}\n")
            f.write(f"Mean L0 (nm): {mean_l0 * utils.nm_per_oxdna_length}\n")

        if not no_archive:
            zip_file(str(bp60_dir / "output.dat"), str(bp60_dir / "output.dat.zip"))
            os.remove(str(bp60_dir / "output.dat"))

        ref_info = (traj_states, calc_energies, jnp.array(ref_avg_angles), unweighted_corr_curves, unweighted_l0_avgs, all_state_rises)

        return ref_info


    ## Setup hairpin Tm system
    hairpin_basedir = Path("data/templates/hairpins")
    sys_basedir_hpin = hairpin_basedir / f"{stem_bp_hpin}bp_stem_{loop_nt_hpin}nt_loop"
    assert(sys_basedir_hpin.exists())
    conf_path_unbound_hpin = sys_basedir_hpin / "init_unbound.conf"
    conf_path_bound_hpin = sys_basedir_hpin / "init_bound.conf"
    top_path_hpin = sys_basedir_hpin / "sys.top"
    input_template_path_hpin = sys_basedir_hpin / "input"
    op_path_hpin = sys_basedir_hpin / "op.txt"
    wfile_path_hpin = sys_basedir_hpin / "wfile.txt"

    top_info_hpin = topology.TopologyInfo(
        top_path_hpin,
        reverse_direction=False
        # reverse_direction=True
    )
    seq_oh_hpin = jnp.array(utils.get_one_hot(top_info_hpin.seq), dtype=jnp.float64)
    n_nt_hpin = seq_oh_hpin.shape[0]
    assert(n_nt_hpin == 2*stem_bp_hpin + loop_nt_hpin)


    conf_info_unbound_hpin = trajectory.TrajectoryInfo(
        top_info_hpin, read_from_file=True, traj_path=conf_path_unbound_hpin,
        # reverse_direction=True
        reverse_direction=False
    )
    conf_info_bound_hpin = trajectory.TrajectoryInfo(
        top_info_hpin, read_from_file=True, traj_path=conf_path_bound_hpin,
        # reverse_direction=True
        reverse_direction=False
    )
    box_size_hpin = conf_info_bound_hpin.box_size

    kT_hpin = utils.get_kt(t_kelvin_hpin)
    beta_hpin = 1 / kT_hpin

    ### Process the weights information
    weights_df_hpin = pd.read_fwf(wfile_path_hpin, names=["op1", "op2", "weight"])
    num_ops_hpin = len(weights_df_hpin)
    n_stem_bp_hpin = len(weights_df_hpin.op1.unique())
    n_dist_thresholds_hpin = len(weights_df_hpin.op2.unique())
    pair2idx_hpin = dict()
    idx2pair_hpin = dict()
    idx2weight_hpin = dict()
    unbound_op_idxs_hpin = list()
    bound_op_idxs_hpin = list()
    for row_idx, row in weights_df_hpin.iterrows():
        op1 = int(row.op1)
        op2 = int(row.op2)
        pair2idx_hpin[(op1, op2)] = row_idx
        idx2pair_hpin[row_idx] = (op1, op2)
        idx2weight_hpin[row_idx] = row.weight

        if op1 == 0:
            unbound_op_idxs_hpin.append(row_idx)
        else:
            bound_op_idxs_hpin.append(row_idx)
    bound_op_idxs_hpin = onp.array(bound_op_idxs_hpin)
    unbound_op_idxs_hpin = onp.array(unbound_op_idxs_hpin)

    def compute_ratio_hpin(ub_counts):
        ub_unbiased_counts = ub_counts[unbound_op_idxs_hpin]
        ub_biased_counts = ub_counts[bound_op_idxs_hpin]

        return ub_biased_counts.sum() / ub_unbiased_counts.sum()

    max_seed_tries_hpin = 5
    seed_check_sample_freq_hpin = 10
    seed_check_steps_hpin = 100


    def get_hpin_tasks(iter_dir, params, recompile=False):
        if recompile:
            recompile_start = time.time()
            recompile_oxdna(params, oxdna_path, t_kelvin_hpin, num_threads=n_threads)
            recompile_end = time.time()

        hpin_dir = iter_dir / "hpin"
        hpin_dir.mkdir(parents=False, exist_ok=False)

        all_repeat_dirs = list()
        for r in range(n_sims_hpin):
            repeat_dir = hpin_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            all_repeat_dirs.append(repeat_dir)

            shutil.copy(top_path_hpin, repeat_dir / "sys.top")
            shutil.copy(wfile_path_hpin, repeat_dir / "wfile.txt")
            shutil.copy(op_path_hpin, repeat_dir / "op.txt")

            if r % 2 == 0:
                conf_info_copy = deepcopy(conf_info_bound_hpin)
            else:
                conf_info_copy = deepcopy(conf_info_unbound_hpin)

            conf_info_copy.traj_df.t = onp.full(seq_oh_hpin.shape[0], r*n_steps_per_sim_hpin)

            conf_info_copy.write(repeat_dir / "init.conf",
                                 reverse=False,
                                 # reverse=True,
                                 write_topology=False)


            check_seed_dir = repeat_dir / "check_seed"
            check_seed_dir.mkdir(parents=False, exist_ok=False)
            s_idx = 0
            valid_seed = None
            while s_idx < max_seed_tries_hpin and valid_seed is None:
                seed_try = random.randrange(10000)
                seed_dir = check_seed_dir / f"s{seed_try}"
                seed_dir.mkdir(parents=False, exist_ok=False)

                rewrite_input_file(
                    input_template_path_hpin, seed_dir,
                    temp=f"{t_kelvin_hpin}K", steps=seed_check_steps_hpin,
                    init_conf_path=str(repeat_dir / "init.conf"),
                    top_path=str(repeat_dir / "sys.top"),
                    save_interval=seed_check_sample_freq_hpin, seed=seed_try,
                    equilibration_steps=0,
                    no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str_hpin,
                    weights_file=str(repeat_dir / "wfile.txt"),
                    op_file=str(repeat_dir / "op.txt"),
                    log_file=str(repeat_dir / "sim.log"),
                    interaction_type="DNA2_nomesh",
                    salt_concentration=salt_concentration_hpin
                )

                seed_proc = subprocess.Popen([oxdna_exec_path, seed_dir / "input"])
                seed_proc.wait()
                seed_rc = seed_proc.returncode

                if seed_rc == 0:
                    valid_seed = seed_try

                s_idx += 1

            if valid_seed is None:
                raise RuntimeError(f"Could not find valid seed.")


            rewrite_input_file(
                input_template_path_hpin, repeat_dir,
                temp=f"{t_kelvin_hpin}K", steps=n_steps_per_sim_hpin,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_hpin, seed=valid_seed,
                equilibration_steps=n_eq_steps_hpin,
                no_stdout_energy=0, extrapolate_hist=extrapolate_temp_str_hpin,
                weights_file=str(repeat_dir / "wfile.txt"),
                op_file=str(repeat_dir / "op.txt"),
                log_file=str(repeat_dir / "sim.log"),
                interaction_type="DNA2_nomesh",
                salt_concentration=salt_concentration_hpin
            )

        hpin_tasks = [run_oxdna_ray.remote(oxdna_exec_path, rdir) for rdir in all_repeat_dirs]
        return hpin_tasks, all_repeat_dirs

    def process_hpin(iter_dir, params):
        hpin_dir = iter_dir / "hpin"

        ## Combine the output files
        combine_cmd = "cat "
        for r in range(n_sims_hpin):
            repeat_dir = hpin_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {hpin_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        if not no_delete:
            files_to_remove = ["output.dat"]
            for r in range(n_sims_hpin):
                repeat_dir = hpin_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()

        ## Compute running avg. from `traj_hist.dat`
        all_traj_hist_fpaths = list()
        for r in range(n_sims_hpin):
            repeat_dir = hpin_dir / f"r{r}"
            all_traj_hist_fpaths.append(repeat_dir / "traj_hist.dat")

        ### Note: the below should really go in its own loss file
        all_running_tms, all_running_widths = hairpin_tm_running_avg(
            all_traj_hist_fpaths, n_stem_bp_hpin, n_dist_thresholds_hpin)

        plt.plot(all_running_tms)
        plt.xlabel("Iteration")
        plt.ylabel("Tm (C)")
        plt.savefig(hpin_dir / "traj_hist_running_tm.png")
        # plt.show()
        plt.clf()

        plt.plot(all_running_widths)
        plt.xlabel("Iteration")
        plt.ylabel("Width (C)")
        plt.savefig(hpin_dir / "traj_hist_running_width.png")
        # plt.show()
        plt.clf()

        ## Load states from oxDNA simulation
        traj_info = trajectory.TrajectoryInfo(
            top_info_hpin, read_from_file=True,
            traj_path=hpin_dir / "output.dat",
            reverse_direction=False
            # reverse_direction=True
        )
        ref_states = traj_info.get_states()
        assert(len(ref_states) == n_ref_states_hpin)
        ref_states = utils.tree_stack(ref_states)


        ## Load the oxDNA energies
        energy_df_columns = [
            "time", "potential_energy", "acc_ratio_trans", "acc_ratio_rot",
            "acc_ratio_vol", "op1", "op2", "op_weight"
        ]
        energy_dfs = [pd.read_csv(hpin_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims_hpin)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)


        ## Load all last_hist.dat and combine
        last_hist_columns = ["num_bp", "dist_threshold_idx", "count_biased", "count_unbiased"] \
                            + [str(et) for et in extrapolate_temps_hpin]
        last_hist_df = None
        for r in range(n_sims_hpin):
            repeat_dir = hpin_dir / f"r{r}"
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
        em_base = model.EnergyModel(displacement_fn_free, params, t_kelvin=t_kelvin_hpin, salt_conc=salt_concentration_hpin)
        energy_fn = lambda body: em_base.energy_fn(
            body,
            seq=seq_oh_hpin,
            bonded_nbrs=top_info_hpin.bonded_nbrs,
            unbonded_nbrs=top_info_hpin.unbonded_nbrs.T,
            is_end=top_info_hpin.is_end
        )
        energy_fn = jit(energy_fn)

        ref_energies = list()
        for rs_idx in tqdm(range(n_ref_states_hpin), desc="Calculating energies"):
            rs = ref_states[rs_idx]
            ref_energies.append(energy_fn(rs))
        ref_energies = jnp.array(ref_energies)

        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh_hpin.shape[0]

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
        plt.savefig(hpin_dir / "energies.png")
        plt.clf()

        sns.histplot(energy_diffs)
        plt.savefig(hpin_dir / "energy_diffs.png")
        # plt.show()
        plt.clf()


        ## Check uniformity across biased counts
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ### First, the periodic counts derived from the energy file(s)
        count_df = energy_df.groupby(['op1', 'op2', 'op_weight']).size().reset_index().rename(columns={0: "count"})
        op_names = list()
        op_weights = list()
        op_counts_periodic = list()
        for row_idx, row in weights_df_hpin.iterrows():
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
        for row_idx, row in weights_df_hpin.iterrows():
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

        plt.savefig(hpin_dir / "biased_counts.png")
        plt.clf()


        ## Unbias reference counts
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        op_counts_periodic_unbiased = op_counts_periodic / op_weights
        sns.barplot(x=op_names, y=op_counts_periodic_unbiased, ax=ax[0])
        ax[0].set_title(f"Periodic Counts, Reference T={t_kelvin_hpin}K")
        ax[0].set_xlabel("O.P.")

        op_counts_frequent_unbiased = op_counts_frequent / op_weights

        sns.barplot(x=op_names, y=op_counts_frequent_unbiased, ax=ax[1])
        ax[1].set_title(f"Frequent Counts, Reference T={t_kelvin_hpin}K")

        plt.savefig(hpin_dir / "unbiased_counts.png")
        plt.clf()


        ## Unbias counts for each temperature
        all_ops = list(zip(energy_df.op1.to_numpy(), energy_df.op2.to_numpy()))
        all_unbiased_counts = list()
        all_unbiased_counts_ref = list()
        compute_running_avg_every = 100
        all_running_avg_ratios = list()
        for extrap_t_kelvin, extrap_kt in zip(extrapolate_temps_hpin, extrapolate_kts_hpin):
            em_temp = model.EnergyModel(displacement_fn_free, params, t_kelvin=extrap_t_kelvin, salt_conc=salt_concentration_hpin)
            energy_fn_temp = lambda body: em_temp.energy_fn(
                body,
                seq=seq_oh_hpin,
                bonded_nbrs=top_info_hpin.bonded_nbrs,
                unbonded_nbrs=top_info_hpin.unbonded_nbrs.T,
                is_end=top_info_hpin.is_end
            )
            energy_fn_temp = jit(energy_fn_temp)

            temp_unbiased_counts = onp.zeros(num_ops_hpin)
            temp_running_avg_ratios = list()
            for rs_idx in tqdm(range(n_ref_states_hpin), desc=f"Extrapolating to {extrap_t_kelvin}K"):
                rs = ref_states[rs_idx]
                op1, op2 = all_ops[rs_idx]
                op_idx = pair2idx_hpin[(op1, op2)]
                op_weight = idx2weight_hpin[int(op_idx)]

                calc_energy = ref_energies[rs_idx]
                calc_energy_temp = energy_fn_temp(rs)

                boltz_diff = jnp.exp(calc_energy/kT_hpin - calc_energy_temp/extrap_kt)
                temp_unbiased_counts[op_idx] += 1/op_weight * boltz_diff

                if rs_idx and rs_idx % compute_running_avg_every == 0:
                    curr_unbound_count = temp_unbiased_counts[unbound_op_idxs_hpin].sum()
                    curr_bound_count = temp_unbiased_counts[bound_op_idxs_hpin].sum()
                    curr_ratio = curr_bound_count / curr_unbound_count
                    temp_running_avg_ratios.append(curr_ratio)

            all_unbiased_counts.append(temp_unbiased_counts)
            all_running_avg_ratios.append(temp_running_avg_ratios)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.barplot(x=op_names, y=temp_unbiased_counts, ax=ax[0])
            ax[0].set_title(f"Periodic Counts, T={extrap_t_kelvin}K")
            ax[0].set_xlabel("O.P.")

            # Get the frequent counts at the extrapolated temp. from the oxDNA histogram
            frequent_extrap_counts = list()
            for row_idx, row in weights_df_hpin.iterrows():
                op1 = int(row.op1)
                op2 = int(row.op2)

                last_hist_row = last_hist_df[(last_hist_df.num_bp == op1) \
                                             & (last_hist_df.dist_threshold_idx == op2)]
                frequent_extrap_counts.append(last_hist_row[str(extrap_t_kelvin)].to_numpy()[0])
            frequent_extrap_counts = onp.array(frequent_extrap_counts)
            all_unbiased_counts_ref.append(frequent_extrap_counts)

            sns.barplot(x=op_names, y=frequent_extrap_counts, ax=ax[1])
            ax[1].set_title(f"Frequent Counts, T={extrap_t_kelvin}K")

            plt.savefig(hpin_dir / f"unbiased_counts_{extrap_t_kelvin}K_extrap.png")
            plt.clf()

        all_unbiased_counts = onp.array(all_unbiased_counts)
        all_unbiased_counts_ref = onp.array(all_unbiased_counts_ref)

        all_running_avg_ratios = onp.array(all_running_avg_ratios) # n_temps x n_running_avg_points_hpin
        n_running_avg_points_hpin = all_running_avg_ratios.shape[1]
        running_tms = list()
        running_widths = list()
        for ra_idx in range(n_running_avg_points_hpin):
            curr_ratios = all_running_avg_ratios[:, ra_idx]

            curr_tm = tm.compute_tm(extrapolate_temps_hpin, curr_ratios)
            running_tms.append(curr_tm)

            curr_width = tm.compute_width(extrapolate_temps_hpin, curr_ratios)
            running_widths.append(curr_width)

        plt.plot(running_tms)
        plt.savefig(hpin_dir / "discrete_running_avg_tm.png")
        plt.clf()

        plt.plot(running_widths)
        plt.savefig(hpin_dir / "discrete_running_avg_width.png")
        plt.clf()

        # Compute the final Tms and widths

        unbound_unbiased_counts = all_unbiased_counts[:, unbound_op_idxs_hpin]
        bound_unbiased_counts = all_unbiased_counts[:, bound_op_idxs_hpin]

        ratios = list()
        for t_idx in range(len(extrapolate_temps_hpin)):
            unbound_count = unbound_unbiased_counts[t_idx].sum()
            bound_count = bound_unbiased_counts[t_idx].sum()

            ratio = bound_count / unbound_count
            ratios.append(ratio)
        ratios = onp.array(ratios)

        calc_tm = tm.compute_tm(extrapolate_temps_hpin, ratios)
        calc_width = tm.compute_width(extrapolate_temps_hpin, ratios)

        rev_ratios = jnp.flip(ratios)
        rev_temps = jnp.flip(extrapolate_temps_hpin)
        ratios_extrap = jnp.arange(0.1, 1., 0.05)
        temps_extrap = jnp.interp(ratios_extrap, rev_ratios, rev_temps)
        plt.plot(temps_extrap, ratios_extrap)
        plt.xlabel("T/K")
        plt.ylabel("Hairpin Yield")
        plt.title(f"Tm={onp.round(calc_tm, 2)}, width={onp.round(calc_width, 2)}")
        plt.savefig(hpin_dir / "melting_curve_calc.png")
        plt.clf()


        unbound_unbiased_counts_ref = all_unbiased_counts_ref[:, unbound_op_idxs_hpin]
        bound_unbiased_counts_ref = all_unbiased_counts_ref[:, bound_op_idxs_hpin]
        ratios_ref = list()
        for t_idx in range(len(extrapolate_temps_hpin)):
            unbound_count_ref = unbound_unbiased_counts_ref[t_idx].sum()
            bound_count_ref = bound_unbiased_counts_ref[t_idx].sum()

            ratio_ref = bound_count_ref / unbound_count_ref
            ratios_ref.append(ratio_ref)
        ratios_ref = onp.array(ratios_ref)

        calc_tm_ref = tm.compute_tm(extrapolate_temps_hpin, ratios_ref)
        calc_width_ref = tm.compute_width(extrapolate_temps_hpin, ratios_ref)

        rev_ratios = jnp.flip(ratios_ref)
        rev_temps = jnp.flip(extrapolate_temps_hpin)
        ratios_extrap = jnp.arange(0.1, 1., 0.05)
        temps_extrap = jnp.interp(ratios_extrap, rev_ratios, rev_temps)
        plt.plot(temps_extrap, ratios_extrap)
        plt.xlabel("T/K")
        plt.ylabel("Hairpin Yield")
        plt.title(f"Tm={onp.round(calc_tm_ref, 2)}, width={onp.round(calc_width_ref, 2)}")
        plt.savefig(hpin_dir / "melting_curve_ref.png")
        plt.clf()

        summary_str = ""
        summary_str += f"Reference Tm: {calc_tm_ref}\n"
        summary_str += f"Reference width: {calc_width_ref}\n"
        summary_str += f"Calc. Tm: {calc_tm}\n"
        summary_str += f"Calc. width: {calc_width}\n"
        with open(hpin_dir / "summary.txt", "w+") as f:
            f.write(summary_str)

        all_op_weights = list()
        all_op_idxs = list()
        for op1, op2 in all_ops:
            op_idx = pair2idx_hpin[(op1, op2)]
            op_weight = idx2weight_hpin[int(op_idx)]
            all_op_weights.append(op_weight)
            all_op_idxs.append(op_idx)

        all_ops = jnp.array(all_ops).astype(jnp.int32)
        all_op_weights = jnp.array(all_op_weights)
        all_op_idxs = jnp.array(all_op_idxs)

        if not no_archive:
            zip_file(str(hpin_dir / "output.dat"), str(hpin_dir / "output.dat.zip"))
            os.remove(str(hpin_dir / "output.dat"))

        plt.plot(all_op_idxs)
        for i in range(n_sims_hpin):
            plt.axvline(x=i*n_ref_states_per_sim_hpin, linestyle="--", color="red")
        plt.savefig(hpin_dir / "op_trajectory.png")
        plt.clf()

        ref_info = (ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs)
        return ref_info

    ## Setup LAMMPS stretch/torsionn system
    sys_basedir_st = Path("data/templates/lammps-stretch-tors")
    lammps_data_rel_path = sys_basedir_st / "data"
    lammps_data_abs_path = os.getcwd() / lammps_data_rel_path

    p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", lammps_data_abs_path], cwd=run_dir)
    p.wait()
    rc = p.returncode
    if rc != 0:
        raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

    init_conf_fpath = run_dir / "data.oxdna"
    assert(init_conf_fpath.exists())
    os.rename(init_conf_fpath, run_dir / "init.conf")

    top_fpath_st = run_dir / "data.top"
    assert(top_fpath_st.exists())
    os.rename(top_fpath_st, run_dir / "sys_st.top")
    top_fpath_st = run_dir / "sys_st.top"

    top_info_st = topology.TopologyInfo(top_fpath_st, reverse_direction=False)
    seq_oh_st = jnp.array(utils.get_one_hot(top_info_st.seq), dtype=jnp.float64)
    seq_st = top_info_st.seq
    n_st = seq_oh_st.shape[0]
    assert(n_st % 2 == 0)
    n_bp_st = n_st // 2
    strand_length_st = int(seq_oh_st.shape[0] // 2)

    strand1_start = 0
    strand1_end = n_bp_st-1
    strand2_start = n_bp_st
    strand2_end = n_bp_st*2-1

    ### The region for which theta and distance are measured
    quartets_st = utils.get_all_quartets(n_nucs_per_strand=n_bp_st)
    quartets_st = quartets_st[4:n_bp_st-5]

    bp1_meas = [4, strand2_end-4]
    bp2_meas = [strand1_end-4, strand2_start+4]


    @jit
    def compute_distance(body):
        bp1_meas_pos = get_bp_pos(body, bp1_meas)
        bp2_meas_pos = get_bp_pos(body, bp2_meas)
        dist = jnp.abs(bp1_meas_pos[2] - bp2_meas_pos[2])
        return dist

    @jit
    def compute_theta(body):
        pitches = compute_pitches(body, quartets_st, displacement_fn_free, model.com_to_hb)
        return pitches.sum()

    t_kelvin_st = 300.0
    kT_st = utils.get_kt(t_kelvin_st)
    beta_st = 1 / kT_st
    salt_conc_st = 0.15
    q_eff_st = 0.815

    def get_stretch_tors_tasks(iter_dir, params, prev_states_force, prev_states_torque):
        all_sim_dirs = list()
        repeat_seeds = [random.randrange(1, 100) for _ in range(n_sims_st)]
        for f_idx, force_pn in enumerate(forces_pn):
            sim_dir = iter_dir / f"sim-f{force_pn}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims_st):
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
                    lammps_utils.stretch_tors_data_constructor(prev_states_force[f_idx][r], seq_st, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT_st, salt_conc=salt_conc_st, qeff=q_eff_st,
                    force_pn=force_pn, torque_pnnm=0,
                    save_every=sample_every_st, n_steps=n_total_steps_st,
                    seq_avg=seq_avg, seed=repeat_seed, timestep=timestep)

        for t_idx, torque_pnnm in enumerate(torques_pnnm):
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            sim_dir.mkdir(parents=False, exist_ok=False)

            for r in range(n_sims_st):
                repeat_dir = sim_dir / f"r{r}"
                repeat_dir.mkdir(parents=False, exist_ok=False)

                all_sim_dirs.append(repeat_dir)

                repeat_seed = repeat_seeds[r]

                if prev_states_torque is None:
                    shutil.copy(lammps_data_abs_path, repeat_dir / "data")
                else:
                    lammps_utils.stretch_tors_data_constructor(prev_states_torque[t_idx][r], seq_st, repeat_dir / "data")
                lammps_in_fpath = repeat_dir / "in"
                lammps_utils.stretch_tors_constructor(
                    params, lammps_in_fpath, kT=kT_st, salt_conc=salt_conc_st, qeff=q_eff_st,
                    force_pn=2.0, torque_pnnm=torque_pnnm,
                    save_every=sample_every_st, n_steps=n_total_steps_st,
                    seq_avg=seq_avg, seed=repeat_seed)

        stretch_tors_tasks = [run_lammps_ray.remote(lammps_exec_path, rdir) for rdir in all_sim_dirs]
        return stretch_tors_tasks, all_sim_dirs


    def process_stretch_tors(iter_dir, params):
        # Convert via TacoxDNA
        for force_pn in forces_pn:
            sim_dir = iter_dir / f"sim-f{force_pn}"
            for r in range(n_sims_st):
                repeat_dir = sim_dir / f"r{r}"
                p = subprocess.Popen([tacoxdna_basedir / "src/LAMMPS_oxDNA.py", "data", "filename.dat"], cwd=repeat_dir)
                p.wait()
                rc = p.returncode
                if rc != 0:
                    raise RuntimeError(f"tacoxDNA conversion failed with error code: {rc}")

        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            for r in range(n_sims_st):
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
            for r in range(n_sims_st):
                repeat_dir = sim_dir / f"r{r}"
                combine_cmd += f"{repeat_dir}/data.oxdna "
            combine_cmd += f"> {sim_dir}/output.dat"
            combine_proc = subprocess.run(combine_cmd, shell=True)
            if combine_proc.returncode != 0:
                raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")


        for torque_pnnm in torques_pnnm:
            sim_dir = iter_dir / f"sim-t{torque_pnnm}"
            combine_cmd = "cat "
            for r in range(n_sims_st):
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
                for r in range(n_sims_st):
                    repeat_dir = sim_dir / f"r{r}"
                    for f_stem in files_to_remove:
                        file_to_rem = repeat_dir / f_stem
                        file_to_rem.unlink()

            for torque_pnnm in torques_pnnm:
                sim_dir = iter_dir / f"sim-t{torque_pnnm}"
                for r in range(n_sims_st):
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
                [strand_length_st, strand_length_st],
                is_oxdna=False,
                n_processes=n_threads,
            )
            full_traj_states = [ns.to_rigid_body() for ns in traj_.states]

            assert(len(full_traj_states) == (1+n_total_states_st)*n_sims_st)
            sim_freq = 1+n_total_states_st
            traj_states = list()
            force_last_states = list()
            for r in range(n_sims_st):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states_st)
                traj_states += sampled_sim_states
                force_last_states.append(sampled_sim_states[-1])
            assert(len(traj_states) == n_sample_states_st*n_sims_st)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims_st):
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
                assert(rpt_log_df.shape[0] == n_total_states_st+1)
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
                    t_kelvin=t_kelvin_st
                )
            em = model.EnergyModel(displacement_fn_free,
                                   params,
                                   t_kelvin=t_kelvin_st,
                                   ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                                   salt_conc=salt_conc_st, q_eff=q_eff_st, seq_avg=seq_avg,
                                   ignore_exc_vol_bonded=True # Because we're in LAMMPS
            )
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh_st,
                bonded_nbrs=top_info_st.bonded_nbrs,
                unbonded_nbrs=top_info_st.unbonded_nbrs.T)
            energy_fn = jit(energy_fn)

            ## Compute the energies via our energy function
            energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
            _, calc_energies = scan(energy_scan_fn, None, traj_states)

            ## Check energies
            gt_energies = (log_df.PotEng * seq_oh_st.shape[0]).to_numpy()
            energy_diffs = list()
            for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                diff = onp.abs(calc - gt)
                energy_diffs.append(diff)


            ## Compute the mean distance
            traj_distances = list()
            for rs_idx in range(n_sample_states_st*n_sims_st):
                ref_state = traj_states[rs_idx]
                dist = compute_distance(ref_state)
                traj_distances.append(dist)

            traj_distances = onp.array(traj_distances)

            ## Compute the mean theta
            traj_thetas = list()
            for rs_idx in range(n_sample_states_st*n_sims_st):
                ref_state = traj_states[rs_idx]
                theta = compute_theta(ref_state)
                traj_thetas.append(theta)

            traj_thetas = onp.array(traj_thetas)

            ## Record some plots
            plt.plot(traj_distances)
            plt.savefig(sim_dir / "dist_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_distances) / onp.arange(1, (n_sample_states_st*n_sims_st)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_dist.png")
            plt.clf()
            running_avgs_force_dists.append(running_avg)

            last_half = int((n_sample_states_st * n_sims_st) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_dist_second_half.png")
            plt.clf()


            plt.plot(traj_thetas)
            plt.savefig(sim_dir / "theta_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_thetas) / onp.arange(1, (n_sample_states_st*n_sims_st)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_theta.png")
            plt.clf()

            last_half = int((n_sample_states_st * n_sims_st) // 2)
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
        running_avgs_force_dists = onp.array(running_avgs_force_dists) # (n_forces, n_sample_states_st*n_sims)
        running_avg_idxs = onp.arange(n_sample_states_st*n_sims_st)
        n_running_avg_points = 100
        check_every = (n_sample_states_st*n_sims_st) // n_running_avg_points
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
                [strand_length_st, strand_length_st],
                is_oxdna=False,
                n_processes=n_threads,
            )
            full_traj_states = [ns.to_rigid_body() for ns in traj_.states]


            assert(len(full_traj_states) == (1+n_total_states_st)*n_sims_st)
            sim_freq = 1+n_total_states_st
            traj_states = list()
            torque_last_states = list()
            for r in range(n_sims_st):
                sim_states = full_traj_states[r*sim_freq:(r+1)*sim_freq]
                sampled_sim_states = sim_states[1+n_eq_states:]
                assert(len(sampled_sim_states) == n_sample_states_st)
                traj_states += sampled_sim_states
                torque_last_states.append(sampled_sim_states[-1])
            assert(len(traj_states) == n_sample_states_st*n_sims_st)
            traj_states = utils.tree_stack(traj_states)

            ## Load the LAMMPS energies
            log_dfs = list()
            for r in range(n_sims_st):
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
                assert(rpt_log_df.shape[0] == n_total_states_st+1)
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
                    t_kelvin=t_kelvin_st
                )
            em = model.EnergyModel(displacement_fn_free,
                                   params,
                                   t_kelvin=t_kelvin_st,
                                   ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights,
                                   salt_conc=salt_conc_st, q_eff=q_eff_st, seq_avg=seq_avg,
                                   ignore_exc_vol_bonded=True # Because we're in LAMMPS
            )
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh_st,
                bonded_nbrs=top_info_st.bonded_nbrs,
                unbonded_nbrs=top_info_st.unbonded_nbrs.T)
            energy_fn = jit(energy_fn)

            ## Compute the energies via our energy function
            energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
            _, calc_energies = scan(energy_scan_fn, None, traj_states)

            ## Check energies
            gt_energies = (log_df.PotEng * seq_oh_st.shape[0]).to_numpy()
            energy_diffs = list()
            for idx, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
                diff = onp.abs(calc - gt)
                energy_diffs.append(diff)


            ## Compute the mean distance
            traj_distances = list()
            for rs_idx in range(n_sample_states_st*n_sims_st):
                ref_state = traj_states[rs_idx]
                dist = compute_distance(ref_state)
                traj_distances.append(dist)

            traj_distances = onp.array(traj_distances)

            ## Compute the mean theta
            traj_thetas = list()
            for rs_idx in range(n_sample_states_st*n_sims_st):
                ref_state = traj_states[rs_idx]
                theta = compute_theta(ref_state)
                traj_thetas.append(theta)

            traj_thetas = onp.array(traj_thetas)

            ## Record some plots
            plt.plot(traj_distances)
            plt.savefig(sim_dir / "dist_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_distances) / onp.arange(1, (n_sample_states_st*n_sims_st)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_dist.png")
            plt.clf()

            last_half = int((n_sample_states_st * n_sims_st) // 2)
            plt.plot(running_avg[-last_half:])
            plt.savefig(sim_dir / "running_avg_dist_second_half.png")
            plt.clf()


            plt.plot(traj_thetas)
            plt.savefig(sim_dir / "theta_traj.png")
            plt.clf()

            running_avg = onp.cumsum(traj_thetas) / onp.arange(1, (n_sample_states_st*n_sims_st)+1)
            plt.plot(running_avg)
            plt.savefig(sim_dir / "running_avg_theta.png")
            plt.clf()

            last_half = int((n_sample_states_st * n_sims_st) // 2)
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



    def get_ref_states(params, i, seed, prev_states_force, prev_states_torque, prev_basedir, resample_hpin):
        random.seed(seed)
        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        # Run the simulations
        if compute_st:
            stretch_tors_tasks, all_sim_dirs = get_stretch_tors_tasks(iter_dir, params, prev_states_force, prev_states_torque)
        if compute_60bp:
            bp60_tasks, all_sim_dirs_60bp = get_60bp_tasks(iter_dir, params, prev_basedir)

        if compute_hpin and resample_hpin:
            hpin_tasks, all_sim_dirs_hpin = get_hpin_tasks(iter_dir, params, recompile=(not compute_60bp))

        ## Archive the previous basedir now that we've loaded states from it
        if not no_archive and prev_basedir is not None:
            shutil.make_archive(prev_basedir, 'zip', prev_basedir)
            shutil.rmtree(prev_basedir)

        if compute_st:
            all_ret_info = ray.get(stretch_tors_tasks)
        if compute_60bp:
            all_ret_info_60bp = ray.get(bp60_tasks) # FIXME: for now, not doing anything with this! Just want to run the simulations and see them. Then, we do analysis and what not.
        if compute_hpin and resample_hpin:
            all_ret_info_hpin = ray.get(hpin_tasks)

        if compute_st:
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

        if compute_60bp:
            bp60_ref_info = process_60bp(iter_dir, params)
        else:
            bp60_ref_info = None
        if compute_st:
            stretch_tors_ref_info, all_force_t0_last_states, all_f2_torque_last_states = process_stretch_tors(iter_dir, params)
        else:
            stretch_tors_ref_info, all_force_t0_last_states, all_f2_torque_last_states = None, None, None
        if compute_hpin and resample_hpin:
            hpin_ref_info = process_hpin(iter_dir, params)
        else:
            hpin_ref_info = None

        return stretch_tors_ref_info, all_force_t0_last_states, all_f2_torque_last_states, bp60_ref_info, hpin_ref_info, iter_dir

    @jit
    def loss_fn(params, stretch_tors_ref_info, bp60_ref_info, hpin_ref_info):

        # 60bp simulation (Pitch, Rise, Persistence Length)
        if compute_60bp:
            ref_states, ref_energies, ref_avg_angles, unweighted_corr_curves, unweighted_l0_avgs, unweighted_rises = bp60_ref_info
            em = model.EnergyModel(displacement_fn_free, params, t_kelvin=t_kelvin_60bp)

            energy_fn = lambda body: em.energy_fn(body,
                                                  seq=seq_oh_60bp,
                                                  bonded_nbrs=top_info_60bp.bonded_nbrs,
                                                  unbonded_nbrs=top_info_60bp.unbonded_nbrs.T,
                                                  is_end=top_info_60bp.is_end)
            energy_fn = jit(energy_fn)
            new_energies = vmap(energy_fn)(ref_states)
            diffs = new_energies - ref_energies # element-wise subtraction
            boltzs = jnp.exp(-beta_60bp * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom

            expected_angle = jnp.dot(weights, ref_avg_angles)
            expected_pitch = 2*jnp.pi / expected_angle

            loss_pitch = uncertainty_loss_fn(expected_pitch, pitch_lo, pitch_hi)
            # mse = (expected_pitch - target_pitch)**2
            # rmse_pitch = jnp.sqrt(mse)
            # rel_diff_pitch = abs_relative_diff(target_pitch, expected_pitch)
            # rel_diff_pitch = abs_relative_diff_uncertainty(expected_pitch, pitch_lo, pitch_hi)


            expected_rise = jnp.dot(unweighted_rises, weights)
            expected_rise *= utils.nm_per_oxdna_length

            weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
            expected_corr_curve = jnp.sum(weighted_corr_curves, axis=0)
            expected_l0_avg = jnp.dot(unweighted_l0_avgs, weights)
            expected_lp, expected_offset = persistence_length.persistence_length_fit(
                expected_corr_curve[:truncation_lp],
                expected_l0_avg)
            expected_lp = expected_lp * utils.nm_per_oxdna_length
            expected_lp_n_bp = expected_lp / expected_rise

            # rel_diff_lp = abs_relative_diff_uncertainty(expected_lp, lp_lo, lp_hi) # note that this is in nm
            loss_lp = uncertainty_loss_fn(expected_lp, lp_lo, lp_hi)

            n_eff_60bp = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
        else:
            expected_pitch = -1
            # rmse_pitch = 0.0
            # rel_diff_pitch = 0.0
            loss_pitch = 0.0

            expected_rise = -1

            expected_lp = -1
            expected_lp_n_bp = -1
            expected_offset = -1
            # rel_diff_lp = -1
            loss_lp = 0.0
            expected_l0_avg = -1

            n_eff_60bp = n_ref_states_60bp


        # Stretch-torsion

        if compute_st:
            all_ref_states_f, all_ref_energies_f, all_ref_dists_f, all_ref_thetas_f, all_ref_states_t, all_ref_energies_t, all_ref_dists_t, all_ref_thetas_t = stretch_tors_ref_info

            # Setup energy function
            em = model.EnergyModel(displacement_fn_free,
                                   params,
                                   t_kelvin=t_kelvin_st,
                                   salt_conc=salt_conc_st, q_eff=q_eff_st, seq_avg=seq_avg,
                                   ignore_exc_vol_bonded=True # Because we're in LAMMPS
            )
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh_st,
                bonded_nbrs=top_info_st.bonded_nbrs,
                unbonded_nbrs=top_info_st.unbonded_nbrs.T)
            energy_fn = jit(energy_fn)
            energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

            def get_expected_vals(ref_states, ref_energies, ref_dists, ref_thetas):
                _, new_energies = scan(energy_scan_fn, None, ref_states)

                diffs = new_energies - ref_energies
                boltzs = jnp.exp(-beta_st * diffs)
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

            # rel_diff_s_eff = abs_relative_diff_uncertainty(s_eff, s_eff_lo, s_eff_hi)
            # rmse_s_eff = rmse_uncertainty(s_eff, s_eff_lo, s_eff_hi)
            loss_s_eff = uncertainty_loss_fn(s_eff, s_eff_lo, s_eff_hi)

            # rel_diff_c = abs_relative_diff_uncertainty(c, c_lo, c_hi)
            # rmse_c = rmse_uncertainty(c, c_lo, c_hi)
            loss_c = uncertainty_loss_fn(c, c_lo, c_hi)

            # rel_diff_g = abs_relative_diff_uncertainty(g, g_lo, g_hi)
            # rmse_g = rmse_uncertainty(g, g_lo, g_hi)
            loss_g = uncertainty_loss_fn(g, g_lo, g_hi)
        else:
            # rmse_s_eff, rmse_c, rmse_g = 0.0, 0.0, 0.0
            # rel_diff_s_eff, rel_diff_c, rel_diff_g = 0.0, 0.0, 0.0
            loss_s_eff, loss_c, loss_g = 0.0, 0.0, 0.0
            s_eff, c, g = -1, -1, -1
            a1, a3, a4 = -1, -1, -1
            n_effs_f, n_effs_t = jnp.full((n_forces,), n_sample_states_st*n_sims_st), jnp.full((n_torques,), n_sample_states_st*n_sims_st)
            # n_effs_f, n_effs_t = n_sample_states_st*n_sims_st, n_sample_states_st*n_sims_st


        # Hairpin

        if compute_hpin:
            ref_states, ref_energies, all_ops, all_op_weights, all_op_idxs = hpin_ref_info

            em = model.EnergyModel(displacement_fn_free, params, t_kelvin=t_kelvin_hpin, salt_conc=salt_concentration_hpin)
            energy_fn = lambda body: em.energy_fn(
                body,
                seq=seq_oh_hpin,
                bonded_nbrs=top_info_hpin.bonded_nbrs,
                unbonded_nbrs=top_info_hpin.unbonded_nbrs.T,
                is_end=top_info_hpin.is_end)
            energy_fn = jit(energy_fn)

            energy_scan_fn = lambda state, rs: (None, energy_fn(rs))
            _, new_energies = scan(energy_scan_fn, None, ref_states)

            diffs = new_energies - ref_energies # element-wise subtraction
            boltzs = jnp.exp(-beta_hpin * diffs)
            denom = jnp.sum(boltzs)
            weights = boltzs / denom

            def compute_extrap_temp_ratios(t_kelvin_extrap):
                extrap_kt = utils.get_kt(t_kelvin_extrap)
                em_temp = model.EnergyModel(displacement_fn_free, params, t_kelvin=t_kelvin_extrap, salt_conc=salt_concentration_hpin)
                energy_fn_temp = lambda body: em_temp.energy_fn(
                    body,
                    seq=seq_oh_hpin,
                    bonded_nbrs=top_info_hpin.bonded_nbrs,
                    unbonded_nbrs=top_info_hpin.unbonded_nbrs.T,
                    is_end=top_info_hpin.is_end)
                energy_fn_temp = jit(energy_fn_temp)

                def unbias_scan_fn(unb_counts, rs_idx):
                    rs = ref_states[rs_idx]
                    op1, op2 = all_ops[rs_idx]
                    op_idx = all_op_idxs[rs_idx]
                    op_weight = all_op_weights[rs_idx]

                    # calc_energy = ref_energies[rs_idx] # this is wrong
                    calc_energy = new_energies[rs_idx]
                    calc_energy_temp = energy_fn_temp(rs)

                    boltz_diff = jnp.exp(calc_energy/kT_hpin - calc_energy_temp/extrap_kt)

                    difftre_weight = weights[rs_idx]
                    # weighted_add_term = n_ref_states/difftre_weight * 1/op_weight * boltz_diff
                    weighted_add_term = n_ref_states_hpin*difftre_weight * 1/op_weight * boltz_diff
                    return unb_counts.at[op_idx].add(weighted_add_term), None

                temp_unbiased_counts, _ = scan(unbias_scan_fn, jnp.zeros(num_ops_hpin), jnp.arange(n_ref_states_hpin))
                temp_ratios = compute_ratio_hpin(temp_unbiased_counts)
                return temp_ratios

            ratios = vmap(compute_extrap_temp_ratios)(extrapolate_temps_hpin)
            curr_tm = tm.compute_tm(extrapolate_temps_hpin, ratios)
            curr_width = tm.compute_width(extrapolate_temps_hpin, ratios)

            n_eff_hpin = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
            # rel_diff_hpin = abs_relative_diff_uncertainty(curr_tm, tm_hpin_lo, tm_hpin_hi)
            loss_hpin = uncertainty_loss_fn(curr_tm, tm_hpin_lo, tm_hpin_hi)
        else:
            curr_tm, curr_width = -1, -1
            n_eff_hpin = n_ref_states_hpin
            # rel_diff_hpin = 0.0
            loss_hpin = 0.0


        # loss = s_eff_coeff*rmse_s_eff + c_coeff*rmse_c + g_coeff*rmse_g
        # loss = s_eff_coeff*rel_diff_s_eff + c_coeff*rel_diff_c + g_coeff*rel_diff_g + pitch_coeff*rel_diff_pitch + hpin_coeff*rel_diff_hpin + lp_coeff*rel_diff_lp
        loss = s_eff_coeff*loss_s_eff + c_coeff*loss_c + g_coeff*loss_g + pitch_coeff*loss_pitch + hpin_coeff*loss_hpin + lp_coeff*loss_lp

        return loss, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g, expected_pitch, expected_rise, expected_lp, expected_l0_avg, expected_lp_n_bp, n_eff_60bp, curr_tm, curr_width, n_eff_hpin, loss_s_eff, loss_c, loss_g, loss_pitch, loss_hpin, loss_lp)
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


    # Setup orbax checkpointing
    ex_ckpt = {"params": params, "optimizer": optimizer, "opt_state": opt_state}
    save_args = orbax_utils.save_args_from_target(ex_ckpt)

    ckpt_dir = run_dir / "ckpt/orbax/managed/"
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=10, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(str(ckpt_dir.resolve()), orbax_checkpointer, options) # note: checkpoint directory has to be an absoltue path

    ## Load orbax checkpoint if necessary
    if orbax_ckpt_path is not None:
        state_restored = orbax_checkpointer.restore(orbax_ckpt_path, item=ex_ckpt)
        params = state_restored["params"]
        # optimizer = state_restored["optimizer"]
        opt_state = state_restored["opt_state"]


    min_n_eff_st = int(n_sample_states_st*n_sims_st * min_neff_factor)

    all_losses = list()
    all_n_effs_st = list()
    all_seffs = list()
    all_cs = list()
    all_gs = list()
    all_pitches = list()
    all_rises = list()
    all_tms = list()
    all_widths = list()
    all_lps = list()
    all_l0s = list()

    all_ref_losses = list()
    all_ref_times = list()
    all_ref_seffs = list()
    all_ref_cs = list()
    all_ref_gs = list()
    all_ref_pitches = list()
    all_ref_rises = list()
    all_ref_tms = list()
    all_ref_widths = list()
    all_ref_lps = list()
    all_ref_l0s = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    start = time.time()
    prev_ref_basedir = None
    stretch_tors_ref_info, prev_last_states_force, prev_last_states_torque, bp60_ref_info, hpin_ref_info, ref_iter_dir = get_ref_states(params, i=0, seed=30362, prev_states_force=None, prev_states_torque=None, prev_basedir=prev_ref_basedir, resample_hpin=True)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()
        (loss, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g, expected_pitch, expected_rise, curr_lp, curr_l0_avg, curr_lp_n_bp, n_eff_60bp, expected_tm, expected_width, n_eff_hpin, loss_s_eff, loss_c, loss_g, loss_pitch, loss_hpin, loss_lp)), grads = grad_fn(params, stretch_tors_ref_info, bp60_ref_info, hpin_ref_info)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_seffs.append(s_eff)
            all_ref_cs.append(c)
            all_ref_gs.append(g)
            all_ref_pitches.append(expected_pitch)
            all_ref_rises.append(expected_rise)
            all_ref_tms.append(expected_tm)
            all_ref_widths.append(expected_width)
            all_ref_lps.append(curr_lp)
            all_ref_l0s.append(curr_l0_avg)

        resample = False
        n_effs_st = jnp.concatenate([n_effs_f, n_effs_t])
        for n_eff in n_effs_st:
            if n_eff < min_n_eff_st:
                resample = True
                break
        if n_eff_60bp < min_n_eff_60bp:
            resample = True
        resample_hpin = False
        if n_eff_hpin < min_n_eff_hpin:
            resample = True
            resample_hpin = True

        if resample or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- min n_eff_st was {n_effs_st.min()}...")

            start = time.time()
            stretch_tors_ref_info, prev_last_states_force, prev_last_states_torque, bp60_ref_info, new_hpin_ref_info, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_states_force=prev_last_states_force, prev_states_torque=prev_last_states_torque, prev_basedir=prev_ref_basedir, resample_hpin=resample_hpin)
            end = time.time()
            if resample_hpin:
                hpin_ref_info = deepcopy(new_hpin_ref_info)
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_effs_f, n_effs_t, a1, a3, a4, s_eff, c, g, expected_pitch, expected_rise, curr_lp, curr_l0_avg, curr_lp_n_bp, n_eff_60bp, expected_tm, expected_width, n_eff_hpin, loss_s_eff, loss_c, loss_g, loss_pitch, loss_hpin, loss_lp)), grads = grad_fn(params, stretch_tors_ref_info, bp60_ref_info, hpin_ref_info)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_seffs.append(s_eff)
            all_ref_cs.append(c)
            all_ref_gs.append(g)
            all_ref_pitches.append(expected_pitch)
            all_ref_rises.append(expected_rise)
            all_ref_tms.append(expected_tm)
            all_ref_widths.append(expected_width)
            all_ref_lps.append(curr_lp)
            all_ref_l0s.append(curr_l0_avg)

        iter_end = time.time()


        with open(s_eff_loss_path, "a") as f:
            f.write(f"{loss_s_eff}\n")
        with open(c_loss_path, "a") as f:
            f.write(f"{loss_c}\n")
        with open(g_loss_path, "a") as f:
            f.write(f"{loss_g}\n")
        with open(pitch_loss_path, "a") as f:
            f.write(f"{loss_pitch}\n")
        with open(hpin_loss_path, "a") as f:
            f.write(f"{loss_hpin}\n")
        with open(lp_loss_path, "a") as f:
            f.write(f"{loss_lp}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")
        with open(l0_avg_path, "a") as f:
            f.write(f"{curr_l0_avg}\n")
        with open(tm_path, "a") as f:
            f.write(f"{expected_tm}\n")
        with open(width_path, "a") as f:
            f.write(f"{expected_width}\n")
        with open(pitch_path, "a") as f:
            f.write(f"{expected_pitch}\n")
        with open(rise_path, "a") as f:
            f.write(f"{expected_rise}\n")
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
        with open(neffs_st_path, "a") as f:
            f.write(f"{n_effs_st}\n")
        with open(neff_60bp_path, "a") as f:
            f.write(f"{n_eff_60bp}\n")
        with open(neff_hpin_path, "a") as f:
            f.write(f"{n_eff_hpin}\n")
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
        all_pitches.append(expected_pitch)
        all_tms.append(expected_tm)
        all_widths.append(expected_width)
        all_n_effs_st.append(n_effs_st)
        all_lps.append(curr_lp)
        all_l0s.append(curr_l0_avg)

        if i % ckpt_freq == 0:
            ckpt = {"params": params, "optimizer": optimizer, "opt_state": opt_state}
            checkpoint_manager.save(i, ckpt, save_kwargs={'save_args': save_args})

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

            plt.plot(onp.arange(i+1), all_tms, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_tms, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_tm_hpin, linestyle='--', label="Target Tm", color='red')
            plt.legend()
            plt.ylabel("Expected Tm")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"tms_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_widths, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_widths, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Expected Melting Width")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"widths_iter{i}.png")
            plt.clf()


        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)

            onp.save(obj_dir / f"ref_seffs_i{i}.npy", onp.array(all_ref_seffs), allow_pickle=False)
            onp.save(obj_dir / f"seffs_i{i}.npy", onp.array(all_seffs), allow_pickle=False)

            onp.save(obj_dir / f"ref_gs_i{i}.npy", onp.array(all_ref_gs), allow_pickle=False)
            onp.save(obj_dir / f"gs_i{i}.npy", onp.array(all_gs), allow_pickle=False)

            onp.save(obj_dir / f"ref_cs_i{i}.npy", onp.array(all_ref_cs), allow_pickle=False)
            onp.save(obj_dir / f"cs_i{i}.npy", onp.array(all_cs), allow_pickle=False)

            onp.save(obj_dir / f"ref_pitches_i{i}.npy", onp.array(all_ref_pitches), allow_pickle=False)
            onp.save(obj_dir / f"pitches_i{i}.npy", onp.array(all_pitches), allow_pickle=False)

            onp.save(obj_dir / f"ref_rises_i{i}.npy", onp.array(all_ref_rises), allow_pickle=False)
            onp.save(obj_dir / f"rises_i{i}.npy", onp.array(all_rises), allow_pickle=False)

            onp.save(obj_dir / f"ref_tms_i{i}.npy", onp.array(all_ref_tms), allow_pickle=False)
            onp.save(obj_dir / f"tms_i{i}.npy", onp.array(all_tms), allow_pickle=False)

            onp.save(obj_dir / f"ref_widths_i{i}.npy", onp.array(all_ref_widths), allow_pickle=False)
            onp.save(obj_dir / f"widths_i{i}.npy", onp.array(all_widths), allow_pickle=False)

            onp.save(obj_dir / f"ref_lps_i{i}.npy", onp.array(all_ref_lps), allow_pickle=False)
            onp.save(obj_dir / f"lps_i{i}.npy", onp.array(all_lps), allow_pickle=False)

            onp.save(obj_dir / f"ref_l0s_i{i}.npy", onp.array(all_ref_l0s), allow_pickle=False)
            onp.save(obj_dir / f"l0s_i{i}.npy", onp.array(all_l0s), allow_pickle=False)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_seffs.npy", onp.array(all_ref_seffs), allow_pickle=False)
    onp.save(obj_dir / f"fin_seffs.npy", onp.array(all_seffs), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_gs.npy", onp.array(all_ref_gs), allow_pickle=False)
    onp.save(obj_dir / f"fin_gs.npy", onp.array(all_gs), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_cs.npy", onp.array(all_ref_cs), allow_pickle=False)
    onp.save(obj_dir / f"fin_cs.npy", onp.array(all_cs), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_pitches.npy", onp.array(all_ref_pitches), allow_pickle=False)
    onp.save(obj_dir / f"fin_pitches.npy", onp.array(all_pitches), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_rises.npy", onp.array(all_ref_rises), allow_pickle=False)
    onp.save(obj_dir / f"fin_rises.npy", onp.array(all_rises), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_tms.npy", onp.array(all_ref_tms), allow_pickle=False)
    onp.save(obj_dir / f"fin_tms.npy", onp.array(all_tms), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_widths.npy", onp.array(all_ref_widths), allow_pickle=False)
    onp.save(obj_dir / f"fin_widths.npy", onp.array(all_widths), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_lps.npy", onp.array(all_ref_lps), allow_pickle=False)
    onp.save(obj_dir / f"fin_lps.npy", onp.array(all_lps), allow_pickle=False)

    onp.save(obj_dir / f"fin_ref_l0s.npy", onp.array(all_ref_l0s), allow_pickle=False)
    onp.save(obj_dir / f"fin_l0s.npy", onp.array(all_l0s), allow_pickle=False)


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
    parser.add_argument('--n-sample-steps-st', type=int, default=3000000,
                        help="Number of steps per simulation for stretch-torsion")
    parser.add_argument('--n-eq-steps-st', type=int, default=100000,
                        help="Number of equilibration steps for stretch-torsion")
    parser.add_argument('--sample-every-st', type=int, default=500,
                        help="Frequency of sampling reference states for stretch-torsion.")
    parser.add_argument('--n-sims-st', type=int, default=2,
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

    parser.add_argument('--no-compute-st', action='store_true')


    # Structural information

    parser.add_argument('--n-steps-per-sim-60bp', type=int, default=100000,
                        help="Number of steps for sampling reference states per simulation for structural info")
    parser.add_argument('--n-eq-steps-60bp', type=int, default=0,
                        help="Number of equilibration steps for structural info")
    parser.add_argument('--sample-every-60bp', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims-60bp', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--offset-60bp', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--target-pitch', type=float, default=10.25,
                        help="Target pitch in number of bps")
    parser.add_argument('--pitch-uncertainty', type=float, default=0.25,
                        help="Uncertainty for pitch")

    parser.add_argument('--no-compute-60bp', action='store_true')
    parser.add_argument('--pitch-coeff', type=float, default=0.0,
                        help="Coefficient for pitch component")

    ## Persistence length (uses same simulation)
    parser.add_argument('--target-lp', type=float, default=42.0,
                        help="Target persistence length in nanometers")
    parser.add_argument('--lp-uncertainty', type=float, default=2.0,
                        help="Uncertainty for Lp in nanometers")
    parser.add_argument('--truncation-lp', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--lp-coeff', type=float, default=0.0,
                        help="Coefficient for Lp component")


    # Hairpin Tm information
    parser.add_argument('--n-sims-hpin', type=int, default=2,
                        help="Number of individual simulations for hairpin tm")
    parser.add_argument('--n-steps-per-sim-hpin', type=int, default=int(5e6),
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps-hpin', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every-hpin', type=int, default=int(1e3),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--temp-hpin', type=float, default=330.15,
                        help="Temperature in kelvin")
    parser.add_argument('--extrapolate-temps-hpin', nargs='+',
                        help='Temperatures for extrapolation in Kelvin in ascending order',
                        required=True)
    parser.add_argument('--salt-concentration-hpin', type=float, default=0.5,
                        help="Salt concentration in molar (M)")

    parser.add_argument('--stem-bp-hpin', type=int, default=6,
                        help="Number of base pairs comprising the stem")
    parser.add_argument('--loop-nt-hpin', type=int, default=6,
                        help="Number of nucleotides comprising the loop")
    parser.add_argument('--target-tm-hpin', type=float, required=True,
                        help="Target melting temperature in Kelvin")
    parser.add_argument('--tm-hpin-uncertainty', type=float, default=0.5,
                        help="Uncertainty for hairpin Tm")

    parser.add_argument('--no-compute-hpin', action='store_true')
    parser.add_argument('--hpin-coeff', type=float, default=0.0,
                        help="Coefficient for hairpin Tm")

    parser.add_argument('--min-neff-factor-hpin', type=float, default=0.85,
                        help="Factor for determining min Neff for hairpin")

    parser.add_argument('--no-standardize', action='store_true')

    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help='Checkpointing frequency')
    parser.add_argument('--orbax-ckpt-path', type=str, required=False,
                        help='Optional path to orbax checkpoint directory')

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
