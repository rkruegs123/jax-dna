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
import zipfile
import os

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", 'cpu')

import optax
import jax.numpy as jnp
from jax_md import space
from jax import vmap, jit, lax, grad, value_and_grad

from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.loss.rna import geom_rise, geom_angle
from jax_dna.loss.rna.geom_a_with_bb import rise as geom_rise_alt
from jax_dna.loss.rna.geom_a_with_bb import angle as geom_angle_alt
from jax_dna.loss.rna.geom_a_with_bb import inclination as geom_inclination_alt
from jax_dna.rna2 import model, oxrna_utils
from jax_dna.rna2.load_params import read_seq_specific, DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS
import jax_dna.input.trajectory as jdt

from jax_dna.dna1 import oxdna_utils
from jax_dna.dna1 import model as model_dna1
from jax_dna.dna2.oxdna_utils import recompile_oxdna




checkpoint_every = 50
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(
        checkpoint.checkpoint_scan,
        checkpoint_every=checkpoint_every
    )




@jit
def get_site_positions(body):
    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
    base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
    cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

    # RNA values
    com_to_backbone_x = -0.4
    com_to_backbone_y = 0.2
    com_to_stacking = 0.34
    com_to_hb = 0.4
    back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals
    stack_sites = body.center + com_to_stacking * back_base_vectors
    base_sites = body.center + com_to_hb * back_base_vectors

    return back_sites, stack_sites, base_sites



@jit
def fit_plane(points):
    """Fits plane through points, return normal to the plane."""

    n_points = points.shape[0]
    rc = jnp.sum(points, axis=0) / n_points
    points_centered = points - rc

    As = vmap(lambda point: jnp.kron(point, point).reshape(3, 3))(points_centered)
    A = jnp.sum(As, axis=0)
    vals, vecs = jnp.linalg.eigh(A)
    return vecs[:, 0]



@functools.partial(jax.jit, static_argnums=(5, 6))
def get_localized_axis(body, back_sites, stack_sites, base_sites, base_id, down_length, up_length):

    n = body.center.shape[0]

    def get_base_plane_nuc_info(i):
        midpoint_A = 0.5*(stack_sites[i] + stack_sites[n-i-1])
        midpoint_B = 0.5*(stack_sites[n-i-1-1] + stack_sites[i+1])

        guess = -midpoint_A + midpoint_B

        p1 = back_sites[i] - back_sites[i+1]
        p2 = back_sites[n-i-1] - back_sites[n-i-1-1]
        p3 = midpoint_A - midpoint_B
        return guess, jnp.array([p1, p2, p3])

    n_base_plane_idxs = 1 + down_length + up_length
    base_plane_idxs = base_id-down_length + jnp.arange(n_base_plane_idxs)

    # base_plane_idxs = jnp.arange(base_id-down_length, base_id+up_length+1)
    guesses, plane_points = vmap(get_base_plane_nuc_info)(base_plane_idxs)
    plane_points = plane_points.reshape(-1, 3)

    mean_guess = jnp.mean(guesses, axis=0)
    mean_guess = mean_guess / jnp.linalg.norm(mean_guess)
    plane_vector = fit_plane(plane_points)

    plane_vector = jnp.where(jnp.dot(mean_guess, plane_vector) < 0, -1.0*plane_vector, plane_vector)

    return plane_vector / jnp.linalg.norm(plane_vector)



def get_rises(body, first_base, last_base):

    back_sites, stack_sites, base_sites = get_site_positions(body)

    back_poses = []
    n = body.center.shape[0]

    def get_back_positions(i):
        nt11 = i
        nt12 = n-i-1

        nt21 = i+1
        nt22 = n-(i+1)-1

        back_pos1 = stack_sites[nt11] - stack_sites[nt21]
        back_pos2 = stack_sites[nt12] - stack_sites[nt22]
        return jnp.array([back_pos1, back_pos2])
    back_poses = vmap(get_back_positions)(jnp.arange(first_base, last_base))
    back_poses = back_poses.reshape(-1, 3)

    # now we try to fit a plane through all these points
    plane_vector = fit_plane(back_poses)

    midp_first_base = (stack_sites[first_base] + stack_sites[n-first_base-1]) / 2

    midp_last_base = (stack_sites[last_base] + stack_sites[n-last_base-1]) / 2
    guess = (-midp_first_base + midp_last_base) # vector pointing from midpoint of the first bp to the last bp, an estimate of the vector
    guess = guess / jnp.linalg.norm(guess)


    # Check if plane vector is pointing in opposite direction
    plane_vector = jnp.where(jnp.dot(guess, plane_vector) < 0, -1.0*plane_vector, plane_vector)

    """
    if (onp.rad2deg(math.acos(np.dot(plane_vector,guess/my_norm(guess)))) > 20):
        # print 'Warning, guess vector and plane vector have angles:', np.rad2deg(math.acos(np.dot(guess/my_norm(guess),plane_vector)))
        pdb.set_trace()
        pass
    """

    # Now, compute the rises
    n_bps_lp = last_base - first_base

    def single_rise(bp_idx):
        i = first_base + bp_idx

        midp = (stack_sites[i] + stack_sites[n-i-1]) / 2
        midp_ip1 = (stack_sites[i+1] + stack_sites[n-(i+1)-1]) / 2

        midp_proj = jnp.dot(plane_vector, midp)
        midp_ip1_proj = jnp.dot(plane_vector, midp_ip1)

        rise = midp_ip1_proj - midp_proj
        return rise
    rises = vmap(single_rise)(jnp.arange(n_bps_lp))

    return rises.mean()


def get_corrs_jax_full(body, base_start, down_neigh, up_neigh, max_dist):
    back_sites, stack_sites, base_sites = get_site_positions(body)
    bp_idxs = base_start + jnp.arange(max_dist)
    compute_l_vector = lambda bp_idx: get_localized_axis(body, back_sites, stack_sites, base_sites, bp_idx, down_neigh, up_neigh)
    all_l_vectors = vmap(compute_l_vector)(bp_idxs)
    autocorr = persistence_length.vector_autocorrelate(all_l_vectors)
    return autocorr

def get_corrs_jax(body, base_start, down_neigh, up_neigh, max_dist):

    back_sites, stack_sites, base_sites = get_site_positions(body)
    l_0 = get_localized_axis(body, back_sites, stack_sites, base_sites, base_start, down_neigh, up_neigh)

    corr_dists = jnp.arange(max_dist) # FIXME: max_dist is really one too big. It's really one more than the max dist
    def get_corr(dist):
        i = base_start + dist
        l_i = get_localized_axis(body, back_sites, stack_sites, base_sites, i, down_neigh, up_neigh)
        corr = jnp.dot(l_0, l_i)
        return corr
    body_corrs_jax = vmap(get_corr)(corr_dists)

    return body_corrs_jax



def zip_file(file_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        zipf.write(file_path, os.path.basename(file_path))

def abs_relative_diff(target, current):
    rel_diff = (current - target) / target
    return jnp.sqrt(rel_diff**2)

def run(args):
    # Load parameters
    n_threads = args['n_threads']

    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    salt_conc = args['salt_concentration']
    no_delete = args['no_delete']
    no_archive = args['no_archive']
    plot_every = args['plot_every']
    save_obj_every = args['save_obj_every']
    t_kelvin = utils.DEFAULT_TEMP
    seq_avg_opt_keys = args['seq_avg_opt_keys']

    standardize = not args['no_standardize']

    ## Persistence length
    n_sims_lp = args['n_sims_lp']
    n_steps_per_sim_lp = args['n_steps_per_sim_lp']
    n_eq_steps_lp = args['n_eq_steps_lp']
    sample_every_lp = args['sample_every_lp']
    assert(n_steps_per_sim_lp % sample_every_lp == 0)
    n_ref_states_per_sim_lp = n_steps_per_sim_lp // sample_every_lp
    n_ref_states_lp = n_ref_states_per_sim_lp * n_sims_lp
    offset = args['offset']
    truncation = args['truncation']
    target_lp = args['target_lp']

    ## Structure
    n_sims_struc = args['n_sims_struc']
    n_steps_per_sim_struc = args['n_steps_per_sim_struc']
    n_eq_steps_struc = args['n_eq_steps_struc']
    sample_every_struc = args['sample_every_struc']
    assert(n_steps_per_sim_struc % sample_every_struc == 0)
    n_ref_states_per_sim_struc = n_steps_per_sim_struc // sample_every_struc
    n_ref_states_struc = n_ref_states_per_sim_struc * n_sims_struc
    offset_struc = args['offset_struc']

    target_rise = args['target_rise']
    target_pitch = args['target_pitch']
    target_inclination = args['target_inclination']




    # Recompile oxDNA
    recompile_params = deepcopy(model_dna1.EMPTY_BASE_PARAMS)
    oxdna_utils.recompile_oxdna(recompile_params, oxdna_path, t_kelvin, num_threads=n_threads)


    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    obj_dir = run_dir / "obj"
    obj_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)


    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    neff_lp_path = log_dir / "neff_lp.txt"
    neff_struc_path = log_dir / "neff_struc.txt"
    lp_path = log_dir / "lp.txt"
    rise_from_lp_path = log_dir / "rise_from_lp.txt"
    rise_path = log_dir / "rise.txt"
    pitch_path = log_dir / "pitch.txt"
    inclination_path = log_dir / "inclination.txt"
    rmse_lp_path = log_dir / "rmse_lp.txt"
    rmse_rise_path = log_dir / "rmse_rise.txt"
    rmse_pitch_path = log_dir / "rmse_pitch.txt"
    rmse_inclination_path = log_dir / "rmse_inclination.txt"
    lp_n_bp_path = log_dir / "lp_n_bp.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states (Lp): {n_ref_states_lp}\n"
    params_str += f"n_ref_states (structure): {n_ref_states_struc}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system

    ## Persistence length
    lp_sys_basedir = Path("data/templates/ds-142bp-rna")
    input_template_path_lp = lp_sys_basedir / "input"

    top_path_lp = lp_sys_basedir / "sys.top"
    top_info_lp = topology.TopologyInfo(top_path_lp, reverse_direction=False, is_rna=True)
    seq_oh_lp = jnp.array(utils.get_one_hot(top_info_lp.seq, is_rna=True), dtype=jnp.float64)

    n_bp_lp = int(seq_oh_lp.shape[0] // 2)

    conf_path_lp = lp_sys_basedir / "init.conf"
    conf_info_lp = trajectory.TrajectoryInfo(
        top_info_lp,
        read_from_file=True, traj_path=conf_path_lp,
        reverse_direction=False

    )
    centered_conf_info_lp = center_configuration.center_conf(top_info_lp, conf_info_lp)
    box_size_lp = conf_info_lp.box_size


    ## Structure
    struc_sys_basedir = Path("data/templates/rna2-13bp-md")
    input_template_path_struc = struc_sys_basedir / "input"

    top_path_struc = struc_sys_basedir / "sys.top"
    top_info_struc = topology.TopologyInfo(top_path_struc, reverse_direction=False, is_rna=True)
    seq_oh_struc = jnp.array(utils.get_one_hot(top_info_struc.seq, is_rna=True), dtype=jnp.float64)

    n_bp_struc = int(seq_oh_struc.shape[0] // 2)
    assert(n_bp_struc == 13)

    conf_path_struc = struc_sys_basedir / "init.conf"
    conf_info_struc = trajectory.TrajectoryInfo(
        top_info_struc,
        read_from_file=True,
        traj_path=conf_path_struc,
        reverse_direction=False

    )
    centered_conf_info_struc = center_configuration.center_conf(top_info_struc, conf_info_struc)
    box_size_struc = conf_info_struc.box_size



    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT


    # Do the simulation
    def get_ref_states(params, i, seed, prev_basedir):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        lp_dir = iter_dir / "lp"
        lp_dir.mkdir(parents=False, exist_ok=False)

        struc_dir = iter_dir / "struc"
        struc_dir.mkdir(parents=False, exist_ok=False)

        sim_start = time.time()
        procs = list()

        # Start Lp simulations
        for r in range(n_sims_lp):
            repeat_dir = lp_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path_lp, repeat_dir / "sys.top")


            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info_lp)
            else:

                prev_lp_basedir = prev_basedir / "lp"

                prev_repeat_dir = prev_lp_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info_lp,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info_lp, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh_lp.shape[0], r*n_steps_per_sim_lp)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            external_model_fpath = repeat_dir / "external_model.txt"
            oxrna_utils.write_external_model(params["seq_avg"], t_kelvin, salt_conc, external_model_fpath)

            oxdna_utils.rewrite_input_file(
                input_template_path_lp,
                repeat_dir,
                temp=f"{t_kelvin}K",
                steps=n_steps_per_sim_lp,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_lp,
                seed=random.randrange(1000),
                equilibration_steps=n_eq_steps_lp,
                dt=dt,
                no_stdout_energy=0,
                log_file=str(repeat_dir / "sim.log"),
                external_model=str(external_model_fpath),
                salt_concentration=salt_conc
            )

            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        # Start structure simulations
        for r in range(n_sims_struc):
            repeat_dir = struc_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path_struc, repeat_dir / "sys.top")


            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info_struc)
            else:

                prev_struc_basedir = prev_basedir / "struc"

                prev_repeat_dir = prev_struc_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info_struc,
                    read_from_file=True,
                    traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info_struc,
                    prev_lastconf_info
                )

            init_conf_info.traj_df.t = onp.full(seq_oh_struc.shape[0], r*n_steps_per_sim_struc)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            external_model_fpath = repeat_dir / "external_model.txt"
            oxrna_utils.write_external_model(params["seq_avg"], t_kelvin, salt_conc, external_model_fpath)

            oxdna_utils.rewrite_input_file(
                input_template_path_struc,
                repeat_dir,
                temp=f"{t_kelvin}K",
                steps=n_steps_per_sim_struc,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every_struc,
                seed=random.randrange(1000),
                equilibration_steps=n_eq_steps_struc,
                dt=dt,
                no_stdout_energy=0,
                log_file=str(repeat_dir / "sim.log"),
                external_model=str(external_model_fpath),
                salt_concentration=salt_conc
            )

            procs.append(subprocess.Popen([oxdna_exec_path, repeat_dir / "input"]))

        for p in procs:
            p.wait()

        for p in procs:
            rc = p.returncode
            if rc != 0:
                raise RuntimeError(f"oxDNA simulation failed with error code: {rc}")

        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulation took {sim_end - sim_start} seconds\n")

        # Combine trajectories
        combine_cmd = "cat "
        for r in range(n_sims_lp):
            repeat_dir = lp_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {lp_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        combine_cmd = "cat "
        for r in range(n_sims_struc):
            repeat_dir = struc_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {struc_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        if not no_delete:
            files_to_remove = ["output.dat"]
            for r in range(n_sims_lp):
                repeat_dir = lp_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()

            for r in range(n_sims_struc):
                repeat_dir = struc_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()


        # Analyze (Lp)

        ## Load states from oxDNA simulation
        load_start = time.time()
        strand_length = int(seq_oh_lp.shape[0] // 2)
        traj_ = jdt.from_file(
            lp_dir / "output.dat",
            [strand_length, strand_length],
            is_oxdna=False,
            n_processes=n_threads,
        )
        traj_states = [ns.to_rigid_body() for ns in traj_.states]
        n_traj_states = len(traj_states)
        traj_states = utils.tree_stack(traj_states)
        load_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Loading took {load_end - load_start} seconds\n")

        ## Load the oxDNA energies

        energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
        energy_dfs = [pd.read_csv(lp_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims_lp)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_lp,
            bonded_nbrs=top_info_lp.bonded_nbrs,
            unbonded_nbrs=top_info_lp.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # Check energies
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")


        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh_lp.shape[0]

        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        ## Plot the energies
        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(lp_dir / f"energies.png")
        plt.clf()

        ## Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(lp_dir / f"energy_diffs.png")
        plt.clf()

        ## Compute the persistence lengths
        analyze_start = time.time()

        down_neigh = 1
        up_neigh = 1
        max_dist = n_bp_lp - offset * 2
        all_rises = vmap(get_rises, (0, None, None))(traj_states, offset, n_bp_lp-offset-2)
        unweighted_corr_curves = vmap(get_corrs_jax, (0, None, None, None, None))(traj_states, offset, down_neigh, up_neigh, max_dist)
        mean_corr_curve = jnp.mean(unweighted_corr_curves, axis=0)
        unweighted_corr_curves_full = vmap(get_corrs_jax_full, (0, None, None, None, None))(traj_states, offset, down_neigh, up_neigh, max_dist)
        mean_corr_curve_full = jnp.mean(unweighted_corr_curves_full, axis=0)

        avg_rise = jnp.mean(all_rises)

        mean_Lp_truncated, fit_offset = persistence_length.persistence_length_fit(mean_corr_curve[:truncation], avg_rise)
        mean_Lp_truncated_full, fit_offset_full = persistence_length.persistence_length_fit(mean_corr_curve_full[:truncation], avg_rise)

        compute_every = 10
        n_curves = unweighted_corr_curves.shape[0]
        all_inter_lps = list()
        all_inter_lps_truncated = list()
        all_inter_lps_full = list()
        all_inter_lps_truncated_full = list()
        for i in range(0, n_curves, compute_every):
            # Not full version
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], avg_rise)
            all_inter_lps_truncated.append(inter_mean_Lp_truncated * utils.nm_per_oxrna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, avg_rise)
            all_inter_lps.append(inter_mean_Lp * utils.nm_per_oxrna_length)

            # Full version
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves_full[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], avg_rise)
            all_inter_lps_truncated_full.append(inter_mean_Lp_truncated * utils.nm_per_oxrna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, avg_rise)
            all_inter_lps_full.append(inter_mean_Lp * utils.nm_per_oxrna_length)

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

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_full)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(lp_dir / "running_avg_full.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated_full)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(lp_dir / "running_avg_truncated_full.png")
        plt.clf()

        plt.plot(mean_corr_curve)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(lp_dir / "full_corr_curve.png")
        plt.clf()

        plt.plot(mean_corr_curve_full)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(lp_dir / "full_corr_curve_full.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "full_log_corr_curve.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve_full))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "full_log_corr_curve_full.png")
        plt.clf()

        fit_fn = lambda n: -n * (avg_rise / mean_Lp_truncated) + fit_offset
        plt.plot(jnp.log(mean_corr_curve)[:truncation])
        neg_inverse_slope = mean_Lp_truncated / avg_rise # in nucleotides
        rounded_offset = onp.round(fit_offset, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation)), linestyle='--', label=fit_str)

        plt.title(f"Log-Correlation Curve, Truncated.")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "log_corr_curve.png")
        plt.clf()


        fit_fn = lambda n: -n * (avg_rise / mean_Lp_truncated_full) + fit_offset_full
        plt.plot(jnp.log(mean_corr_curve_full)[:truncation])
        neg_inverse_slope = mean_Lp_truncated_full / avg_rise # in nucleotides
        rounded_offset = onp.round(fit_offset_full, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation)), linestyle='--', label=fit_str)

        plt.title(f"Log-Correlation Curve, Truncated.")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(lp_dir / "log_corr_curve_full.png")
        plt.clf()


        ## Record the loss
        with open(lp_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Max energy diff: {onp.max(energy_diffs)}\n")
            f.write(f"Min energy diff: {onp.min(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

            f.write(f"\nMean Lp truncated (oxDNA units): {mean_Lp_truncated}\n")
            f.write(f"Mean Rise (oxDNA units): {avg_rise}\n")
            f.write(f"Mean Lp truncated (num bp via oxDNA units): {mean_Lp_truncated / avg_rise}\n")

            f.write(f"\nMean Lp truncated (nm): {mean_Lp_truncated * utils.nm_per_oxrna_length}\n")


            f.write(f"\nMean Lp truncated, full (oxDNA units): {mean_Lp_truncated_full}\n")
            f.write(f"Mean Lp truncated, full (num bp via oxDNA units): {mean_Lp_truncated_full / avg_rise}\n")

            f.write(f"\nMean Lp truncated, full (nm): {mean_Lp_truncated_full * utils.nm_per_oxrna_length}\n")

        # Analyze (structure)

        ## Load states from oxDNA simulation
        strand_length = int(seq_oh_struc.shape[0] // 2)
        traj_ = jdt.from_file(
            struc_dir / "output.dat",
            [strand_length, strand_length],
            is_oxdna=False,
            n_processes=n_threads,
        )
        traj_states_struc = [ns.to_rigid_body() for ns in traj_.states]
        n_traj_states_struc = len(traj_states_struc)
        traj_states_struc = utils.tree_stack(traj_states_struc)

        ## Load the oxDNA energies
        energy_df_columns = ["time", "potential_energy", "kinetic_energy", "total_energy"]
        energy_dfs = [pd.read_csv(struc_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims_struc)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc
        )
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_struc,
            bonded_nbrs=top_info_struc.bonded_nbrs,
            unbonded_nbrs=top_info_struc.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # Check energies
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies_struc = scan(energy_scan_fn, None, traj_states_struc)

        gt_energies_struc = energy_df.potential_energy.to_numpy() * seq_oh_struc.shape[0]

        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies_struc, gt_energies_struc)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        ## Plot the energies
        sns.distplot(calc_energies_struc, label="Calculated", color="red")
        sns.distplot(gt_energies_struc, label="Reference", color="green")
        plt.legend()
        plt.savefig(struc_dir / f"energies.png")
        plt.clf()


        ## Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(struc_dir / f"energy_diffs.png")
        plt.clf()

        ## Compute the rise (with geom_rise)
        first_base = offset_struc
        last_base = n_bp_struc-offset_struc-1
        def compute_body_rise(body):
            """
            n = body.center.shape[0]

            Q = body.orientation
            back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized

            com_to_hb = 0.4

            base_sites = body.center + com_to_hb * back_base_vectors

            return geom_rise.get_rises(base_sites, n, first_base, last_base)
            """
            return geom_rise_alt.get_mean_rise(body, first_base, last_base)
        all_rises_struc = vmap(compute_body_rise)(traj_states_struc)
        assert(all_rises_struc.shape[0] == n_ref_states_struc)
        assert(len(all_rises_struc.shape) == 1)

        rise_running_avg = onp.cumsum(all_rises_struc) / onp.arange(1, n_ref_states_struc+1)
        plt.plot(rise_running_avg)
        plt.xlabel("Samples")
        plt.ylabel("Rise (oxDNA units)")
        plt.savefig(struc_dir / f"rise_running_avg.png")
        plt.close()

        mean_rise_struc = all_rises_struc.mean()

        ## Compute the pitch (with geom_angle)

        def compute_body_angle(body):
            """
            n = body.center.shape[0]

            Q = body.orientation
            back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
            base_normals = utils.Q_to_base_normal(Q) # space frame, normalized

            com_to_backbone_x = -0.4
            com_to_backbone_y = 0.2
            back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals

            com_to_hb = 0.4
            base_sites = body.center + com_to_hb * back_base_vectors

            return geom_angle.get_angles(base_sites, back_sites, n, first_base, last_base)
            """
            return geom_angle_alt.get_mean_angle(body, first_base, last_base)
        all_angles_struc = vmap(compute_body_angle)(traj_states_struc) # radians
        assert(all_angles_struc.shape[0] == n_ref_states_struc)
        assert(len(all_angles_struc.shape) == 1)

        angle_running_avg = onp.cumsum(all_angles_struc) / onp.arange(1, n_ref_states_struc+1)
        plt.plot(angle_running_avg)
        plt.xlabel("Samples")
        plt.ylabel("Angle (rad)")
        plt.savefig(struc_dir / f"angle_running_avg_rad.png")
        plt.close()

        plt.plot(180.0 * angle_running_avg / onp.pi)
        plt.xlabel("Samples")
        plt.ylabel("Angle (deg)")
        plt.savefig(struc_dir / f"angle_running_avg_deg.png")
        plt.close()

        plt.plot(2*onp.pi / angle_running_avg)
        plt.xlabel("Samples")
        plt.ylabel("Pitch (# bp / turn)")
        plt.savefig(struc_dir / f"pitch_running_avg.png")
        plt.close()

        mean_angle_struc = all_angles_struc.mean()


        ## Compute the inclination

        def compute_body_inclination(body):
            return geom_inclination_alt.get_mean_inclination(body, first_base, last_base)
        all_inclinations_struc = vmap(compute_body_inclination)(traj_states_struc) # degrees
        assert(all_inclinations_struc.shape[0] == n_ref_states_struc)
        assert(len(all_inclinations_struc.shape) == 1)

        inclinations_running_avg = onp.cumsum(all_inclinations_struc) / onp.arange(1, n_ref_states_struc+1)
        plt.plot(inclinations_running_avg)
        plt.xlabel("Samples")
        plt.ylabel("Inclination (deg)")
        plt.savefig(struc_dir / f"inclination_running_avg.png")
        plt.close()

        mean_inclination_struc = all_inclinations_struc.mean()



        ## Record the loss
        with open(struc_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Max energy diff: {onp.max(energy_diffs)}\n")
            f.write(f"Min energy diff: {onp.min(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

            f.write(f"\nMean rise (oxDNA units): {mean_rise_struc}\n")
            f.write(f"\nMean rise (nm): {mean_rise_struc * utils.nm_per_oxrna_length}\n")
            f.write(f"Mean angle (rad): {mean_angle_struc}\n")
            f.write(f"Mean angle (deg): {180.0 * mean_angle_struc / onp.pi}\n")
            f.write(f"Pitch (# bp / turn): {2*onp.pi / mean_angle_struc}\n")
            f.write(f"Mean inclination (deg): {mean_inclination_struc}\n")



        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        if not no_archive:
            zip_file(str(lp_dir / "output.dat"), str(lp_dir / "output.dat.zip"))
            os.remove(str(lp_dir / "output.dat"))

        return traj_states, calc_energies, unweighted_corr_curves, all_rises, iter_dir, \
            traj_states_struc, all_rises_struc, all_angles_struc, all_inclinations_struc, calc_energies_struc


    # Construct the loss function
    @jit
    def loss_fn(
        params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises_lp,
        ref_states_struc, ref_energies_struc, unweighted_rises, unweighted_angles, unweighted_inclinations
    ):
        em = model.EnergyModel(
            displacement_fn,
            params["seq_avg"],
            t_kelvin=t_kelvin,
            salt_conc=salt_conc
        )

        # Lp analysis
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_lp,
            bonded_nbrs=top_info_lp.bonded_nbrs,
            unbonded_nbrs=top_info_lp.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

        _, new_energies = scan(energy_scan_fn, None, ref_states)

        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        expected_corr_curve = jnp.sum(weighted_corr_curves, axis=0)

        expected_rise_lp = jnp.dot(unweighted_rises_lp, weights)

        expected_lp, expected_offset = persistence_length.persistence_length_fit(
            expected_corr_curve[:truncation],
            expected_rise_lp
        )
        expected_lp = expected_lp * utils.nm_per_oxrna_length

        n_eff_lp = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        if standardize:
            rmse_lp = abs_relative_diff(target_lp, expected_lp)
        else:
            mse_lp = (expected_lp - target_lp)**2
            rmse_lp = jnp.sqrt(mse_lp)


        # Structural analysis
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh_struc,
            bonded_nbrs=top_info_struc.bonded_nbrs,
            unbonded_nbrs=top_info_struc.unbonded_nbrs.T
        )
        energy_fn = jit(energy_fn)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))

        _, new_energies = scan(energy_scan_fn, None, ref_states_struc)

        diffs = new_energies - ref_energies_struc # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        expected_rise = jnp.dot(weights, unweighted_rises)
        expected_rise *= utils.nm_per_oxrna_length

        expected_angle = jnp.dot(weights, unweighted_angles)
        expected_pitch = 2*jnp.pi / expected_angle
        expected_inclination = jnp.dot(weights, unweighted_inclinations)

        if standardize:
            rmse_rise = abs_relative_diff(target_rise, expected_rise)
        else:
            mse_rise = (expected_rise - target_rise)**2
            rmse_rise = jnp.sqrt(mse_rise)

        if standardize:
            rmse_pitch = abs_relative_diff(target_pitch, expected_pitch)
        else:
            mse_pitch = (expected_pitch - target_pitch)**2
            rmse_pitch = jnp.sqrt(mse_pitch)

        if standardize:
            rmse_inclination = abs_relative_diff(target_inclination, expected_inclination)
        else:
            mse_inclination = (expected_inclination - target_inclination)**2
            rmse_inclination = jnp.sqrt(mse_inclination)

        n_eff_struc = jnp.exp(-jnp.sum(weights * jnp.log(weights)))


        # Aggregate
        expected_lp_n_bp = expected_lp / expected_rise
        rmse_total = rmse_lp + rmse_rise + rmse_pitch + rmse_inclination

        return rmse_total, (n_eff_lp, expected_lp, expected_corr_curve, expected_rise_lp, expected_offset, n_eff_struc, expected_rise, expected_pitch, expected_lp_n_bp, expected_inclination, rmse_lp, rmse_rise, rmse_pitch, rmse_inclination)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    seq_avg_params = deepcopy(EMPTY_BASE_PARAMS)
    for opt_key in seq_avg_opt_keys:
        seq_avg_params[opt_key] = deepcopy(DEFAULT_BASE_PARAMS[opt_key])
    params = {"seq_avg": seq_avg_params}

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)


    min_n_eff_lp = int(n_ref_states_lp * min_neff_factor)
    min_n_eff_struc = int(n_ref_states_struc * min_neff_factor)

    all_losses = list()
    all_lps = list()
    all_n_eff_lps = list()
    all_n_eff_strucs = list()
    all_rises = list()
    all_pitches = list()
    all_inclinations = list()

    all_ref_losses = list()
    all_ref_lps = list()
    all_ref_times = list()
    all_ref_rises = list()
    all_ref_pitches = list()
    all_ref_inclinations = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_iter_dir, ref_states_struc, unweighted_rises_struc, unweighted_angles_struc, unweighted_inclinations_struc, ref_energies_struc = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_eff_lp, curr_lp, expected_corr_curve, curr_rise_lp, curr_offset, n_eff_struc, curr_rise, curr_pitch, curr_lp_n_bp, curr_inclination, rmse_lp, rmse_rise, rmse_pitch, rmse_inclination)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_states_struc, ref_energies_struc, unweighted_rises_struc, unweighted_angles_struc, unweighted_inclinations_struc)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_rises.append(curr_rise)
            all_ref_pitches.append(curr_pitch)
            all_ref_inclinations.append(curr_inclination)

        if n_eff_lp < min_n_eff_lp or n_eff_struc < min_n_eff_struc or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff_lp was {n_eff_lp} and n_eff_struc was {n_eff_struc}. Resampling...\n")

            start = time.time()
            ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_iter_dir, ref_states_struc, unweighted_rises_struc, unweighted_angles_struc, unweighted_inclinations_struc, ref_energies_struc = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff_lp, curr_lp, expected_corr_curve, curr_rise_lp, curr_offset, n_eff_struc, curr_rise, curr_pitch, curr_lp_n_bp, curr_inclination, rmse_lp, rmse_rise, rmse_pitch, rmse_inclination)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_states_struc, ref_energies_struc, unweighted_rises_struc, unweighted_angles_struc, unweighted_inclinations_struc)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_rises.append(curr_rise)
            all_ref_pitches.append(curr_pitch)
            all_ref_inclinations.append(curr_inclination)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_lp_path, "a") as f:
            f.write(f"{n_eff_lp}\n")
        with open(neff_struc_path, "a") as f:
            f.write(f"{n_eff_struc}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(rise_from_lp_path, "a") as f:
            f.write(f"{curr_rise_lp}\n")
        with open(rise_path, "a") as f:
            f.write(f"{curr_rise}\n")
        with open(pitch_path, "a") as f:
            f.write(f"{curr_pitch}\n")
        with open(inclination_path, "a") as f:
            f.write(f"{curr_inclination}\n")
        with open(lp_n_bp_path, "a") as f:
            f.write(f"{curr_lp_n_bp}\n")
        with open(rmse_lp_path, "a") as f:
            f.write(f"{rmse_lp}\n")
        with open(rmse_rise_path, "a") as f:
            f.write(f"{rmse_rise}\n")
        with open(rmse_pitch_path, "a") as f:
            f.write(f"{rmse_pitch}\n")
        with open(rmse_inclination_path, "a") as f:
            f.write(f"{rmse_inclination}\n")


        grads_str = f"\nIteration {i}:"
        for k, v in grads.items():
            grads_str += f"\n- {k}"
            for vk, vv in v.items():
                grads_str += f"\n\t- {vk}: {vv}"
        with open(grads_path, "a") as f:
            f.write(grads_str)

        iter_params_str = f"\nIteration {i}:"
        for k, v in params.items():
            iter_params_str += f"\n- {k}"
            for vk, vv in v.items():
                iter_params_str += f"\n\t- {vk}: {vv}"
        with open(iter_params_path, "a") as f:
            f.write(iter_params_str)

        all_losses.append(loss)
        all_n_eff_lps.append(n_eff_lp)
        all_n_eff_strucs.append(n_eff_struc)
        all_lps.append(curr_lp)
        all_rises.append(curr_rise)
        all_pitches.append(curr_pitch)
        all_inclinations.append(curr_inclination)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0 and i:
            plt.plot(expected_corr_curve)
            plt.xlabel("Nuc. Index")
            plt.ylabel("Correlation")
            plt.savefig(img_dir / f"corr_iter{i}.png")
            plt.clf()

            log_corr_fn = lambda n: -n * curr_rise_lp / (curr_lp) + curr_offset
            plt.plot(jnp.log(expected_corr_curve))
            plt.plot(log_corr_fn(jnp.arange(expected_corr_curve.shape[0])), linestyle='--')
            plt.xlabel("Nuc. Index")
            plt.ylabel("Log-Correlation")
            plt.savefig(img_dir / f"log_corr_iter{i}.png")
            plt.clf()

            plt.plot(expected_corr_curve[:truncation])
            plt.xlabel("Nuc. Index")
            plt.ylabel("Correlation")
            plt.savefig(img_dir / f"corr_truncated_iter{i}.png")
            plt.clf()

            plt.plot(jnp.log(expected_corr_curve[:truncation]))
            plt.plot(log_corr_fn(jnp.arange(expected_corr_curve[:truncation].shape[0])), linestyle='--')
            plt.xlabel("Nuc. Index")
            plt.ylabel("Log-Correlation")
            plt.savefig(img_dir / f"log_corr_truncated_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_losses, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points", color="blue")
            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()

            plt.plot(onp.arange(i+1), all_lps, linestyle="--", color="blue")
            plt.scatter(all_ref_times, all_ref_lps, marker='o', label="Resample points", color="blue")
            plt.axhline(y=target_lp, linestyle='--', label="Target Lp", color='red')
            plt.legend()
            plt.ylabel("Expected Lp (nm)")
            plt.xlabel("Iteration")
            plt.savefig(img_dir / f"lps_iter{i}.png")
            plt.clf()

        if i % save_obj_every == 0 and i:
            onp.save(obj_dir / f"ref_iters_i{i}.npy", onp.array(all_ref_times), allow_pickle=False)
            onp.save(obj_dir / f"ref_lps_i{i}.npy", onp.array(all_ref_lps), allow_pickle=False)
            onp.save(obj_dir / f"ref_rises_i{i}.npy", onp.array(all_ref_rises), allow_pickle=False)
            onp.save(obj_dir / f"ref_pitches_i{i}.npy", onp.array(all_ref_pitches), allow_pickle=False)
            onp.save(obj_dir / f"ref_inclinations_i{i}.npy", onp.array(all_ref_inclinations), allow_pickle=False)

            onp.save(obj_dir / f"lps_i{i}.npy", onp.array(all_lps), allow_pickle=False)
            onp.save(obj_dir / f"rises_i{i}.npy", onp.array(all_rises), allow_pickle=False)
            onp.save(obj_dir / f"pitches_i{i}.npy", onp.array(all_pitches), allow_pickle=False)
            onp.save(obj_dir / f"inclinations_i{i}.npy", onp.array(all_inclinations), allow_pickle=False)


    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_lps.npy", onp.array(all_ref_lps), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_rises.npy", onp.array(all_ref_rises), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_pitches.npy", onp.array(all_ref_pitches), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_inclinations.npy", onp.array(all_ref_inclinations), allow_pickle=False)

    onp.save(obj_dir / f"fin_lps.npy", onp.array(all_lps), allow_pickle=False)
    onp.save(obj_dir / f"fin_rises.npy", onp.array(all_rises), allow_pickle=False)
    onp.save(obj_dir / f"fin_pitches.npy", onp.array(all_pitches), allow_pickle=False)
    onp.save(obj_dir / f"fin_inclinations.npy", onp.array(all_inclinations), allow_pickle=False)



def get_parser():

    parser = argparse.ArgumentParser(description="Optimize persistence length of RNA via standalone oxDNA package")

    # Generic
    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")

    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--salt-concentration', type=float, default=1.0, help="Salt concentration in molar (M)")

    parser.add_argument(
        '--seq-avg-opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "cross_stacking"],
        help='Parameter keys to optimize'
    )

    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--no-archive', action='store_true')

    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--save-obj-every', type=int, default=50,
                        help="Frequency of saving numpy files")

    # Persistence-length-specific
    parser.add_argument('--offset', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--truncation', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--n-sims-lp', type=int, default=1,
                        help="Number of individual simulations for Lp")
    parser.add_argument('--n-steps-per-sim-lp', type=int, default=100000,
                        help="Number of steps per simulation for Lp")
    parser.add_argument('--n-eq-steps-lp', type=int, default=0,
                        help="Number of equilibration steps for Lp")
    parser.add_argument('--sample-every-lp', type=int, default=1000,
                        help="Frequency of sampling reference states for Lp.")
    parser.add_argument('--target-lp', type=float, default=48.0,
                        help="Target persistence length in nanometers")

    # Structure-specific
    parser.add_argument('--n-sims-struc', type=int, default=1,
                        help="Number of individual simulations for structure")
    parser.add_argument('--n-steps-per-sim-struc', type=int, default=100000,
                        help="Number of steps per simulation for structure")
    parser.add_argument('--n-eq-steps-struc', type=int, default=0,
                        help="Number of equilibration steps for structure")
    parser.add_argument('--sample-every-struc', type=int, default=1000,
                        help="Frequency of sampling reference states for structure.")
    parser.add_argument('--offset-struc', type=int, default=1,
                        help="Offset for structural calculation")
    parser.add_argument('--target-rise', type=float, default=0.28,
                        help="Target rise in nanometers")
    parser.add_argument('--target-pitch', type=float, default=10.9,
                        help="Target pitch in # bp / turn")
    parser.add_argument('--target-inclination', type=float, default=15.5,
                        help="Target inclination in deg.")

    parser.add_argument('--no-standardize', action='store_true')


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
