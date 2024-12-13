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
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)




@jit
def get_site_positions(body, is_rna=True):
    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
    base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
    cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

    if is_rna:
        # RNA values
        com_to_backbone_x = -0.4
        com_to_backbone_y = 0.2
        com_to_stacking = 0.34
        com_to_hb = 0.4
        back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals
        stack_sites = body.center + com_to_stacking * back_base_vectors
        base_sites = body.center + com_to_hb * back_base_vectors
    else:
        # In code (like DNA1)
        com_to_backbone = -0.4
        com_to_stacking = 0.34
        com_to_hb = 0.4
        back_sites = body.center + com_to_backbone*back_base_vectors
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
    n_bps = last_base - first_base

    def single_rise(bp_idx):
        i = first_base + bp_idx

        midp = (stack_sites[i] + stack_sites[n-i-1]) / 2
        midp_ip1 = (stack_sites[i+1] + stack_sites[n-(i+1)-1]) / 2

        midp_proj = jnp.dot(plane_vector, midp)
        midp_ip1_proj = jnp.dot(plane_vector, midp_ip1)

        rise = midp_ip1_proj - midp_proj
        return rise
    rises = vmap(single_rise)(jnp.arange(n_bps))

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

def run(args):
    # Load parameters
    n_threads = args['n_threads']
    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    run_name = args['run_name']
    oxdna_path = Path(args['oxdna_path'])
    oxdna_exec_path = oxdna_path / "build/bin/oxDNA"
    offset = args['offset']
    truncation = args['truncation']
    n_iters = args['n_iters']
    lr = args['lr']
    target_lp = args['target_lp']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    salt_conc = args['salt_concentration']
    no_delete = args['no_delete']
    no_archive = args['no_archive']
    plot_every = args['plot_every']
    save_obj_every = args['save_obj_every']
    t_kelvin = utils.DEFAULT_TEMP

    seq_avg_opt_keys = args['seq_avg_opt_keys']
    opt_seq_dep_stacking = args['opt_seq_dep_stacking']

    ss_hb_weights, ss_stack_weights, ss_cross_weights = read_seq_specific(DEFAULT_BASE_PARAMS)


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
    neff_path = log_dir / "neff.txt"
    lp_path = log_dir / "lp.txt"
    rise_path = log_dir / "rise.txt"
    lp_n_bp_path = log_dir / "lp_n_bp.txt"
    resample_log_path = log_dir / "resample_log.txt"
    iter_params_path = log_dir / "iter_params.txt"

    params_str = ""
    params_str += f"n_ref_states: {n_ref_states}\n"
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    # Load the system
    sys_basedir = Path("data/templates/ds-142bp-rna")
    input_template_path = sys_basedir / "input"

    top_path = sys_basedir / "sys.top"
    # top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)

    n_bp = int(seq_oh.shape[0] // 2)

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=False

    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT


    # Do the simulation
    def get_ref_states(params, i, seed, prev_basedir):

        random.seed(seed)

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        sim_start = time.time()
        procs = list()

        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            repeat_dir.mkdir(parents=False, exist_ok=False)

            shutil.copy(top_path, repeat_dir / "sys.top")

            seq_dep_path = repeat_dir / "rna_sequence_dependent_parameters.txt"
            if "stacking" in params["seq_dep"]:
                curr_stack_weights = params["seq_dep"]["stacking"]
            else:
                curr_stack_weights = ss_stack_weights
            oxrna_utils.write_seq_specific(seq_dep_path, params["seq_avg"], ss_hb_weights, curr_stack_weights, ss_cross_weights)

            if prev_basedir is None:
                init_conf_info = deepcopy(centered_conf_info)
            else:
                prev_repeat_dir = prev_basedir / f"r{r}"
                prev_lastconf_path = prev_repeat_dir / "last_conf.dat"
                prev_lastconf_info = trajectory.TrajectoryInfo(
                    top_info,
                    read_from_file=True, traj_path=prev_lastconf_path,
                    # reverse_direction=True
                    reverse_direction=False
                )
                init_conf_info = center_configuration.center_conf(
                    top_info, prev_lastconf_info)

            init_conf_info.traj_df.t = onp.full(seq_oh.shape[0], r*n_steps_per_sim)
            init_conf_info.write(repeat_dir / "init.conf", reverse=False, write_topology=False)

            external_model_fpath = repeat_dir / "external_model.txt"
            oxrna_utils.write_external_model(params["seq_avg"], t_kelvin, salt_conc, external_model_fpath)

            oxdna_utils.rewrite_input_file(
                input_template_path, repeat_dir,
                temp=f"{t_kelvin}K", steps=n_steps_per_sim,
                init_conf_path=str(repeat_dir / "init.conf"),
                top_path=str(repeat_dir / "sys.top"),
                save_interval=sample_every, seed=random.randrange(1000),
                equilibration_steps=n_eq_steps, dt=dt,
                no_stdout_energy=0,
                log_file=str(repeat_dir / "sim.log"),
                external_model=str(external_model_fpath),
                seq_dep_file=str(seq_dep_path),
                seq_dep_file_RNA=str(seq_dep_path),
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

        combine_cmd = "cat "
        for r in range(n_sims):
            repeat_dir = iter_dir / f"r{r}"
            combine_cmd += f"{repeat_dir}/output.dat "
        combine_cmd += f"> {iter_dir}/output.dat"
        combine_proc = subprocess.run(combine_cmd, shell=True)
        if combine_proc.returncode != 0:
            raise RuntimeError(f"Combining trajectories failed with error code: {combine_proc.returncode}")

        if not no_delete:
            files_to_remove = ["output.dat"]
            for r in range(n_sims):
                repeat_dir = iter_dir / f"r{r}"
                for f_stem in files_to_remove:
                    file_to_rem = repeat_dir / f_stem
                    file_to_rem.unlink()


        # Analyze

        ## Load states from oxDNA simulation
        load_start = time.time()
        strand_length = int(seq_oh.shape[0] // 2)
        traj_ = jdt.from_file(
            iter_dir / "output.dat",
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
        energy_dfs = [pd.read_csv(iter_dir / f"r{r}" / "energy.dat", names=energy_df_columns,
                                  delim_whitespace=True)[1:] for r in range(n_sims)]
        energy_df = pd.concat(energy_dfs, ignore_index=True)

        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            # ss_stack_weights=ss_stack_weights)
            ss_stack_weights=curr_stack_weights)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # Check energies
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")


        gt_energies = energy_df.potential_energy.to_numpy() * seq_oh.shape[0]

        energy_diffs = list()
        for i, (calc, gt) in enumerate(zip(calc_energies, gt_energies)):
            diff = onp.abs(calc - gt)
            energy_diffs.append(diff)

        ## Plot the energies
        sns.distplot(calc_energies, label="Calculated", color="red")
        sns.distplot(gt_energies, label="Reference", color="green")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()

        ## Plot the energy differences
        sns.histplot(energy_diffs)
        plt.savefig(iter_dir / f"energy_diffs.png")
        plt.clf()

        # Compute the persistence lengths
        analyze_start = time.time()

        down_neigh = 1
        up_neigh = 1
        max_dist = n_bp - offset * 2
        all_rises = vmap(get_rises, (0, None, None))(traj_states, offset, n_bp-offset-2)
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
            all_inter_lps_truncated.append(inter_mean_Lp_truncated * utils.nm_per_oxdna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, avg_rise)
            all_inter_lps.append(inter_mean_Lp * utils.nm_per_oxdna_length)

            # Full version
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves_full[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], avg_rise)
            all_inter_lps_truncated_full.append(inter_mean_Lp_truncated * utils.nm_per_oxdna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, avg_rise)
            all_inter_lps_full.append(inter_mean_Lp * utils.nm_per_oxdna_length)

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(iter_dir / "running_avg.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(iter_dir / "running_avg_truncated.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_full)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average")
        plt.savefig(iter_dir / "running_avg_full.png")
        plt.clf()

        plt.plot(list(range(0, n_curves, compute_every)), all_inter_lps_truncated_full)
        plt.ylabel("Lp (nm)")
        plt.xlabel("# Samples")
        plt.title("Lp running average, truncated")
        plt.savefig(iter_dir / "running_avg_truncated_full.png")
        plt.clf()

        plt.plot(mean_corr_curve)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(iter_dir / "full_corr_curve.png")
        plt.clf()

        plt.plot(mean_corr_curve_full)
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.legend()
        plt.title("Full Correlation Curve")
        plt.savefig(iter_dir / "full_corr_curve_full.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "full_log_corr_curve.png")
        plt.clf()

        plt.plot(jnp.log(mean_corr_curve_full))
        plt.title("Full Log-Correlation Curve")
        plt.axvline(x=truncation, linestyle='--', label="Truncation")
        plt.xlabel("Nuc. Index")
        plt.ylabel("Log-Correlation")
        plt.legend()
        plt.savefig(iter_dir / "full_log_corr_curve_full.png")
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
        plt.savefig(iter_dir / "log_corr_curve.png")
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
        plt.savefig(iter_dir / "log_corr_curve_full.png")
        plt.clf()


        # Record the loss
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Mean energy diff: {onp.mean(energy_diffs)}\n")
            f.write(f"Max energy diff: {onp.max(energy_diffs)}\n")
            f.write(f"Min energy diff: {onp.min(energy_diffs)}\n")
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Ref. energy var.: {onp.var(gt_energies)}\n")

            f.write(f"\nMean Lp truncated (oxDNA units): {mean_Lp_truncated}\n")
            f.write(f"Mean Rise (oxDNA units): {avg_rise}\n")
            f.write(f"Mean Lp truncated (num bp via oxDNA units): {mean_Lp_truncated / avg_rise}\n")

            f.write(f"\nMean Lp truncated (nm): {mean_Lp_truncated * utils.nm_per_oxdna_length}\n")


            f.write(f"\nMean Lp truncated, full (oxDNA units): {mean_Lp_truncated_full}\n")
            f.write(f"Mean Lp truncated, full (num bp via oxDNA units): {mean_Lp_truncated_full / avg_rise}\n")

            f.write(f"\nMean Lp truncated, full (nm): {mean_Lp_truncated_full * utils.nm_per_oxdna_length}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        if not no_archive:
            zip_file(str(iter_dir / "output.dat"), str(iter_dir / "output.dat.zip"))
            os.remove(str(iter_dir / "output.dat"))

        return traj_states, calc_energies, unweighted_corr_curves, all_rises, iter_dir


    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises):
        if "stacking" in params["seq_dep"]:
            curr_stack_weights = params["seq_dep"]["stacking"]
        else:
            curr_stack_weights = ss_stack_weights
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            # ss_stack_weights=ss_stack_weights)
            ss_stack_weights=curr_stack_weights)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)


        # Compute the weights
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))
        _, new_energies = scan(energy_scan_fn, None, ref_states)

        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        expected_corr_curve = jnp.sum(weighted_corr_curves, axis=0)

        expected_rise = jnp.dot(unweighted_rises, weights)

        expected_lp, expected_offset = persistence_length.persistence_length_fit(
            expected_corr_curve[:truncation],
            expected_rise)
        expected_lp = expected_lp * utils.nm_per_oxdna_length

        expected_rise *= utils.nm_per_oxdna_length
        expected_lp_n_bp = expected_lp / expected_rise

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        mse = (expected_lp - target_lp)**2
        rmse = jnp.sqrt(mse)

        return rmse, (n_eff, expected_lp, expected_corr_curve, expected_rise*utils.nm_per_oxdna_length, expected_lp_n_bp, expected_offset)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    seq_avg_params = deepcopy(EMPTY_BASE_PARAMS)
    for opt_key in seq_avg_opt_keys:
        seq_avg_params[opt_key] = deepcopy(DEFAULT_BASE_PARAMS[opt_key])
    params = {"seq_avg": seq_avg_params, "seq_dep": dict()}
    if opt_seq_dep_stacking:
        params["seq_dep"]["stacking"] = jnp.array(ss_stack_weights)

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)


    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_lps = list()
    all_n_effs = list()
    all_rises = list()

    all_ref_losses = list()
    all_ref_lps = list()
    all_ref_times = list()
    all_ref_rises = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    prev_ref_basedir = None
    ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_iter_dir = get_ref_states(params, i=0, seed=0, prev_basedir=prev_ref_basedir)
    prev_ref_basedir = deepcopy(ref_iter_dir)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (loss, (n_eff, curr_lp, expected_corr_curve, curr_rise, curr_lp_n_bp, curr_offset)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises)
        num_resample_iters += 1

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_rises.append(curr_rise)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            start = time.time()
            ref_states, ref_energies, unweighted_corr_curves, unweighted_rises, ref_iter_dir = get_ref_states(params, i=i, seed=i, prev_basedir=prev_ref_basedir)
            end = time.time()
            prev_ref_basedir = deepcopy(ref_iter_dir)
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (loss, (n_eff, curr_lp, expected_corr_curve, curr_rise, curr_lp_n_bp, curr_offset)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_rises)

            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_lps.append(curr_lp)
            all_ref_rises.append(curr_rise)

        iter_end = time.time()

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(rise_path, "a") as f:
            f.write(f"{curr_rise}\n")
        with open(lp_n_bp_path, "a") as f:
            f.write(f"{curr_lp_n_bp}\n")


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
        all_n_effs.append(n_eff)
        all_lps.append(curr_lp)
        all_rises.append(curr_rise)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0 and i:
            plt.plot(expected_corr_curve)
            plt.xlabel("Nuc. Index")
            plt.ylabel("Correlation")
            plt.savefig(img_dir / f"corr_iter{i}.png")
            plt.clf()

            log_corr_fn = lambda n: -n * curr_rise_avg / (curr_lp) + curr_offset
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
            onp.save(obj_dir / f"lps_i{i}.npy", onp.array(all_lps), allow_pickle=False)
            onp.save(obj_dir / f"ref_rises_i{i}.npy", onp.array(all_ref_rises), allow_pickle=False)
            onp.save(obj_dir / f"l0s_i{i}.npy", onp.array(all_l0s), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_iters.npy", onp.array(all_ref_times), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_lps.npy", onp.array(all_ref_lps), allow_pickle=False)
    onp.save(obj_dir / f"fin_lps.npy", onp.array(all_lps), allow_pickle=False)
    onp.save(obj_dir / f"fin_ref_rises.npy", onp.array(all_ref_rises), allow_pickle=False)
    onp.save(obj_dir / f"fin_l0s.npy", onp.array(all_l0s), allow_pickle=False)

def get_parser():

    parser = argparse.ArgumentParser(description="Optimize persistence length of RNA via standalone oxDNA package")

    parser.add_argument('--n-threads', type=int, default=4,
                        help="Number of threads for oxDNA compilation")
    parser.add_argument('--offset', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")
    parser.add_argument('--truncation', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--oxdna-path', type=str,
                        default="/home/ryan/Documents/Harvard/research/brenner/oxdna-bin/oxDNA/",
                        help='oxDNA base directory')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--target-lp', type=float, default=40.0,
                        help="Target persistence length in nanometers")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--salt-concentration', type=float, default=1.0, help="Salt concentration in molar (M)")

    parser.add_argument(
        '--seq-avg-opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "cross_stacking"],
        help='Parameter keys to optimize'
    )
    parser.add_argument('--opt-seq-dep-stacking', action='store_true')

    parser.add_argument('--no-delete', action='store_true')
    parser.add_argument('--no-archive', action='store_true')

    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--save-obj-every', type=int, default=50,
                        help="Frequency of saving numpy files")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
