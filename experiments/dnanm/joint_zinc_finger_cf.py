import pdb
from pathlib import Path
from copy import deepcopy
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import subprocess
import pandas as pd
import shutil
import seaborn as sns
import argparse
import functools
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)
import optax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, lax, tree_util, random, flatten_util
from jax_md import space, rigid_body, simulate
import orbax.checkpoint
from flax.training import orbax_utils

from jax_dna.loss import rmse
from jax_dna.common import utils, topology_protein_na, trajectory, center_configuration, checkpoint
from jax_dna.dnanm import model_nbrs
from jax_dna.dna2 import model as model_dna2



checkpoint_every = 1
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(
        checkpoint.checkpoint_scan,
        checkpoint_every=checkpoint_every
    )

def normalize(g):
    return g / jnp.linalg.norm(g)


def run(args):
    # Load parameters
    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_ref_states_per_sim = n_steps_per_sim // sample_every
    n_ref_states = n_ref_states_per_sim * n_sims
    assert(n_ref_states >= checkpoint_every)
    run_name = args['run_name']
    n_iters = args['n_iters']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    max_approx_iters = args['max_approx_iters']
    use_nbrs = args['use_nbrs']
    save_checkpoints = args['save_checkpoints']

    save_obj_every = args['save_obj_every']

    seq_avg = True
    if seq_avg:
        ss_hb_weights = utils.HB_WEIGHTS_SA
        ss_stack_weights = utils.STACK_WEIGHTS_SA
    else:

        ss_path = "data/seq-specific/seq_oxdna2.txt"
        ss_hb_weights, ss_stack_weights = read_ss_oxdna(
            ss_path,
            model_dna2.default_base_params_seq_dep['hydrogen_bonding']['eps_hb'],
            model_dna2.default_base_params_seq_dep['stacking']['eps_stack_base'],
            model_dna2.default_base_params_seq_dep['stacking']['eps_stack_kt_coeff'],
            enforce_symmetry=False,
            t_kelvin=t_kelvin
        )

    orbax_ckpt_path = args['orbax_ckpt_path']
    ckpt_freq = args['ckpt_freq']

    t_kelvin = args['t_kelvin']
    salt_conc = args['salt_conc']


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

    pdb_ids = ["1A1L", "1AAY", "1ZAA"]

    times_path = log_dir / "times.txt"
    params_per_iter_path = log_dir / "params_per_iter.txt"
    pct_change_path = log_dir / "pct_change.txt"
    grads_path = log_dir / "grads.txt"
    neff_path = log_dir / "neff.txt"
    rmse_path = log_dir / "rmse.txt"
    id_rmse_paths = {pdb_id: log_dir / f"{pdb_id}_rmse.txt" for pdb_id in pdb_ids}
    resample_log_path = log_dir / "resample_log.txt"

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    nvidia_smi_path = run_dir / "nvidia-smi.txt"
    os.system(f"nvidia-smi >> {nvidia_smi_path}")

    # Load the system
    displacement_fn, shift_fn = space.free()
    pdb_info = {}
    for pdb_id in pdb_ids:

        pdb_info[pdb_id] = {}

        sys_basedir = Path("data/templates/") / pdb_id

        id_top_path = sys_basedir / "complex.top"
        id_par_path = sys_basedir / "protein.par"
        id_top_info = topology_protein_na.ProteinNucAcidTopology(id_top_path, id_par_path)
        pdb_info[pdb_id]["topology"] = id_top_info
        pdb_info[pdb_id]["top_path"] = id_top_path

        id_target_path = sys_basedir / "complex.conf"
        target_info = trajectory.TrajectoryInfo(
            id_top_info,
            read_from_file=True, traj_path=id_target_path,
            reverse_direction=False
        )
        id_target_state = target_info.get_states()[0]
        pdb_info[pdb_id]["target"] = id_target_state

        id_conf_path = sys_basedir / "relaxed.dat"
        conf_info = trajectory.TrajectoryInfo(
            id_top_info,
            read_from_file=True, traj_path=id_conf_path,
            reverse_direction=False
        )
        id_centered_conf_info = center_configuration.center_conf(id_top_info, conf_info)
        pdb_info[pdb_id]["init_body"] = id_centered_conf_info.get_states()[0]
        id_box_size = conf_info.box_size
        pdb_info[pdb_id]["box_size"] = id_box_size

        half_charged_ends = True
        if half_charged_ends:
            id_is_end = jnp.array(id_top_info.is_end)
        else:
            id_is_end = None
        pdb_info[pdb_id]["is_end"] = id_is_end

        id_default_neighbors = None
        if use_nbrs:
            r_cutoff = 10.0
            dr_threshold = 0.2
            neighbor_fn = id_top_info.get_neighbor_list_fn(
                displacement_fn, id_box_size, r_cutoff, dr_threshold
            )
            # Note that we only allocate once
            neighbors = neighbor_fn.allocate(id_target_state.center) # We use the COMs.
            id_default_neighbors = deepcopy(neighbors)
        pdb_info[pdb_id]["default_neighbors"] = id_default_neighbors

    dt = args['dt']
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64)
    )
    mass = rigid_body.RigidBody(
        center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
        orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64)
    )


    # Initialize parameters
    init_dna2_params = deepcopy(model_dna2.EMPTY_BASE_PARAMS)
    init_dna_base_protein_sigma, init_dna_base_protein_epsilon, init_dna_base_protein_alpha, init_dna_back_protein_sigma, init_dna_back_protein_epsilon, init_dna_back_protein_alpha = model_nbrs.get_init_morse_tables()

    init_params_dict = {
        "dna2": init_dna2_params,
        "dnanm": {
            "base_sigma": init_dna_base_protein_sigma,
            "base_epsilon": init_dna_base_protein_epsilon,
            "base_alpha": init_dna_base_protein_alpha,

            "back_sigma": init_dna_back_protein_sigma,
            "back_epsilon": init_dna_back_protein_epsilon,
            "back_alpha": init_dna_back_protein_alpha,

        }
    }
    init_params_flat, ravel_fn = flatten_util.ravel_pytree(init_params_dict)


    def sim_fn(params, body, key, pdb_id):

        default_neighbors = pdb_info[pdb_id]["default_neighbors"]
        top_info = pdb_info[pdb_id]["topology"]
        is_end = pdb_info[pdb_id]["is_end"]


        dna2_params = params["dna2"]

        dnanm_params = params["dnanm"]
        dna_base_protein_sigma = dnanm_params["base_sigma"]
        dna_base_protein_epsilon = dnanm_params["base_epsilon"]
        dna_base_protein_alpha = dnanm_params["base_alpha"]
        dna_back_protein_sigma = dnanm_params["back_sigma"]
        dna_back_protein_epsilon = dnanm_params["back_epsilon"]
        dna_back_protein_alpha = dnanm_params["back_alpha"]

        if use_nbrs:
            neighbors_idx = default_neighbors.idx
        else:
            neighbors_idx = top_info.unbonded_nbrs.T

        em = model_nbrs.EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq=jnp.array(top_info.nt_seq_idx),
            # ANM
            network=jnp.array(top_info.network),
            eq_distances=jnp.array(top_info.eq_distances),
            spring_constants=jnp.array(top_info.spring_constants),
            # DNA2
            override_base_params=dna2_params,
            t_kelvin=t_kelvin,
            salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=ss_stack_weights,
            seq_avg=seq_avg,
            # DNA/Protein interaction
            include_dna_protein_morse=True,
            dna_base_protein_sigma=dna_base_protein_sigma,
            dna_base_protein_epsilon=dna_base_protein_epsilon,
            dna_base_protein_alpha=dna_base_protein_alpha,
            dna_back_protein_sigma=dna_back_protein_sigma,
            dna_back_protein_epsilon=dna_back_protein_epsilon,
            dna_back_protein_alpha=dna_back_protein_alpha,
        )

        energy_fn = lambda body, neighbors_idx: em.energy_fn(
            body,
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=neighbors_idx.T,
            is_end=is_end
        )
        energy_fn = jit(energy_fn)

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, neighbors_idx=neighbors_idx)

        @jit
        def fori_step_fn(t, carry):
            state, neighbors = carry
            if use_nbrs:
                neighbors = neighbors.update(state.position.center)
                neighbors_idx = neighbors.idx
            else:
                neighbors_idx = top_info.unbonded_nbrs.T
            state = step_fn(state, neighbors_idx=neighbors_idx)
            return (state, neighbors)

        @jit
        def scan_fn(carry, step):
            (state, neighbors) = lax.fori_loop(0, sample_every, fori_step_fn, carry)
            return (state, neighbors), state.position

        (eq_state, eq_neighbors) = lax.fori_loop(0, n_eq_steps, fori_step_fn, (init_state, default_neighbors))
        (fin_state, _), traj = scan(scan_fn, (eq_state, eq_neighbors), jnp.arange(n_ref_states_per_sim))

        return traj

    def get_ref_states(params_flat, i, iter_key, body, pdb_id):

        params = ravel_fn(params_flat)

        iter_dir = ref_traj_dir / f"iter{i}"
        if not iter_dir.exists():
            iter_dir.mkdir(parents=False, exist_ok=False)

        pdb_id_dir = iter_dir / pdb_id
        pdb_id_dir.mkdir(parents=False, exist_ok=False)

        top_info = pdb_info[pdb_id]["topology"]
        top_path = pdb_info[pdb_id]["top_path"]
        box_size = pdb_info[pdb_id]["box_size"]
        is_end = pdb_info[pdb_id]["is_end"]
        target_state = pdb_info[pdb_id]["target"]

        iter_key, sim_key = random.split(iter_key)
        sim_keys = random.split(sim_key, n_sims)
        sim_start = time.time()
        # all_batch_ref_states = vmap(sim_fn, (None, None, 0))(params, init_body, sim_keys)
        all_batch_ref_states = vmap(lambda params, body, key: sim_fn(params, body, key, pdb_id), (None, None, 0))(params, body, sim_keys)
        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulating took {sim_end - sim_start} seconds\n")

        combined_center = all_batch_ref_states.center.reshape(-1, top_info.n, 3)
        combined_quat_vec = all_batch_ref_states.orientation.vec.reshape(-1, top_info.n, 4)

        traj_states = rigid_body.RigidBody(
            center=combined_center,
            orientation=rigid_body.Quaternion(combined_quat_vec)
        )

        # n_traj_states = len(ref_states)
        n_traj_states = traj_states.center.shape[0]


        write_traj = True
        if write_traj:

            shutil.copy(top_path, pdb_id_dir / "sys.top")

            traj_info = trajectory.TrajectoryInfo(
                top_info,
                read_from_states=True,
                states=traj_states,
                box_size=box_size
            )
            traj_info.write(pdb_id_dir / "sampled_states.dat", reverse=False)


        ## Generate an energy function

        dna2_params = params["dna2"]

        dnanm_params = params["dnanm"]
        dna_base_protein_sigma = dnanm_params["base_sigma"]
        dna_base_protein_epsilon = dnanm_params["base_epsilon"]
        dna_base_protein_alpha = dnanm_params["base_alpha"]
        dna_back_protein_sigma = dnanm_params["back_sigma"]
        dna_back_protein_epsilon = dnanm_params["back_epsilon"]
        dna_back_protein_alpha = dnanm_params["back_alpha"]

        em = model_nbrs.EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq=jnp.array(top_info.nt_seq_idx),
            # ANM
            network=jnp.array(top_info.network),
            eq_distances=jnp.array(top_info.eq_distances),
            spring_constants=jnp.array(top_info.spring_constants),
            # DNA2
            override_base_params=dna2_params,
            t_kelvin=t_kelvin,
            salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=ss_stack_weights,
            seq_avg=seq_avg,
            # DNA/Protein interaction
            include_dna_protein_morse=True,
            dna_base_protein_sigma=dna_base_protein_sigma,
            dna_base_protein_epsilon=dna_base_protein_epsilon,
            dna_base_protein_alpha=dna_base_protein_alpha,
            dna_back_protein_sigma=dna_back_protein_sigma,
            dna_back_protein_epsilon=dna_back_protein_epsilon,
            dna_back_protein_alpha=dna_back_protein_alpha,
        )

        energy_fn = lambda body: em.energy_fn(
            body,
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=jnp.array(top_info.unbonded_nbrs),
            is_end=is_end
        )
        energy_fn = jit(energy_fn)

        ## Calculate energies
        calc_start = time.time()
        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, calc_energies = scan(energy_scan_fn, None, traj_states)
        calc_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Calculating energies took {calc_end - calc_start} seconds\n")

        ## Calculate RMSDs
        RMSDs, RMSFs = rmse.compute_rmses(traj_states, target_state, top_info)

        ## Plot logging information
        analyze_start = time.time()

        sns.histplot(RMSDs)
        plt.savefig(pdb_id_dir / f"rmsd_hist.png")
        plt.clf()

        running_avg = onp.cumsum(RMSDs) / onp.arange(1, (n_ref_states)+1)
        plt.plot(running_avg)
        plt.savefig(pdb_id_dir / "running_avg_rmsd.png")
        plt.clf()

        last_half = int((n_ref_states) // 2)
        plt.plot(running_avg[-last_half:])
        plt.savefig(pdb_id_dir / "running_avg_rmsd_second_half.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        plt.legend()
        plt.savefig(pdb_id_dir / f"energies.png")
        plt.clf()


        mean_rmsd = onp.mean(RMSDs)
        with open(pdb_id_dir / "summary.txt", "w+") as f:
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Mean RMSD: {mean_rmsd}\n")
            f.write(f"# Traj. States: {n_traj_states}\n")

        with open(pdb_id_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")

        return traj_states, calc_energies, jnp.array(RMSDs)


    # Construct the loss function

    @functools.partial(jit, static_argnums=(4,))
    def loss_fn_pdb_id(params_flat, all_ref_states, all_ref_energies, all_unweighted_rmses, pdb_id):

        params = ravel_fn(params_flat)

        dna2_params = params["dna2"]

        dnanm_params = params["dnanm"]
        dna_base_protein_sigma = dnanm_params["base_sigma"]
        dna_base_protein_epsilon = dnanm_params["base_epsilon"]
        dna_base_protein_alpha = dnanm_params["base_alpha"]
        dna_back_protein_sigma = dnanm_params["back_sigma"]
        dna_back_protein_epsilon = dnanm_params["back_epsilon"]
        dna_back_protein_alpha = dnanm_params["back_alpha"]


        top_info = pdb_info[pdb_id]["topology"]
        top_path = pdb_info[pdb_id]["top_path"]
        box_size = pdb_info[pdb_id]["box_size"]
        is_end = pdb_info[pdb_id]["is_end"]
        target_state = pdb_info[pdb_id]["target"]

        ref_states = all_ref_states[pdb_id]
        ref_energies = all_ref_energies[pdb_id]
        unweighted_rmses = all_unweighted_rmses[pdb_id]

        em = model_nbrs.EnergyModel(
            displacement_fn,
            is_nt_idx=jnp.array(top_info.is_nt_idx),
            is_protein_idx=jnp.array(top_info.is_protein_idx),
            aa_seq=jnp.array(top_info.aa_seq_idx),
            nt_seq=jnp.array(top_info.nt_seq_idx),
            # ANM
            network=jnp.array(top_info.network),
            eq_distances=jnp.array(top_info.eq_distances),
            spring_constants=jnp.array(top_info.spring_constants),
            # DNA2
            override_base_params=dna2_params,
            t_kelvin=t_kelvin,
            salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=ss_stack_weights,
            seq_avg=seq_avg,
            # DNA/Protein interaction
            include_dna_protein_morse=True,
            dna_base_protein_sigma=dna_base_protein_sigma,
            dna_base_protein_epsilon=dna_base_protein_epsilon,
            dna_base_protein_alpha=dna_base_protein_alpha,
            dna_back_protein_sigma=dna_back_protein_sigma,
            dna_back_protein_epsilon=dna_back_protein_epsilon,
            dna_back_protein_alpha=dna_back_protein_alpha,
        )

        energy_fn = lambda body: em.energy_fn(
            body,
            bonded_nbrs_nt=jnp.array(top_info.bonded_nbrs),
            unbonded_nbrs=jnp.array(top_info.unbonded_nbrs),
            is_end=is_end
        )
        energy_fn = jit(energy_fn)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, new_energies = scan(energy_scan_fn, None, ref_states)
        diffs = new_energies - ref_energies
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        expected_rmse = jnp.dot(weights, unweighted_rmses)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return expected_rmse, (n_eff,)
    grad_fn_pdb_id = value_and_grad(loss_fn_pdb_id, has_aux=True)
    grad_fn_pdb_id = jit(grad_fn_pdb_id, static_argnums=(4,))


    # Initialize parameters
    params_flat = deepcopy(init_params_flat)

    # Setup optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params_flat)


    # Setup checkpointer
    if save_checkpoints:
        ex_params = deepcopy(params_flat)
        ex_ckpt = {"params": ex_params, "optimizer": optimizer, "opt_state": opt_state}
        save_args = orbax_utils.save_args_from_target(ex_ckpt)

        ckpt_dir = run_dir / "ckpt/orbax/managed/"
        ckpt_dir.mkdir(parents=True, exist_ok=False)

        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(str(ckpt_dir.resolve()), orbax_checkpointer, options) # note: checkpoint directory has to be an absoltue path

        ## Load orbax checkpoint if necessary
        if orbax_ckpt_path is not None:
            state_restored = orbax_checkpointer.restore(orbax_ckpt_path, item=ex_ckpt)
            params_flat = state_restored["params"]
            # optimizer = state_restored["optimizer"]
            opt_state = state_restored["opt_state"]



    # Optimize!

    init_params = deepcopy(ravel_fn(params_flat))

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_rmses = list()
    all_n_effs = list()
    all_ref_rmses = list()
    all_ref_times = list()

    pdb_ref_rmses = {pdb_id: list() for pdb_id in pdb_ids}
    pdb_ref_times = {pdb_id: list() for pdb_id in pdb_ids}
    pdb_rmses = {pdb_id: list() for pdb_id in pdb_ids}

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    key = random.PRNGKey(0)
    init_bodies = {pdb_id: pdb_info[pdb_id]["init_body"] for pdb_id in pdb_ids}
    start = time.time()

    all_traj_states = dict()
    all_calc_energies = dict()
    all_rmsds = dict()
    for pdb_id in pdb_ids:
        pdb_ref_states, pdb_ref_energies, pdb_unweighted_rmses = get_ref_states(params_flat, i=0, iter_key=key, body=init_bodies[pdb_id], pdb_id=pdb_id)
        all_traj_states[pdb_id] = pdb_ref_states
        all_calc_energies[pdb_id] = pdb_ref_energies
        all_rmsds[pdb_id] = pdb_unweighted_rmses

    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        all_grads = list()
        all_expected_rmses, all_n_effs = dict(), dict()
        total_rmse = 0.0
        for pdb_id in pdb_ids:
            (expected_rmse, (n_eff,)), pdb_id_grads = grad_fn_pdb_id(params_flat, all_traj_states, all_calc_energies, all_rmsds, pdb_id)
            all_grads.append(pdb_id_grads)
            all_expected_rmses[pdb_id] = expected_rmse
            all_n_effs[pdb_id] = n_eff
            total_rmse += expected_rmse
        all_grads = jnp.array(all_grads)
        mean_rmse = total_rmse / len(pdb_ids)

        num_resample_iters += 1

        if i == 0:
            all_ref_times.append(i)
            all_ref_rmses.append(float(mean_rmse))

            for pdb_id in pdb_ids:
                pdb_ref_rmses[pdb_id].append(float(all_expected_rmses[pdb_id]))
                pdb_ref_times[pdb_id].append(i)

        resampled_atleast_one = False
        did_resample = {pdb_id: False for pdb_id in pdb_ids}
        for pdb_id in pdb_ids:
            should_resample = (all_n_effs[pdb_id] < min_n_eff)
            if should_resample:

                with open(resample_log_path, "a") as f:
                    f.write(f"Iteration {i}\n")
                    f.write(f"- n_eff for ID {pdb_id} was {all_n_effs[pdb_id]}. Resampling...\n")

                key, split = random.split(key)
                pdb_ref_states, pdb_ref_energies, pdb_unweighted_rmses = get_ref_states(
                    params_flat,
                    i=i,
                    iter_key=split,
                    # body=ref_states[-1],
                    body=init_bodies[pdb_id],
                    pdb_id=pdb_id
                )

                all_traj_states[pdb_id] = pdb_ref_states
                all_calc_energies[pdb_id] = pdb_ref_energies
                all_rmsds[pdb_id] = pdb_unweighted_rmses

                resampled_atleast_one = True
                did_resample[pdb_id] = True

        if resampled_atleast_one:

            all_grads = list()
            all_expected_rmses, all_n_effs = dict(), dict()
            total_rmse = 0.0
            for pdb_id in pdb_ids:
                (expected_rmse, (n_eff,)), pdb_id_grads = grad_fn_pdb_id(params_flat, all_traj_states, all_calc_energies, all_rmsds, pdb_id)
                all_grads.append(pdb_id_grads)
                all_expected_rmses[pdb_id] = expected_rmse
                all_n_effs[pdb_id] = n_eff
                total_rmse += expected_rmse
            all_grads = jnp.array(all_grads)
            mean_rmse = total_rmse / len(pdb_ids)

            all_ref_times.append(i)
            all_ref_rmses.append(float(mean_rmse))

            for pdb_id in pdb_ids:
                if did_resample[pdb_id]:
                    pdb_ref_times[pdb_id].append(i)
                    pdb_ref_rmses[pdb_id].append(float(all_expected_rmses[pdb_id]))


        iter_end = time.time()


        m = all_grads.shape[0]
        all_grads_norm = vmap(normalize)(all_grads)
        all_grads_norm_pinv = jnp.linalg.pinv(all_grads_norm)
        gu_unnorm = all_grads_norm_pinv @ jnp.ones(m)
        gu_norm = normalize(gu_unnorm)
        proj_dists = vmap(lambda gi: jnp.dot(gi, gu_norm))(all_grads) # Note we use the unnormalized gradients
        g_config = proj_dists.sum() * gu_norm
        grads = g_config


        for pdb_id in pdb_ids:
            pdb_rmse = all_expected_rmses[pdb_id]
            with open(id_rmse_paths[pdb_id], "a") as f:
                f.write(f"{pdb_rmse}\n")

        with open(neff_path, "a") as f:
            f.write(f"{pprint.pformat(all_n_effs)}\n")
        with open(rmse_path, "a") as f:
            f.write(f"{mean_rmse}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(ravel_fn(grads))}\n")
        with open(params_per_iter_path, "a") as f:
            f.write(f"{pprint.pformat(ravel_fn(params_flat))}\n")
        with open(pct_change_path, "a") as f:
            pct_changes = tree_util.tree_map(lambda x, y: (y - x) / jnp.abs(x) * 100, init_params, ravel_fn(params_flat))
            f.write(f"{pprint.pformat(pct_changes)}\n")

        all_rmses.append(mean_rmse)

        for pdb_id in pdb_ids:
            pdb_rmses[pdb_id].append(all_expected_rmses[pdb_id])

        if i % save_obj_every == 0 and i:
            fpath = obj_dir / f"pdb_ref_rmses_i{i}.pkl"
            with open(fpath, 'wb') as of:
                pickle.dump(pdb_ref_rmses, of)

            fpath = obj_dir / f"pdb_ref_times_i{i}.pkl"
            with open(fpath, 'wb') as of:
                pickle.dump(pdb_ref_times, of)

        # Save a checkpoint
        if i % ckpt_freq == 0 and save_checkpoints:
            ckpt = {"params": params_flat, "optimizer": optimizer, "opt_state": opt_state}
            checkpoint_manager.save(i, ckpt, save_kwargs={'save_args': save_args})


        updates, opt_state = optimizer.update(grads, opt_state, params_flat)
        params_flat = optax.apply_updates(params_flat, updates)


        plt.plot(onp.arange(i+1), all_rmses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_rmses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"rmses_iter{i}.png")
        plt.clf()


        colors = ["red", "green", "blue"]
        for pdb_id, color in zip(pdb_ids, colors):
            plt.plot(onp.arange(i+1), pdb_rmses[pdb_id], linestyle="--", label=pdb_id, color=color)
            plt.scatter(pdb_ref_times[pdb_id], pdb_ref_rmses[pdb_id], marker='o', color=color)

        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"individual_rmses_iter{i}.png")
        plt.clf()


def get_parser():

    parser = argparse.ArgumentParser(description="Optimize persistence length via standalone oxDNA package")

    parser.add_argument('--n-sims', type=int, default=1,
                        help="Number of individual simulations")
    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=0,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")

    parser.add_argument('--use-nbrs', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true')


    parser.add_argument('--orbax-ckpt-path', type=str, required=False,
                        help='Optional path to orbax checkpoint directory')
    parser.add_argument('--ckpt-freq', type=int, default=3,
                        help='Checkpointing frequency')

    parser.add_argument('--t-kelvin', type=float, default=293.15, help="Temperature in Kelvin")
    parser.add_argument('--dt', type=float, default=1e-3, help="Timestep")
    parser.add_argument('--salt-conc', type=float, default=0.5, help="Salt concentration")

    parser.add_argument('--save-obj-every', type=int, default=10,
                        help="Frequency of saving numpy files")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
