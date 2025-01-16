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

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad, lax, tree_util, random
from jax_md import space, rigid_body, simulate
import orbax.checkpoint
from flax.training import orbax_utils

from jax_dna.loss import rmse
from jax_dna.common import utils, topology, trajectory, center_configuration, checkpoint
from jax_dna.rna2 import model, oxrna_utils
from jax_dna.rna2.load_params import read_seq_specific, DEFAULT_BASE_PARAMS, EMPTY_BASE_PARAMS
import jax_dna.input.trajectory as jdt

jax.config.update("jax_enable_x64", True)


checkpoint_every = 1
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


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

    use_symm_coax = args['use_symm_coax']

    seq_avg_opt_keys = args['seq_avg_opt_keys']
    opt_seq_dep_stacking = args['opt_seq_dep_stacking']

    full_system = args['full_system']
    init_custom_params = args['init_custom_params']

    orbax_ckpt_path = args['orbax_ckpt_path']
    ckpt_freq = args['ckpt_freq']


    # t_kelvin = utils.DEFAULT_TEMP
    t_kelvin = 293.15

    ss_hb_weights, ss_stack_weights, ss_cross_weights = read_seq_specific(DEFAULT_BASE_PARAMS)
    # ss_hb_weights = utils.HB_WEIGHTS_SA
    # ss_stack_weights = utils.STACK_WEIGHTS_SA
    salt_conc = 1.0


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

    times_path = log_dir / "times.txt"
    params_per_iter_path = log_dir / "params_per_iter.txt"
    pct_change_path = log_dir / "pct_change.txt"
    grads_path = log_dir / "grads.txt"
    neff_path = log_dir / "neff.txt"
    rmse_path = log_dir / "rmse.txt"
    resample_log_path = log_dir / "resample_log.txt"

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    nvidia_smi_path = run_dir / "nvidia-smi.txt"
    os.system(f"nvidia-smi >> {nvidia_smi_path}")

    # Load the system
    if full_system:
        sys_basedir = Path("data/templates/5ht-tc-rmse-rna")
    else:
        sys_basedir = Path("data/templates/2ht-tc-rmse-rna")
    input_template_path = sys_basedir / "input"
    ss_path = sys_basedir / "rna_sequence_dependent_parameters.txt"

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False, is_rna=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq, is_rna=True), dtype=jnp.float64)

    target_path = sys_basedir / "target.conf"
    target_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=target_path,
        reverse_direction=False
    )
    target_state = target_info.get_states()[0]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    default_neighbors = None
    if use_nbrs:
        r_cutoff = 10.0
        dr_threshold = 0.2
        neighbor_fn = top_info.get_neighbor_list_fn(
            displacement_fn, box_size, r_cutoff, dr_threshold)
        # Note that we only allocate once
        neighbors = neighbor_fn.allocate(target_state.center) # We use the COMs.
        default_neighbors = deepcopy(neighbors)

    dt = 3e-3
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    def sim_fn(params, body, key, curr_stack_weights):

        if use_nbrs:
            neighbors_idx = default_neighbors.idx
        else:
            neighbors_idx = top_info.unbonded_nbrs.T

        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body, neighbors_idx: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=neighbors_idx)
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

    def get_ref_states(params, i, iter_key, init_body):

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)


        if "stacking" in params["seq_dep"]:
            curr_stack_weights = params["seq_dep"]["stacking"]
        else:
            curr_stack_weights = ss_stack_weights

        iter_key, sim_key = random.split(iter_key)
        sim_keys = random.split(sim_key, n_sims)
        sim_start = time.time()
        all_batch_ref_states = vmap(sim_fn, (None, None, 0, None))(params, init_body, sim_keys, curr_stack_weights)
        sim_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Simulating took {sim_end - sim_start} seconds\n")

        combined_center = all_batch_ref_states.center.reshape(-1, top_info.n, 3)
        combined_quat_vec = all_batch_ref_states.orientation.vec.reshape(-1, top_info.n, 4)

        traj_states = rigid_body.RigidBody(
            center=combined_center,
            orientation=rigid_body.Quaternion(combined_quat_vec))

        # n_traj_states = len(ref_states)
        n_traj_states = traj_states.center.shape[0]


        ## Generate an energy function
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
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
        plt.savefig(iter_dir / f"rmsd_hist.png")
        plt.clf()

        running_avg = onp.cumsum(RMSDs) / onp.arange(1, (n_ref_states)+1)
        plt.plot(running_avg)
        plt.savefig(iter_dir / "running_avg_rmsd.png")
        plt.clf()

        last_half = int((n_ref_states) // 2)
        plt.plot(running_avg[-last_half:])
        plt.savefig(iter_dir / "running_avg_rmsd_second_half.png")
        plt.clf()

        sns.distplot(calc_energies, label="Calculated", color="red")
        plt.legend()
        plt.savefig(iter_dir / f"energies.png")
        plt.clf()


        mean_rmsd = onp.mean(RMSDs)
        with open(iter_dir / "summary.txt", "w+") as f:
            f.write(f"Calc. energy var.: {onp.var(calc_energies)}\n")
            f.write(f"Mean RMSD: {mean_rmsd}\n")
            f.write(f"# Traj. States: {n_traj_states}\n")

        with open(iter_dir / "params.txt", "w+") as f:
            f.write(f"{pprint.pformat(params)}\n")

        analyze_end = time.time()
        with open(resample_log_path, "a") as f:
            f.write(f"- Remaining analysis took {analyze_end - analyze_start} seconds\n")


        return traj_states, calc_energies, jnp.array(RMSDs), iter_dir


    # Construct the loss function
    @jit
    def loss_fn(params, ref_states, ref_energies, unweighted_rmses):
        if "stacking" in params["seq_dep"]:
            curr_stack_weights = params["seq_dep"]["stacking"]
        else:
            curr_stack_weights = ss_stack_weights
        em = model.EnergyModel(
            displacement_fn, params["seq_avg"], t_kelvin=t_kelvin, salt_conc=salt_conc,
            ss_hb_weights=ss_hb_weights,
            # ss_stack_weights=ss_stack_weights)
            ss_stack_weights=curr_stack_weights, use_symm_coax=use_symm_coax)
        energy_fn = lambda body: em.energy_fn(
            body,
            seq=seq_oh,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, new_energies = scan(energy_scan_fn, None, ref_states)
        diffs = new_energies - ref_energies
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        expected_rmse = jnp.dot(weights, unweighted_rmses)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return expected_rmse, n_eff
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Initialize parameters
    if init_custom_params:
        # From new-theta1-coax-term-5ht-n6-s500k-nbrs-lr0.001
        """
        params = {
            'seq_avg': {
                'coaxial_stacking': {
                    'a_coax_1': 1.96262908,
                    'a_coax_3p': 1.99296538,
                    'a_coax_4': 1.31547434,
                    'a_coax_4p': 1.98005975,
                    'a_coax_5': 0.90579923,
                    'a_coax_6': 0.9,
                    'cos_phi3_star_coax': -0.65744625,
                    'cos_phi4_star_coax': -0.6536279,
                    'delta_theta_star_coax_1': 0.6176235,
                    'delta_theta_star_coax_4': 0.80430099,
                    'delta_theta_star_coax_5': 0.96948441,
                    'delta_theta_star_coax_6': 0.95,
                    'dr0_coax': 0.49690238,
                    'dr_c_coax': 0.62009351,
                    'dr_high_coax': 0.55889348,
                    'dr_low_coax': 0.43296995,
                    'k_coax': 80.01199268,
                    'theta0_coax_1': 2.63764088,
                    'theta0_coax_1_bonus': 0.3126271,
                    'theta0_coax_4': 0.1342195,
                    'theta0_coax_5': 0.66137077,
                    'theta0_coax_6': 0.685},
                'cross_stacking': {
                    'a_cross_1': 2.26508263,
                    'a_cross_2': 1.69865942,
                    'a_cross_3': 1.7,
                    'a_cross_7': 1.7048074,
                    'a_cross_8': 1.7,
                    'delta_theta_star_cross_1': 0.58282376,
                    'delta_theta_star_cross_2': 0.66958152,
                    'delta_theta_star_cross_3': 0.68,
                    'delta_theta_star_cross_7': 0.66579311,
                    'delta_theta_star_cross_8': 0.68,
                    'dr_c_cross': 0.61531136,
                    'dr_high_cross': 0.57318886,
                    'dr_low_cross': 0.4077584,
                    'k_cross': 59.98943963,
                    'r0_cross': 0.5045946,
                    'theta0_cross_1': 0.52710691,
                    'theta0_cross_2': 1.3033864,
                    'theta0_cross_3': 1.266,
                    'theta0_cross_7': 0.27459227,
                    'theta0_cross_8': 0.309},
                'debye': {},
                'excluded_volume': {},
                'fene': {},
                'geometry': {},
                'hydrogen_bonding': {},
                'stacking': {
                    'a_stack': 5.99374692,
                    'a_stack_1': 2.,
                    'a_stack_10': 1.296035,
                    'a_stack_2': 1.99130353,
                    'a_stack_5': 0.93388113,
                    'a_stack_6': 0.9,
                    'a_stack_9': 1.31621372,
                    'delta_theta_star_stack_10': 0.80477338,
                    'delta_theta_star_stack_5': 0.9414867,
                    'delta_theta_star_stack_6': 0.95,
                    'delta_theta_star_stack_9': 0.80666125,
                    'dr0_stack': 0.47092052,
                    'dr_c_stack': 0.92984975,
                    'dr_high_stack': 0.78532626,
                    'dr_low_stack': 0.35526074,
                    'eps_stack_base': 1.39927407,
                    'eps_stack_kt_coeff': 2.76721407,
                    'neg_cos_phi1_star_stack': -0.65,
                    'neg_cos_phi2_star_stack': -0.65,
                    'theta0_stack_10': 0.0007097,
                    'theta0_stack_5': -0.03309463,
                    'theta0_stack_6': 0.,
                    'theta0_stack_9': -0.01325876
                }
            },
            'seq_dep': {}
        }
        """

        # new-theta1-coax-term-5ht-n6-s500k-nbrs-lr0.001-longer
        params = {
            'seq_avg': {
                'coaxial_stacking': {
                    'a_coax_1': 1.95861459,
                    'a_coax_3p': 1.9958477,
                    'a_coax_4': 1.30047826,
                    'a_coax_4p': 1.97295687,
                    'a_coax_5': 0.91548833,
                    'a_coax_6': 0.9,
                    'cos_phi3_star_coax': -0.66187901,
                    'cos_phi4_star_coax': -0.6301233,
                    'delta_theta_star_coax_1': 0.60455419,
                    'delta_theta_star_coax_4': 0.80990688,
                    'delta_theta_star_coax_5': 0.96839335,
                    'delta_theta_star_coax_6': 0.95,
                    'dr0_coax': 0.49216608,
                    'dr_c_coax': 0.61984608,
                    'dr_high_coax': 0.55395259,
                    'dr_low_coax': 0.41081607,
                    'k_coax': 80.0175589,
                    'theta0_coax_1': 2.64801756,
                    'theta0_coax_1_bonus': 0.317804,
                    'theta0_coax_4': 0.14968202,
                    'theta0_coax_5': 0.65632999,
                    'theta0_coax_6': 0.685
                },
                'cross_stacking': {
                    'a_cross_1': 2.28548854,
                    'a_cross_2': 1.71566604,
                    'a_cross_3': 1.7,
                    'a_cross_7': 1.71609397,
                    'a_cross_8': 1.7,
                    'delta_theta_star_cross_1': 0.60423358,
                    'delta_theta_star_cross_2': 0.69296338,
                    'delta_theta_star_cross_3': 0.68,
                    'delta_theta_star_cross_7': 0.67289334,
                    'delta_theta_star_cross_8': 0.68,
                    'dr_c_cross': 0.62501931,
                    'dr_high_cross': 0.60295525,
                    'dr_low_cross': 0.41188735,
                    'k_cross': 60.00082877,
                    'r0_cross': 0.49473886,
                    'theta0_cross_1': 0.53981762,
                    'theta0_cross_2': 1.31077143,
                    'theta0_cross_3': 1.266,
                    'theta0_cross_7': 0.26046686,
                    'theta0_cross_8': 0.309
                },
                'debye': {},
                'excluded_volume': {},
                'fene': {},
                'geometry': {},
                'hydrogen_bonding': {},
                'stacking': {
                    'a_stack': 5.99304602,
                    'a_stack_1': 2.,
                    'a_stack_10': 1.29008657,
                    'a_stack_2': 1.99350629,
                    'a_stack_5': 0.93531628,
                    'a_stack_6': 0.9,
                    'a_stack_9': 1.31047992,
                    'delta_theta_star_stack_10': 0.79276849,
                    'delta_theta_star_stack_5': 0.95095821,
                    'delta_theta_star_stack_6': 0.95,
                    'delta_theta_star_stack_9': 0.79187244,
                    'dr0_stack': 0.47280903,
                    'dr_c_stack': 0.92447461,
                    'dr_high_stack': 0.77280411,
                    'dr_low_stack': 0.3541379,
                    'eps_stack_base': 1.39484306,
                    'eps_stack_kt_coeff': 2.76278306,
                    'neg_cos_phi1_star_stack': -0.65,
                    'neg_cos_phi2_star_stack': -0.65,
                    'theta0_stack_10': -0.00031732,
                    'theta0_stack_5': -0.03266478,
                    'theta0_stack_6': 0.,
                    'theta0_stack_9': -0.0140645
                }
            },
            'seq_dep': {}
        }
    else:
        seq_avg_params = deepcopy(EMPTY_BASE_PARAMS)
        for opt_key in seq_avg_opt_keys:
            seq_avg_params[opt_key] = deepcopy(DEFAULT_BASE_PARAMS[opt_key])
        params = {"seq_avg": seq_avg_params, "seq_dep": dict()}
        if opt_seq_dep_stacking:
            params["seq_dep"]["stacking"] = jnp.array(ss_stack_weights)

        if use_symm_coax:
            assert("coaxial_stacking" in seq_avg_opt_keys)

            params["seq_avg"]["coaxial_stacking"]["theta0_coax_1_bonus"] = 0.35



    # Setup optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)


    # Setup checkpointer
    ex_params = deepcopy(params)
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
        params = state_restored["params"]
        # optimizer = state_restored["optimizer"]
        opt_state = state_restored["opt_state"]



    # Optimize!

    init_params = deepcopy(params)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_rmses = list()
    all_n_effs = list()
    all_ref_rmses = list()
    all_ref_times = list()

    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")

    key = random.PRNGKey(0)
    init_body = centered_conf_info.get_states()[0]
    start = time.time()
    ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=0, iter_key=key, init_body=init_body)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        iter_start = time.time()

        (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)
        num_resample_iters += 1

        if i == 0:
            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters = 0
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")

            key, split = random.split(key)
            start = time.time()
            ref_states, ref_energies, unweighted_rmses, ref_iter_dir = get_ref_states(params, i=i, iter_key=split, init_body=ref_states[-1])
            end = time.time()
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")

            (curr_rmse, n_eff), grads = grad_fn(params, ref_states, ref_energies, unweighted_rmses)

            all_ref_times.append(i)
            all_ref_rmses.append(curr_rmse)

        iter_end = time.time()

        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(rmse_path, "a") as f:
            f.write(f"{curr_rmse}\n")
        with open(times_path, "a") as f:
            f.write(f"{iter_end - iter_start}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")
        with open(params_per_iter_path, "a") as f:
            f.write(f"{pprint.pformat(params)}\n")
        with open(pct_change_path, "a") as f:
            pct_changes = tree_util.tree_map(lambda x, y: (y - x) / jnp.abs(x) * 100, init_params, params)
            f.write(f"{pprint.pformat(pct_changes)}\n")

        all_n_effs.append(n_eff)
        all_rmses.append(curr_rmse)


        # Save a checkpoint
        if i % ckpt_freq == 0:
            ckpt = {"params": params, "optimizer": optimizer, "opt_state": opt_state}
            checkpoint_manager.save(i, ckpt, save_kwargs={'save_args': save_args})


        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)


        plt.plot(onp.arange(i+1), all_rmses, linestyle="--", color="blue")
        plt.scatter(all_ref_times, all_ref_rmses, marker='o', label="Resample points", color="blue")
        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("Iteration")
        plt.savefig(img_dir / f"rmses_iter{i}.png")
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

    parser.add_argument('--full-system', action='store_true')
    parser.add_argument('--use-nbrs', action='store_true')

    parser.add_argument(
        '--seq-avg-opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["stacking", "cross_stacking", "coaxial_stacking"],
        help='Parameter keys to optimize'
    )
    parser.add_argument('--opt-seq-dep-stacking', action='store_true')

    parser.add_argument('--use-symm-coax', action='store_true')

    parser.add_argument('--init-custom-params', action='store_true')

    parser.add_argument('--orbax-ckpt-path', type=str, required=False,
                        help='Optional path to orbax checkpoint directory')
    parser.add_argument('--ckpt-freq', type=int, default=3,
                        help='Checkpointing frequency')


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
