import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp
import argparse
import seaborn as sns

import jax
jax.config.update("jax_enable_x64", True)
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, lax
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.dna1 import model_bp_prob as model
from jax_dna.common.read_seq_specific import read_ss_oxdna
from jax_dna.loss import persistence_length



checkpoint_every = 10
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)

compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))

def run(args):

    # Load parameters
    n_iters = args['n_iters']
    use_gumbel = args['use_gumbel']
    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)

    use_nbrs = args['use_nbrs']

    n_sims = args['n_sims']
    n_sample_steps = args['n_sample_steps']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_sample_steps % sample_every == 0)
    num_points_per_batch = n_sample_steps // sample_every
    n_ref_states = num_points_per_batch * n_sims

    lr = args['lr']
    min_neff_factor = args['min_neff_factor']

    plot_every = args['plot_every']
    run_name = args['run_name']
    target_lp = args['target_lp']
    max_approx_iters = args['max_approx_iters']

    truncation = args['truncation']
    offset = args['offset']

    # Setup the logging directoroy
    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    log_dir = run_dir / "log"
    log_dir.mkdir(parents=False, exist_ok=False)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    pseq_dir = run_dir / "pseq"
    pseq_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    empty_params = deepcopy(model.EMPTY_BASE_PARAMS)

    ss_path = "data/seq-specific/seq_oxdna1.txt"
    ss_hb_weights, ss_stack_weights = read_ss_oxdna(ss_path)
    ss_hb_weights = jnp.array(ss_hb_weights)
    ss_stack_weights = jnp.array(ss_stack_weights)


    # Load the system
    sys_basedir = Path("data/templates/simple-helix-60bp-ss")

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_length = len(top_info.seq)
    n_bp = (seq_length // 2)
    assert(n_bp == 60)

    quartets = utils.get_all_quartets(n_nucs_per_strand=n_bp)
    quartets = quartets[offset:-offset-1]
    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])

    def normalize(logits, temp):
        if use_gumbel:
            pseq = jax.nn.softmax(logits / temp)
        else:
            pseq = jax.nn.softmax(logits)

        return pseq

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    box_size = conf_info.box_size

    # Setup utilities for simulation
    displacement_fn, shift_fn = space.free()

    default_neighbors = None
    if use_nbrs:
        r_cutoff = 10.0
        dr_threshold = 0.2
        neighbor_fn = top_info.get_neighbor_list_fn(
            displacement_fn, box_size, r_cutoff, dr_threshold)
        # Note that we only allocate once
        tmp_nbr_state = centered_conf_info.get_states()[0]
        neighbors = neighbor_fn.allocate(tmp_nbr_state.center) # We use the COMs.
        default_neighbors = deepcopy(neighbors)


    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    bps = jnp.array([[i, seq_length-i-1] for i in range(n_bp)], dtype=jnp.int32)
    n_bps = bps.shape[0]

    idx_to_bp_idx = onp.zeros((seq_length, 2), dtype=onp.int32)
    for bp_idx, (nt1, nt2) in enumerate(bps):
        idx_to_bp_idx[nt1] = [bp_idx, 0]
        idx_to_bp_idx[nt2] = [bp_idx, 1]
    idx_to_bp_idx = jnp.array(idx_to_bp_idx)

    unpaired = jnp.array([])
    is_unpaired = jnp.array([(i in set(onp.array(unpaired))) for i in range(seq_length)]).astype(jnp.int32)
    n_unpaired = unpaired.shape[0]

    idx_to_unpaired_idx = onp.arange(seq_length)
    for up_idx, idx in enumerate(unpaired):
        idx_to_unpaired_idx[idx] = up_idx
    idx_to_unpaired_idx = jnp.array(idx_to_unpaired_idx)


    em = model.EnergyModel(displacement_fn, bps, unpaired, is_unpaired, idx_to_unpaired_idx, idx_to_bp_idx,
                           override_base_params=empty_params, t_kelvin=t_kelvin, ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)



    def sample_fn(body, key, unpaired_pseq, bp_pseq):

        if use_nbrs:
            neighbors_idx = default_neighbors.idx
        else:
            neighbors_idx = top_info.unbonded_nbrs.T


        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(
            key, body, mass=mass,
            unpaired_pseq=unpaired_pseq, bp_pseq=bp_pseq,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)

        @jit
        def fori_step_fn(t, carry):
            state, neighbors = carry
            if use_nbrs:
                neighbors = neighbors.update(state.position.center)
                neighbors_idx = neighbors.idx
            else:
                neighbors_idx = top_info.unbonded_nbrs.T

            state= step_fn(
                state,
                unpaired_pseq=unpaired_pseq, bp_pseq=bp_pseq,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=neighbors_idx)
            return (state, neighbors)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(carry, step):
            (state, neighbors) = lax.fori_loop(0, sample_every, fori_step_fn, carry)
            return (state, neighbors), state.position

        start = time.time()
        (eq_state, eq_neighbors) = lax.fori_loop(0, n_eq_steps, fori_step_fn, (init_state, default_neighbors))
        (fin_state, _), traj = scan(scan_fn, (eq_state, eq_neighbors), jnp.arange(num_points_per_batch))
        end = time.time()

        return traj

    def batch_sim(ref_key, R, unpaired_pseq, bp_pseq):

        sample_keys = random.split(ref_key, n_sims)
        sample_trajs = vmap(sample_fn, (None, 0, None, None))(R, sample_keys, unpaired_pseq, bp_pseq)

        # sample_traj = utils.tree_stack(sample_trajs)
        sample_center = sample_trajs.center.reshape(-1, seq_length, 3)
        sample_qvec = sample_trajs.orientation.vec.reshape(-1, seq_length, 4)
        sample_traj = rigid_body.RigidBody(
            center=sample_center,
            orientation=rigid_body.Quaternion(sample_qvec))
        return sample_traj



    def get_ref_states(params, init_body, key, i, temp):

        curr_bp_logits = params['bp_logits']
        bp_pseq = normalize(curr_bp_logits, temp)

        curr_up_logits = params['up_logits']
        up_pseq = normalize(curr_up_logits, temp)


        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        key, batch_key = random.split(key)
        ref_states = batch_sim(batch_key, init_body, up_pseq, bp_pseq)


        energy_fn = lambda body: em.energy_fn(
            body,
            unpaired_pseq=up_pseq, bp_pseq=bp_pseq,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        energy_scan_fn = lambda state, ts: (None, energy_fn(ts))
        _, ref_energies = scan(energy_scan_fn, None, ref_states)


        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(ref_states, quartets, base_site)
        mean_corr_curve = jnp.mean(unweighted_corr_curves, axis=0)
        mean_l0 = jnp.mean(unweighted_l0_avgs)
        mean_Lp_truncated, offset = persistence_length.persistence_length_fit(mean_corr_curve[:truncation], mean_l0)



        compute_every = 10
        n_curves = unweighted_corr_curves.shape[0]
        all_inter_lps = list()
        all_inter_lps_truncated = list()
        for i in range(0, n_curves, compute_every):
            inter_mean_corr_curve = jnp.mean(unweighted_corr_curves[:i], axis=0)

            inter_mean_Lp_truncated, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve[:truncation], mean_l0)
            all_inter_lps_truncated.append(inter_mean_Lp_truncated * utils.nm_per_oxdna_length)

            inter_mean_Lp, _ = persistence_length.persistence_length_fit(inter_mean_corr_curve, mean_l0)
            all_inter_lps.append(inter_mean_Lp * utils.nm_per_oxdna_length)



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

        # fit_fn = lambda n: -n * mean_l0 / (mean_Lp_truncated/utils.nm_per_oxdna_length) + offset
        fit_fn = lambda n: -n * (mean_l0 / mean_Lp_truncated) + offset
        plt.plot(jnp.log(mean_corr_curve)[:truncation])
        # neg_inverse_slope = (mean_Lp_truncated / utils.nm_per_oxdna_length) / mean_l0 # in nucleotides
        neg_inverse_slope = mean_Lp_truncated / mean_l0 # in nucleotides
        rounded_offset = onp.round(offset, 3)
        rounded_neg_inverse_slope = onp.round(neg_inverse_slope, 3)
        fit_str = f"fit, -n/{rounded_neg_inverse_slope} + {rounded_offset}"
        plt.plot(fit_fn(jnp.arange(truncation)), linestyle='--', label=fit_str)





        # Record the loss
        with open(iter_dir / "summary.txt", "w+") as f:

            f.write(f"\nMean Lp truncated (oxDNA units): {mean_Lp_truncated}\n")
            f.write(f"Mean L0 (oxDNA units): {mean_l0}\n")

            f.write(f"\nMean Lp truncated (nm): {mean_Lp_truncated * utils.nm_per_oxdna_length}\n")
            f.write(f"Mean L0 (nm): {mean_l0 * utils.nm_per_oxdna_length}\n")


        return ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs


    @jit
    def loss_fn(params, ref_states: rigid_body.RigidBody, ref_energies, unweighted_corr_curves, unweighted_l0_avgs, temp):
        curr_bp_logits = params['bp_logits']
        bp_pseq = normalize(curr_bp_logits, temp)

        curr_up_logits = params['up_logits']
        up_pseq = normalize(curr_up_logits, temp)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(
            body,
            unpaired_pseq=up_pseq, bp_pseq=bp_pseq,
            bonded_nbrs=top_info.bonded_nbrs,
            unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # new_energies = vmap(energy_fn)(ref_states)
        energy_scan_fn = lambda state, rs: (None, energy_fn(rs))
        _, new_energies = scan(energy_scan_fn, None, ref_states)


        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        # Compute the observable
        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        expected_corr_curve = jnp.sum(weighted_corr_curves, axis=0)

        expected_l0_avg = jnp.dot(unweighted_l0_avgs, weights)

        expected_lp, expected_offset = persistence_length.persistence_length_fit(
            expected_corr_curve[:truncation],
            expected_l0_avg)

        expected_lp = expected_lp * utils.nm_per_oxdna_length

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        mse = (expected_lp - target_lp)**2
        rmse = jnp.sqrt(mse)

        return rmse, (n_eff, expected_lp, expected_corr_curve, expected_l0_avg*utils.nm_per_oxdna_length, expected_offset, up_pseq, bp_pseq)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization

    bp_logits = onp.full((n_bps, 4), 100.0)
    bp_logits = jnp.array(bp_logits, dtype=jnp.float64)

    if n_unpaired == 0:
        up_logits = onp.full((1, 4), 100.0)
    else:
        up_logits = onp.full((n_unpaired, 4), 100.0)
    up_logits = jnp.array(up_logits, dtype=jnp.float64)



    params = {
        "up_logits": up_logits,
        "bp_logits": bp_logits
    }

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    key = random.PRNGKey(0)

    init_body = centered_conf_info.get_states()[0]
    print(f"Generating initial reference states and energies...")
    ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs = get_ref_states(params, init_body, key, 0, temp=gumbel_temps[0])

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_ref_losses = list()
    all_ref_times = list()

    loss_path = log_dir / "loss.txt"
    lp_path = log_dir / "lp.txt"
    neff_path = log_dir / "neff.txt"
    argmax_seq_path = log_dir / "argmax_seq.txt"
    argmax_seq_scaled_path = log_dir / "argmax_seq_scaled.txt"
    grads_path = log_dir / "grads.txt"

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        (loss, (n_eff, curr_lp, expected_corr_curve, curr_l0_avg, curr_offset, up_pseq, bp_pseq)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs, gumbel_temps[i])


        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters

            print(f"Resampling reference states...")
            key, split = random.split(key)
            ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs = get_ref_states(params, ref_states[-1], split, i, temp=gumbel_temps[i])
            (loss, (n_eff, curr_lp, expected_corr_curve, curr_l0_avg, curr_offset, up_pseq, bp_pseq)), grads = grad_fn(params, ref_states, ref_energies, unweighted_corr_curves, unweighted_l0_avgs, gumbel_temps[i])

            all_ref_losses.append(loss)
            all_ref_times.append(i)


        up_pseq_fpath = pseq_dir / f"up_pseq_i{i}.npy"
        jnp.save(up_pseq_fpath, up_pseq, allow_pickle=False)

        bp_pseq_fpath = pseq_dir / f"bp_pseq_i{i}.npy"
        jnp.save(bp_pseq_fpath, bp_pseq, allow_pickle=False)


        argmax_seq = ["X"] * seq_length
        argmax_seq_scaled = ["X"] * seq_length
        max_up_nts = jnp.argmax(up_pseq, axis=1)
        if n_unpaired:
            raise RuntimeError(f"Haven't implemented including max unpaired")

        max_bps = jnp.argmax(bp_pseq, axis=1)
        max_bp_probs = jnp.max(bp_pseq, axis=1)
        for (nt1_idx, nt2_idx), bp_type_idx, bp_type_prob in zip(bps, max_bps, max_bp_probs):
            bp1, bp2 = model.BP_TYPES[bp_type_idx]

            argmax_seq[nt1_idx] = bp1
            argmax_seq[nt2_idx] = bp2

            if bp_type_prob < 0.5:
                argmax_seq_scaled[nt1_idx] = bp1.lower()
                argmax_seq_scaled[nt2_idx] = bp2.lower()
            else:
                argmax_seq_scaled[nt1_idx] = bp1
                argmax_seq_scaled[nt2_idx] = bp2

        argmax_seq = ''.join(argmax_seq)
        argmax_seq_scaled = ''.join(argmax_seq_scaled)

        with open(argmax_seq_path, "a") as f:
            f.write(f"{argmax_seq}\n")
        with open(argmax_seq_scaled_path, "a") as f:
            f.write(f"{argmax_seq_scaled}\n")

        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads)}\n")

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        all_losses.append(loss)


        print(f"Loss: {loss}")
        print(f"Effective sample size: {n_eff}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0:

            # Plot the losses
            plt.plot(onp.arange(i+1), all_losses, linestyle="--")
            plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            # plt.title(f"DiffTRE E2E Dist Optimization, Neff factor={min_neff_factor}")
            plt.savefig(img_dir / f"losses_iter{i}.png")
            plt.clf()


def get_parser():
    parser = argparse.ArgumentParser(description="Optimize structural properties using differentiable trajectory reweighting")

    parser.add_argument('--n-iters', type=int, default=100,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=100000,
                        help="Number of total steps for sampling reference states")
    parser.add_argument('--n-sims', type=int, default=5,
                        help="Number of independent simulations")
    parser.add_argument('--sample-every', type=int, default=500,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--run-name', type=str,
                        help='Run name')
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")

    parser.add_argument('--use-gumbel', action='store_true',
                        help="If true, will use gumbel softmax with an annealing temperature")
    parser.add_argument('--gumbel-start', type=float, default=1.0,
                        help="Starting temperature for gumbel softmax")
    parser.add_argument('--gumbel-end', type=float, default=0.01,
                        help="End temperature for gumbel softmax")

    parser.add_argument('--target-lp', type=float, default=0.0,
                        help="Target persistence length in nanometers")

    parser.add_argument('--use-nbrs', action='store_true')

    parser.add_argument('--truncation', type=int, default=40,
                        help="Truncation of quartets for fitting correlatoin curve")
    parser.add_argument('--offset', type=int, default=4,
                        help="Offset for number of quartets to skip on either end of the duplex")


    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())
    run(args)
