import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp

import optax
import jax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util, pmap
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = None
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


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

    # Load arguments
    n_eq_steps = args['n_eq_steps']
    skipped_quartets_per_end = args['skipped_quartets_per_end']
    n_devices = jax.local_device_count()
    n_expected_devices = args['n_expected_devices']
    assert(n_devices == n_expected_devices)
    n_steps_per_batch = args['n_steps_per_batch']
    sample_every = args['sample_every']
    lr = args['lr']
    target_lp = args['target_lp']
    n_iters = args['n_iters']
    min_neff_factor = args['min_neff_factor']
    run_name = args['run_name']

    # Setup the logging directoroy
    if run_name is None:
        raise RuntimeError(f"Must set a run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    img_dir = run_dir / "img"
    img_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/sys-defs/persistence-length")
    top_path = sys_basedir / "init.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "relaxed.dat"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = conf_info.get_states()[0]

    # Setup utilities for simulation
    displacement_fn, shift_fn = space.free()
    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    # Setup functions for generation of reference states
    def eq_fn(params, body, key):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)

        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass)

        fori_step_fn = lambda t, state: step_fn(state)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position
    batched_eq_fn = pmap(eq_fn, in_axes=(None, None, 0))

    n_steps = n_steps_per_batch * n_devices
    assert(n_steps_per_batch % sample_every == 0)
    n_ref_states_per_batch = n_steps_per_batch // sample_every
    n_ref_states = n_ref_states_per_batch * n_devices

    def get_ref_states(params, eq_init_body, key):

        # Equilibrate
        key, eq_key = random.split(key)
        eq_keys = random.split(eq_key, n_devices)
        eq_bodies = batched_eq_fn(params, eq_init_body, eq_keys)


        # Simulate the equilibrated states
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)

        def batch_fn(batch_key, eq_body):
            init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
            init_state = init_fn(batch_key, eq_body, mass=mass)

            step_fn = jit(step_fn)

            fori_step_fn = lambda t, state: step_fn(state)
            fori_step_fn = jit(fori_step_fn)

            @jit
            def scan_fn(state, step):
                state = lax.fori_loop(0, sample_every, fori_step_fn, state)
                return state, state.position

            _, batch_ref_states = lax.scan(scan_fn, init_state, jnp.arange(n_ref_states_per_batch))
            return batch_ref_states

        batch_keys = random.split(key, n_devices)
        all_batch_ref_states = pmap(batch_fn)(batch_keys, eq_bodies)

        num_bases = all_batch_ref_states.center.shape[2] # FIXME: should just use top_info.n
        assert(all_batch_ref_states.center.shape[3] == 3)

        combined_center = all_batch_ref_states.center.reshape(-1, num_bases, 3)
        combined_quat_vec = all_batch_ref_states.orientation.vec.reshape(-1, num_bases, 4)

        ref_states = rigid_body.RigidBody(
            center=combined_center,
            orientation=rigid_body.Quaternion(combined_quat_vec))
        ref_energies = vmap(energy_fn)(ref_states)

        return ref_states, ref_energies

    quartets = get_all_quartets(n_nucs_per_strand=init_body.center.shape[0] // 2)
    quartets = quartets[skipped_quartets_per_end:]
    quartets = quartets[:-skipped_quartets_per_end]
    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])
    compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))


    def log_ref_states_info(ref_states, i):
        all_curves = list()
        all_l0_avg = list()
        intermediate_lps = dict()
        running_avg_interval = 10
        min_running_avg_idx = 50
        for s_idx in tqdm(range(n_ref_states), desc="Computing running average of reference states"):
            body = ref_states[s_idx]
            correlation_curve, l0_avg = persistence_length.get_correlation_curve(body, quartets, base_site)
            all_curves.append(correlation_curve)
            all_l0_avg.append(l0_avg)

            if s_idx % running_avg_interval == 0 and s_idx != 0:
                mean_correlation_curve = jnp.mean(jnp.array(all_curves), axis=0)
                mean_l0_avg = jnp.mean(jnp.array(all_l0_avg))
                mean_Lp = persistence_length.persistence_length_fit(mean_correlation_curve, mean_l0_avg)
                intermediate_lps[i*sample_every] = mean_Lp * utils.nm_per_oxdna_length

        plt.plot(intermediate_lps.keys(), intermediate_lps.values())
        plt.xlabel("Time")
        plt.ylabel("Lp (nm)")
        plt.title("Running Average")
        plt.savefig(img_dir / "running_avg_i{i}.png")
        plt.clf()

        plt.plot(list(intermediate_lps.keys())[min_running_avg_idx:],
                 list(intermediate_lps.values())[min_running_avg_idx:])
        plt.xlabel("Time")
        plt.ylabel("Lp (nm)")
        plt.title("Running Average, Initial Truncation")
        plt.savefig(img_dir / "truncated_running_avg_i{i}.png")
        plt.clf()

    # Construct the loss function

    @jit
    def loss_fn(params, ref_states, ref_energies):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)

        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        new_energies = vmap(energy_fn)(ref_states)
        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(ref_states, quartets, base_site)
        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        weighted_l0_avgs = vmap(lambda l0, w: l0 * w)(unweighted_l0_avgs, weights)
        expected_corr_curv = jnp.sum(weighted_corr_curves, axis=0)
        expected_l0_avg = jnp.sum(weighted_l0_avgs)
        expected_lp = persistence_length.persistence_length_fit(expected_corr_curv,
                                                                expected_l0_avg)
        expected_lp = expected_lp * utils.nm_per_oxdna_length


        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return (expected_lp - target_lp)**2, (n_eff, expected_lp, expected_corr_curv)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    key = random.PRNGKey(0)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_times = list()

    loss_path = run_dir / "loss.txt"
    neff_path = run_dir / "neff.txt"
    lp_path = run_dir / "lp.txt"
    resample_log_path = run_dir / "resample_log.txt"

    init_body = conf_info.get_states()[0]
    print(f"Generating initial reference states and energies...")
    with open(resample_log_path, "a") as f:
        f.write(f"Generating initial reference states and energies...\n")
    start = time.time()
    ref_states, ref_energies = get_ref_states(params, init_body, key)
    end = time.time()
    with open(resample_log_path, "a") as f:
        f.write(f"Finished generating initial reference states. Took {end - start} seconds.\n\n")
    log_ref_states_info(ref_states, i)

    for i in tqdm(range(n_iters)):
        (loss, (n_eff, curr_lp, expected_corr_curv)), grads = grad_fn(params, ref_states, ref_energies)

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)

        if n_eff < min_n_eff:
            with open(resample_log_path, "a") as f:
                f.write(f"Iteration {i}\n")
                f.write(f"- n_eff was {n_eff}. Resampling...\n")
            key, split = random.split(key)

            eq_key, ref_key = random.split(split)

            start = time.time()
            ref_states, ref_energies = get_ref_states(params, ref_states[-1], ref_key)
            end = time.time()
            with open(resample_log_path, "a") as f:
                f.write(f"- time to resample: {end - start} seconds\n\n")
            log_ref_states_info(ref_states, i)
            (loss, (n_eff, curr_lp, expected_corr_curv)), grads = grad_fn(params, ref_states, ref_energies)

            all_ref_losses.append(loss)
            all_ref_times.append(i)

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        with open(lp_path, "a") as f:
            f.write(f"{curr_lp}\n")

        all_losses.append(loss)
        all_n_effs.append(n_eff)

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        plt.plot(expected_corr_curv)
        plt.savefig(img_dir / f"corr_iter{i}.png")
        plt.clf()

        plt.plot(onp.arange(i+1), all_losses, linestyle="--")
        plt.scatter(all_ref_times, all_ref_losses, marker='o')
        plt.savefig(img_dir / f"losses_iter{i}.png")
        plt.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize persistence length using differentiable trajectory reweighting")

    parser.add_argument('--n-expected-devices', type=int,
                        help="Expected number of devices. Present as a sanity check. This also serves as the batch size.")
    parser.add_argument('--n-iters', type=int, default=100,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--n-eq-steps', type=int,
                        help="Number of equilibration steps per device/batch. One batch per device.")
    parser.add_argument('--n-steps-per-batch', type=int, default=int(1e7),
                        help="Number of total steps (post-equilibration) for sampling per batch.")
    parser.add_argument('--sample-every', type=int, default=int(1e4),
                        help="Frequency of sampling reference states.")
    parser.add_argument('--skipped-quartets-per-end', type=int, default=5,
                        help="Number of quartets to ignore per end when calculating persistence length")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--target-lp', type=float,
                        help="Target persistence length in nanometers")
    parser.add_argument('--run-name', type=str,
                        help='Run name')

    args = vars(parser.parse_args())

    run(args)
