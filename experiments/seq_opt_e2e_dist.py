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

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, lax
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.dna1 import model
from jax_dna.common.read_seq_specific import read_ss_oxdna

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


checkpoint_every = None
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args):

    # Load parameters
    n_iters = args['n_iters']
    use_gumbel = args['use_gumbel']
    gumbel_end = args['gumbel_end']
    gumbel_start = args['gumbel_start']
    gumbel_temps = onp.linspace(gumbel_start, gumbel_end, n_iters)

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
    target_dist = args['target_dist']
    max_approx_iters = args['max_approx_iters']

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

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    empty_params = deepcopy(model.EMPTY_BASE_PARAMS)

    ss_path = "data/seq-specific/seq_oxdna1.txt"
    ss_hb_weights, ss_stack_weights = read_ss_oxdna(ss_path)


    # Load the system
    sys_basedir = Path("data/templates/ss20")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_length = len(top_info.seq)

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


    em = model.EnergyModel(displacement_fn, empty_params, t_kelvin=t_kelvin, ss_hb_weights=ss_hb_weights, ss_stack_weights=ss_stack_weights)

    def eq_fn(eq_key, body, pseq):
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=pseq,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)
        def fori_step_fn(t, state):
            return step_fn(state,
                           seq=pseq,
                           bonded_nbrs=top_info.bonded_nbrs,
                           unbonded_nbrs=top_info.unbonded_nbrs.T)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position

    def sample_fn(body, key, pseq):
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=pseq,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        def fori_step_fn(t, state):
            return step_fn(state,
                           seq=pseq,
                           bonded_nbrs=top_info.bonded_nbrs,
                           unbonded_nbrs=top_info.unbonded_nbrs.T)
        fori_step_fn = jit(fori_step_fn)

        @jit
        def scan_fn(state, step):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            return state, state.position

        start = time.time()
        fin_state, traj = scan(scan_fn, init_state, jnp.arange(num_points_per_batch))
        end = time.time()

        return traj

    def batch_sim(ref_key, R, pseq):

        ref_key, eq_key = random.split(ref_key)
        eq_keys = random.split(eq_key, n_sims)
        eq_states = vmap(eq_fn, (0, None, None))(eq_keys, R, pseq)

        sample_keys = random.split(ref_key, n_sims)
        sample_trajs = vmap(sample_fn, (0, 0, None))(eq_states, sample_keys, pseq)

        # sample_traj = utils.tree_stack(sample_trajs)
        sample_center = sample_trajs.center.reshape(-1, seq_length, 3)
        sample_qvec = sample_trajs.orientation.vec.reshape(-1, seq_length, 4)
        sample_traj = rigid_body.RigidBody(
            center=sample_center,
            orientation=rigid_body.Quaternion(sample_qvec))
        return sample_traj


    def e2e_distance(body):
        return space.distance(displacement_fn(body.center[0], body.center[-1]))

    def get_ref_states(params, init_body, key, i, temp):

        curr_logits = params['logits']
        pseq = normalize(curr_logits, temp)


        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)

        key, batch_key = random.split(key)
        ref_states = batch_sim(batch_key, init_body, pseq)


        energy_fn = lambda body: em.energy_fn(body,
                                              seq=pseq,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # ref_energies = [energy_fn(body) for body in ref_states]
        # return ref_states, jnp.array(ref_energies)

        ref_energies = vmap(energy_fn)(ref_states)

        ref_dists = vmap(e2e_distance)(ref_states) # FIXME: this doesn't depend on params...

        n_traj_states = len(ref_dists)
        running_avg_dists = onp.cumsum(ref_dists) / onp.arange(1, n_traj_states + 1)
        plt.plot(running_avg_dists)
        plt.savefig(iter_dir / f"running_avg.png")
        plt.close()

        plt.plot(running_avg_dists[-int(n_traj_states // 2):])
        plt.savefig(iter_dir / f"running_avg_second_half.png")
        plt.close()

        return ref_states, ref_energies, ref_dists


    @jit
    def loss_fn(params, ref_states: rigid_body.RigidBody, ref_energies, ref_dists, temp):
        logits = params['logits']
        pseq = normalize(logits, temp)


        # Compute the weights
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=pseq,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)
        new_energies = vmap(energy_fn)(ref_states)
        diffs = new_energies - ref_energies # element-wise subtraction
        boltzs = jnp.exp(-beta * diffs)
        denom = jnp.sum(boltzs)
        weights = boltzs / denom

        # Compute the observable
        expected_dist = jnp.dot(weights, ref_dists)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        mse = (expected_dist - target_dist)**2
        rmse = jnp.sqrt(mse)

        return rmse, (n_eff, expected_dist)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    init_logits = onp.full((seq_length, 4), 100.0)
    init_logits = jnp.array(init_logits, dtype=jnp.float64)
    params = {"logits": init_logits}
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    key = random.PRNGKey(0)

    init_body = conf_info.get_states()[0]
    print(f"Generating initial reference states and energies...")
    ref_states, ref_energies, ref_dists = get_ref_states(params, init_body, key, 0, temp=gumbel_temps[0])

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_edists = list()
    all_ref_losses = list()
    all_ref_edists = list()
    all_ref_times = list()

    loss_path = log_dir / "loss.txt"
    neff_path = log_dir / "neff.txt"
    dist_path = log_dir / "dist.txt"

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        (loss, (n_eff, expected_dist)), grads = grad_fn(params, ref_states, ref_energies, ref_dists, gumbel_temps[0])


        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)
            all_ref_edists.append(expected_dist)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters

            print(f"Resampling reference states...")
            key, split = random.split(key)
            # ref_states, ref_energies, ref_dists = get_ref_states(params, ref_states[-1], split, i, temp=gumbel_temps[i])
            ref_states, ref_energies, ref_dists = get_ref_states(params, init_body, split, i, temp=gumbel_temps[i])
            (loss, (n_eff, expected_dist)), grads = grad_fn(params, ref_states, ref_energies, ref_dists, gumbel_temps[i])

            all_ref_losses.append(loss)
            all_ref_edists.append(expected_dist)
            all_ref_times.append(i)

        with open(loss_path, "a") as f:
            f.write(f"{loss}\n")
        with open(dist_path, "a") as f:
            f.write(f"{expected_dist}\n")
        with open(neff_path, "a") as f:
            f.write(f"{n_eff}\n")
        all_losses.append(loss)
        all_edists.append(expected_dist)


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

            # Plot the persistence lengths
            plt.plot(onp.arange(i+1), all_edists, linestyle="--", color='blue')
            plt.scatter(all_ref_times, all_ref_edists, marker='o', label="Resample points", color='blue')
            plt.axhline(y=target_dist, linestyle='--', label="Target E2E dist", color='red')
            plt.xlabel("Iteration")
            plt.ylabel("Expected E2E Dist (oxDNA units)")
            plt.legend()
            # plt.title(f"DiffTRE E2E Dist Optimization, Neff factor={min_neff_factor}")
            plt.savefig(img_dir / f"edists_iter{i}.png")
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
    parser.add_argument('--target-dist', type=float, default=0.0,
                        help="Target end to end distance in oxDNA units")
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

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())
    run(args)
