import pdb
import tqdm
import functools
import time
import pickle
from pathlib import Path
import datetime
import shutil
from functools import partial
import numpy as onp

import jax
from jax import jit, vmap, lax, random, value_and_grad
from jax.config import config as jax_config
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt # FIXME: change to optax
from pprint import pprint

from jax_md import space, util, simulate
from jax_md.rigid_body import RigidBody, Quaternion
from jax.tree_util import Partial

from utils import DEFAULT_TEMP
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import get_one_hot, bcolors
from utils import Q_to_base_normal

from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
# from energy import factory
from energy import hb_seq_dependent_factory as factory
from checkpoint import checkpoint_scan
from loss import propeller
from cgdna.utils import lb_dnas, get_marginals

from jax.config import config
config.update("jax_enable_x64", True)


f64 = util.f64

mass = RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))
base_site = jnp.array(
    [com_to_hb, 0.0, 0.0], dtype=f64
)
stack_site = jnp.array(
    [com_to_stacking, 0.0, 0.0], dtype=f64
)
back_site = jnp.array(
    [com_to_backbone, 0.0, 0.0], dtype=f64
)


checkpoint_every = 1 # FIXME: should really do sqrt(n_steps)
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args, init_params,
        T=DEFAULT_TEMP, dt=5e-3,
        output_basedir="v2/data/output"):


    top_path = args['top_path']
    conf_path = args['conf_path']
    sim_length = args['n_steps']
    batch_size = args['batch_size']
    opt_steps = args['opt_steps']
    lr = args['lr']
    key = random.PRNGKey(args['key'])
    save_every = args['save_every']

    num_eq_steps = args['n_eq'] # Number of equilibration steps
    sample_every = args['sample_every'] # Number of steps to sample an ensemble state after

    assert(num_eq_steps < sim_length)
    assert(sim_length - num_eq_steps > sample_every) # at least one sample

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    output_basedir = Path(output_basedir)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"optimize_hb_seq_params_{timestamp}_b{batch_size}_lr{lr}_n{sim_length}"
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    shutil.copy(top_path, run_dir)
    shutil.copy(conf_path, run_dir)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"

    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)
    print(bcolors.WARNING + f"Created directory and copied optimization information at location: {run_dir}" + bcolors.ENDC)

    print(bcolors.OKBLUE + f"Running optimization..." + bcolors.ENDC)

    # Information for a single "test case"
    # Note: in the future, we will have multiple of these
    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    # displacement_fn, shift_fn = space.periodic(config_info.box_size)
    displacement_fn, shift_fn = space.free()

    # Setup the simulation
    body = config_info.states[0]
    assert(len(top_info.seq) == 24*2)
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT
    gamma_grad = RigidBody(center=jnp.array([T*0.1/2.5]), orientation=jnp.array([T*0.1/7.5])) # gamma for gradients
    gamma_eq = RigidBody(center=jnp.array([kT/2.5]), orientation=jnp.array([kT/7.5])) # gamma for equilibration

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                             back_site, stack_site, base_site,
                                             top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq, op_nbrs_idx=top_info.unbonded_nbrs.T)

    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma_grad)
    step_fn = jit(step_fn)
    init_fn = jit(init_fn)

    @jit
    def run_eval_simulation(params, key, init_body):
        init_state = init_fn(key, R=init_body, mass=mass, params=params)
        # Take steps with `lax.scan`
        @jit
        def scan_fn(state, step):
            state = step_fn(state, params=params)
            return state, state.position
        _, trajectory = scan(scan_fn, init_state, jnp.arange(sim_length))

        return trajectory


    ## Set up a different simulation for equliibration
    init_fn_eq, step_fn_eq = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma_eq)
    step_fn_eq = jit(step_fn_eq)
    init_fn_eq = jit(init_fn_eq)

    @jit
    def run_eq_simulation(params, key):

        init_state = init_fn_eq(key, R=body, mass=mass, params=params)
        # Take steps with `lax.scan`
        @jit
        def scan_fn(state, step):
            state = step_fn_eq(state, params=params)
            return state, state.position
        final_state, _ = scan(scan_fn, init_state, jnp.arange(num_eq_steps))

        return final_state.position



    # construct loss function
    intra_coord_means, intra_coord_vars = get_marginals(top_info.seq[:24], verbose=False)
    reference_propeller_twist = 20.0
    propeller_means = reference_propeller_twist + jnp.array(intra_coord_means["propeller"][1:-1]) * 11.5 # convert to degrees
    propeller_vars = reference_propeller_twist + jnp.array(intra_coord_vars["propeller"][1:-1]) * 11.5 # convert to degrees

    propeller_base_pairs = list(zip(onp.arange(1, 23), onp.arange(46, 24, -1))) # FIXME: don't specialize for 23
    propeller_base_pairs = jnp.array(propeller_base_pairs)
    @jit
    def compute_all_propeller_twists(body):
        Q = body.orientation
        base_normals = Q_to_base_normal(Q)
        prop_twists = vmap(propeller.compute_single_propeller_twist, (0, None))(propeller_base_pairs, base_normals)
        return prop_twists

    @jit
    def kl_divergence(true_mean, true_var, est_mean, est_var):
        return jnp.log(est_var / true_var) + (true_var**2 + (true_mean - est_mean)**2) / (2 * est_var**2) - 1/2

    @jit
    def compute_kl_divergences(all_prop_twists):
        prop_twists_means = jnp.mean(all_prop_twists, axis=0)
        prop_twists_vars = jnp.var(all_prop_twists, axis=0)
        all_kl_divergences = vmap(kl_divergence, (0, 0, 0, 0))(propeller_means, propeller_vars, prop_twists_means, prop_twists_vars)
        return all_kl_divergences

    @jit
    def compute_trajectory_propeller_twists(trajectory):
        states_to_eval = trajectory[::sample_every]
        all_prop_twists = vmap(compute_all_propeller_twists)(states_to_eval)
        return all_prop_twists

    @jit
    def loss_fn(params, eq_bodies, keys):
        trajectories = vmap(run_eval_simulation, (None, 0, 0))(params, keys, eq_bodies)

        all_trajectory_prop_twists = vmap(compute_trajectory_propeller_twists)(trajectories)
        # note: dimension of all_trajectory_prop_twists will be (# trajectories, num_steps, 22)
        combined_trajectory_prop_twists = all_trajectory_prop_twists.reshape(-1, all_trajectory_prop_twists.shape[-1])
        kl_divergences = compute_kl_divergences(combined_trajectory_prop_twists)
        return jnp.mean(kl_divergences)


    # Get our gradients ready
    grad_fn = value_and_grad(loss_fn)

    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)

    # Setup some logging, some required and some not
    params_ = list()
    all_losses = list()
    all_grads = list()

    loss_path = run_dir / "losses.txt"
    grad_path = run_dir / "grads.txt"

    # Do the optimization
    step_times = list()
    for i in tqdm.trange(opt_steps, position=0):
        start = time.time()
        key, iter_key = random.split(key)
        equilibration_key, grad_key = random.split(iter_key)
        eq_seeds = jax.random.split(equilibration_key, batch_size)

        # Equilibrate
        eq_bodies = vmap(run_eq_simulation, (None, 0))(optimizer.params_fn(opt_state), eq_seeds)

        # Get the grad for our single test case (would have to average for multiple)
        grad_seeds = jax.random.split(grad_key, batch_size)
        loss, grad = grad_fn(optimizer.params_fn(opt_state), eq_bodies, grad_seeds)
        opt_state = optimizer.update_fn(i, grad, opt_state)

        end = time.time()

        if i % save_every == 0:
            step_times.append(end - start)
            # print(optimizer.params_fn(opt_state))

            all_grads.append(grad)
            params_.append(optimizer.params_fn(opt_state))
            all_losses.append(loss)

            with open(loss_path, "a") as f:
                f.write(f"{loss}\n")
            with open(grad_path, "a") as f:
                f.write(f"{grad}\n")

    with open(run_dir / "final_params.pkl", "wb") as f:
        pickle.dump(optimizer.params_fn(opt_state), f)
    with open(run_dir / "final_loss.pkl", "wb") as f:
        pickle.dump(loss, f)
    with open(run_dir / "all_losses.pkl", "wb") as f:
        pickle.dump(all_losses, f)
    with open(run_dir / "params.pkl", "wb") as f:
        pickle.dump(params_, f)
    with open(run_dir / "all_grads.pkl", "wb") as f:
        pickle.dump(all_grads, f)
    with open(run_dir / "step_times.pkl", "wb") as f:
        pickle.dump(step_times, f)
    return


if __name__ == "__main__":
    import argparse

    from loader import get_params


    parser = argparse.ArgumentParser(description="Optimizing over oxDNA parameters")
    parser.add_argument('-b', '--batch-size', type=int, default=2,
                        help="Num. batches for each round of gradient descent")
    parser.add_argument('--save-every', type=int, default=1,
                        help="Frequency of saving data from optimization")
    parser.add_argument('--opt-steps', type=int, default=3,
                        help="Num. iterations of gradient descent")
    parser.add_argument('-n', '--n-steps', type=int, default=100,
                        help="Num. steps per simulation")
    parser.add_argument('-k', '--key', type=int, default=0,
                        help="Random key")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate for optimization")
    parser.add_argument('--top-path', type=str,
                        default="data/lb-seqs/seq1/seq.top",
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        default="data/lb-seqs/seq1/seq.conf",
                        help='Path to configuration file')
    parser.add_argument('--n-eq', type=int, default=50,
                        help="Num. equilibration steps per simulation")
    parser.add_argument('--sample-every', type=int, default=10,
                        help="Num. steps per sample (after equilibration)")
    args = vars(parser.parse_args())


    # starting with the correct parameters
    init_params = get_params.get_init_optimize_params_hb_seq_dependent("oxdna")
    init_params = jnp.array(init_params)

    start = time.time()
    run(args, init_params=init_params)
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
