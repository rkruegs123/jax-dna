import pdb
import tqdm
import functools
import time
import pickle
from pathlib import Path
import datetime
import shutil

import jax
from jax import jit, vmap, lax, random
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

from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from energy import factory
from checkpoint import checkpoint_scan
from loss import geometry
from loss import structural

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


checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run_simulation(params, key, steps, init_fn, step_fn):

    init_state = init_fn(key, params=params)
    # Take steps with `lax.scan`
    @jit
    def scan_fn(state, step):
        state = step_fn(state, params=params)
        return state, state.position
    final_state, trajectory = scan(scan_fn, init_state, jnp.arange(steps))

    return trajectory


def run(args, init_params=init_params,
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

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    output_basedir = Path(output_basedir)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"optimize_basic_{timestamp}_b{batch_size}_lr{lr}_n{sim_length}"
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

    # Construct the simulation loss function
    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT
    gamma = RigidBody(center=jnp.array([kT/2.5]), orientation=jnp.array([kT/7.5]))

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                             back_site, stack_site, base_site,
                                             top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq)

    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)
    init_fn = Partial(init_fn, R=body, mass=mass)
    init_fn = jit(init_fn)

    backbone_dist_pairs = top_info.bonded_nbrs
    helical_pairs = top_info.bonded_nbrs
    pitch_quartets = jnp.array([
        [0, 15, 1, 14],
        [1, 14, 2, 13],
        [2, 13, 3, 12],
        [3, 12, 4, 11],
        [4, 11, 5, 10],
        [5, 10, 6, 9],
        [6, 9, 7, 8]
    ])
    propeller_base_pairs = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    body_loss_fn = structural.get_structural_loss_fn(
        displacement_fn,
        backbone_dist_pairs,
        helical_pairs,
        pitch_quartets,
        propeller_base_pairs)
    body_loss_fn = jit(loss_fn)
    mapped_body_loss_fn = jit(vmap(body_loss_fn))

    def trajectory_loss_fn(trajectory):
        states_to_eval = trajectory[-3000:][::100]
        body_losses = mapped_body_loss_fn(states_to_eval)
        return jnp.mean(body_losses)

    run_single_simulation = Partial(run_simulation, steps=steps, init_fn=init_fn, step_fn=step_fn)

    def sim_loss_fn(params, key):
        trajectory = run_single_simulation(params, key)
        return trajectory_loss_fn(trajectory)


    # Get our gradients ready
    grad_fn = value_and_grad(sim_loss_fn)
    batched_grad_fn = vmap(grad_fn, (None, 0))

    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)

    # Setup some logging, some required and some not
    params_ = list()
    all_losses = list()
    all_grads = list()

    # Do the optimization
    step_times = list()
    for i in tqdm.trange(opt_steps, position=0):
        start = time.time()
        key, split = random.split(key)

        # Get the grad for our single test case (would have to average for multiple)
        losses, grads = batched_grad_fn(optimizer.params_fn(opt_state), split)
        avg_loss = jnp.mean(losses)
        avg_grad = jnp.mean(grads, axis=0)
        opt_state = optimizer.update_fn(i, avg_grad, opt_state)

        end = time.time()

        if i % save_every == 0:
            step_times.append(end - start)
            # print(optimizer.params_fn(opt_state))

            all_grads.append(grads)
            params_.append(optimizer.params_fn(opt_state))
            all_losses.append(losses)

    with open(run_dir / "final_params.pkl", "wb") as f:
        pickle.dump(optimizer.params_fn(opt_state), f)
    with open(run_dir / "final_loss.pkl", "wb") as f:
        pickle.dump(avg_loss, f)
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

    parser = argparse.ArgumentParser(description="Optimizing over oxDNA parameters")
    parser.add_argument('-b', '--batch-size', type=int, default=10,
                        help="Num. batches for each round of gradient descent")
    parser.add_argument('--save-every', type=int, default=1,
                        help="Frequency of saving data from optimization")
    parser.add_argument('--opt-steps', type=int, default=1,
                        help="Num. iterations of gradient descent")
    parser.add_argument('-n', '--n-steps', type=int, default=10000,
                        help="Num. steps per simulation")
    parser.add_argument('-k', '--key', type=int, default=0,
                        help="Random key")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate for optimization")
    parser.add_argument('--top-path', type=str,
                        default="data/simple-helix/generated.top",
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        default="data/simple-helix/start.conf",
                        help='Path to configuration file')
    args = vars(parser.parse_args())


    # starting with the correct parameters
    init_fene_params = [2.0, 0.25, 0.7525]
    init_stacking_params = [
        1.3448, 2.6568, 6.0, 0.4, 0.9, 0.32, 0.75, # f1(dr_stack)
        1.30, 0.0, 0.8, # f4(theta_4)
        0.90, 0.0, 0.95, # f4(theta_5p)
        0.90, 0.0, 0.95, # f4(theta_6p)
        2.0, -0.65, # f5(-cos(phi1))
        2.0, -0.65 # f5(-cos(phi2))
    ]

    init_params = jnp.array(init_fene_params + init_stacking_params)
    # init_params = jnp.array(init_fene_params)


    start = time.time()
    run(args, init_params=init_params)
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
