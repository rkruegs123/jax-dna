import pdb
import tqdm
import functools
import time
import pickle
from pathlib import Path
import datetime
import shutil

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
    helical_pairs = top_info.bonded_nbrs[1:-1]
    pitch_quartets = jnp.array([
        # [0, 15, 1, 14],
        [1, 14, 2, 13],
        [2, 13, 3, 12],
        [3, 12, 4, 11],
        [4, 11, 5, 10],
        [5, 10, 6, 9],
        # [6, 9, 7, 8]
    ])
    propeller_base_pairs = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    body_loss_fn = structural.get_structural_loss_fn(
        displacement_fn,
        backbone_dist_pairs,
        helical_pairs,
        pitch_quartets,
        propeller_base_pairs)
    body_loss_fn = jit(body_loss_fn)
    mapped_body_loss_fn = jit(vmap(body_loss_fn))


    # Start: added quickly for march meeting
    d = space.map_bond(functools.partial(displacement_fn))
    bb_nbrs_i = backbone_dist_pairs[:, 0]
    bb_nbrs_j = backbone_dist_pairs[:, 1]
    from utils import Q_to_back_base
    @jit
    def bb_dist_fn(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[bb_nbrs_i], back_sites[bb_nbrs_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)
        return r_back
    mapped_bb_dist_fn = jit(vmap(bb_dist_fn))

    hel_bp_i = helical_pairs[:, 0]
    hel_bp_j = helical_pairs[:, 1]
    backbone_radius = 0.675
    @jit
    def helical_dist_fn(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[hel_bp_i], back_sites[hel_bp_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)
        r_back += 2*backbone_radius
        return r_back
    mapped_helical_dist_fn = jit(vmap(helical_dist_fn))

    from loss import pitch
    n_quartets = pitch_quartets.shape[0]
    @jit
    def pitch_fn(body):
        pitches = pitch.get_pitches(body, pitch_quartets)
        num_turns = jnp.sum(pitches) / (2*jnp.pi)
        avg_pitch = (n_quartets+1) / num_turns
        return avg_pitch
    mapped_pitch_fn = jit(vmap(pitch_fn))

    from loss import propeller
    @jit
    def propeller_fn(body):
        avg_p_twist = propeller.get_avg_propeller_twist(body, propeller_base_pairs)
        avg_p_twist_deg = 180.0 - (avg_p_twist * 180.0 / jnp.pi)
        return avg_p_twist_deg
    mapped_propeller_fn = jit(vmap(propeller_fn))

    # End: added quickly for march meeting

    @jit
    def trajectory_loss_fn(trajectory):
        # states_to_eval = trajectory[-3000:][::100]
        states_to_eval = trajectory[-25000:][::100]
        body_losses = mapped_body_loss_fn(states_to_eval)

        avg_bb_dist = jnp.mean(mapped_bb_dist_fn(states_to_eval))
        avg_helical_dist = jnp.mean(mapped_helical_dist_fn(states_to_eval))
        avg_pitch = jnp.mean(mapped_pitch_fn(states_to_eval))
        avg_propeller_twist = jnp.mean(mapped_propeller_fn(states_to_eval))

        return jnp.mean(body_losses), (avg_bb_dist, avg_helical_dist, avg_pitch, avg_propeller_twist)

    run_single_simulation = Partial(run_simulation, steps=sim_length, init_fn=init_fn, step_fn=step_fn)

    @jit
    def sim_loss_fn(params, key):
        trajectory = run_single_simulation(params, key)
        loss, (avg_bb_dist, avg_helical_dist, avg_pitch, avg_propeller_twist) = trajectory_loss_fn(trajectory)
        # return trajectory_loss_fn(trajectory)
        return loss, (avg_bb_dist, avg_helical_dist, avg_pitch, avg_propeller_twist)


    # Get our gradients ready
    grad_fn = value_and_grad(sim_loss_fn, has_aux=True)
    batched_grad_fn = vmap(grad_fn, (None, 0))
    batched_grad_fn = jit(batched_grad_fn)

    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)

    # Setup some logging, some required and some not
    params_ = list()
    all_losses = list()
    all_grads = list()
    loss_path = run_dir / "loss.txt"
    bb_dist_path = run_dir / "bb_dists.txt"
    helical_dist_path = run_dir / "helical_dists.txt"
    pitches_path = run_dir / "pitches.txt"
    propeller_twists_path = run_dir / "propeller_twists.txt"

    # Do the optimization
    step_times = list()
    for i in tqdm.trange(opt_steps, position=0):
        start = time.time()
        key, iter_key = random.split(key)
        seeds = jax.random.split(iter_key, batch_size)

        # Get the grad for our single test case (would have to average for multiple)
        (losses, (bb_dists, helical_dists, pitches, propeller_twists)), grads = batched_grad_fn(optimizer.params_fn(opt_state), seeds)
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

            with open(loss_path, "a") as f:
                f.write(f"{losses}\n")
            with open(bb_dist_path, "a") as f:
                f.write(f"{bb_dists}\nAvg: {jnp.mean(bb_dists)}\n")
            with open(helical_dist_path, "a") as f:
                f.write(f"{helical_dists}\nAvg: {jnp.mean(helical_dists)}\n")
            with open(pitches_path, "a") as f:
                f.write(f"{pitches}\nAvg: {jnp.mean(pitches)}\n")
            with open(propeller_twists_path, "a") as f:
                f.write(f"{propeller_twists}\nAvg: {jnp.mean(propeller_twists)}\n")

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

    from loader import get_params


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
    init_params = get_params.get_init_optimize_params("oxdna")
    init_params = jnp.array(init_params)


    start = time.time()
    run(args, init_params=init_params)
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
