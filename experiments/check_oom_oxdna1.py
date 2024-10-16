import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import argparse

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.loss import pitch
from jax_dna.dna1 import model
from jax_dna import dna1, loss
from jax_dna.loss import geometry, pitch, propeller

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)




def run(args):
    run_name = args['run_name']
    n_skip_quartets = args['n_skip_quartets']
    init_hi = args['hi']
    init_lo = args['lo']

    # Setup the logging directory
    if run_name is None:
        raise RuntimeError(f"Must set run name")
    output_dir = Path("output/")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)

    trajs_dir = run_dir / "trajs"
    trajs_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)

    log_path = run_dir / "log.txt"

    displacement_fn, shift_fn = space.free()


    # Load the system
    sys_basedir = Path("data/sys-defs/simple-helix")

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    compute_helical_diameters, helical_diam_loss_fn = geometry.get_helical_diameter_loss_fn(
        top_info.bonded_nbrs[1:-1], displacement_fn, model.com_to_backbone)

    compute_bb_distances, bb_dist_loss_fn = geometry.get_backbone_distance_loss_fn(
        top_info.bonded_nbrs, displacement_fn, model.com_to_backbone)

    simple_helix_quartets = jnp.array([
        [1, 14, 2, 13], [2, 13, 3, 12],
        [3, 12, 4, 11], [4, 11, 5, 10],
        [5, 10, 6, 9]])
    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        simple_helix_quartets, displacement_fn, model.com_to_hb)

    simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12],
                                  [4, 11], [5, 10], [6, 9]])
    compute_avg_p_twist, p_twist_loss_fn = propeller.get_propeller_loss_fn(simple_helix_bps)

    conf_path = sys_basedir / "bound_relaxed.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    init_body = conf_info.get_states()[0]
    box_size = conf_info.box_size


    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    # gamma_scale = 2500
    gamma_scale = 1
    gamma = rigid_body.RigidBody(
        center=gamma.center * gamma_scale,
        orientation=gamma.orientation * gamma_scale)
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    checkpoint_every = None
    if checkpoint_every is None:
        scan = lax.scan
    else:
        scan = functools.partial(checkpoint.checkpoint_scan,
                                 checkpoint_every=checkpoint_every)


    sample_every = 1000
    def body_metadata_fn(body):
        helical_diams = compute_helical_diameters(body)
        mean_helical_diam = jnp.mean(helical_diams)

        bb_dists = compute_bb_distances(body)
        mean_bb_dist = jnp.mean(bb_dists)

        mean_pitch = compute_avg_pitch(body)

        mean_p_twist = compute_avg_p_twist(body)

        return (mean_helical_diam, mean_bb_dist, mean_pitch, mean_p_twist)


    @jit
    def body_loss_fn(body):
        loss = helical_diam_loss_fn(body) + bb_dist_loss_fn(body) \
               + pitch_loss_fn(body) + p_twist_loss_fn(body)
        return loss, body_metadata_fn(body)

    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses, all_metadata = vmap(body_loss_fn)(states_to_eval)
        return losses.mean(), all_metadata

    def sim_fn(params, body, n_steps, key, gamma):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        @jit
        def scan_fn(state, step):
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=top_info.unbonded_nbrs.T)
            return state, state.position

        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj


    def check_oom(n_steps, checkpoint_every):

        if checkpoint_every is None:
            scan = lax.scan
        else:
            scan = functools.partial(checkpoint.checkpoint_scan,
                                     checkpoint_every=checkpoint_every)

        @jit
        def loss_fn(params, eq_body, key):
            fin_pos, traj = sim_fn(params, eq_body, n_steps, key, gamma)
            loss, metadata = traj_loss_fn(traj)
            return loss, metadata
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        grad_fn = jit(grad_fn)

        params = deepcopy(model.EMPTY_BASE_PARAMS)
        params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
        params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]

        key = random.PRNGKey(0)
        try:
            start = time.time()
            (loss, metadata), grads = grad_fn(params, init_body, key)
            end = time.time()
            first_eval_time = end - start

            grad_vals = [v for v in grads['stacking'].values()]

            pdb.set_trace()

            traj_info = trajectory.TrajectoryInfo(
                top_info, read_from_states=True, states=traj, box_size=conf_info.box_size)
            traj_info.write(trajs_dir / f"traj_c{checkpoint_every}_n{n_steps}.dat", reverse=True)

            start = time.time()
            (loss, traj), grads = grad_fn(params, init_body, key)
            end = time.time()
            second_eval_time = end - start

            print("success")

            failed = False
        except Exception as e:
            print(e)
            print("failed")

            first_eval_time = -1
            second_eval_time = -1

            failed = True
        return failed, first_eval_time, second_eval_time


    with open(log_path, "a") as f:
        f.write(f"Checkpoint every: {checkpoint_every}\n")

    hi = init_hi
    lo = init_lo
    curr = (hi - lo) // 2

    found_max = False
    while not found_max:

        print(f"curr: {curr}")
        print(f"lo: {lo}")
        print(f"hi: {hi}")

        failed, first_eval_time, second_eval_time = check_oom(curr*sample_every, checkpoint_every)
        with open(log_path, "a") as f:
            f.write(f"- # 1000 steps: {curr}\n")
            f.write(f"\t- failed: {failed}\n")
            f.write(f"\t- 1st eval time: {first_eval_time}\n")
            f.write(f"\t- 2nd eval time: {second_eval_time}\n")
        if failed:
            pdb.set_trace()
            hi = curr
            curr = lo + (hi - lo) // 2
        else:
            lo = curr
            curr = lo + (hi - lo) // 2

        if curr == lo:
            found_max = True

    return curr




def get_parser():
    parser = argparse.ArgumentParser(description="Check OOM for oxDNA simulations in JAX-MD")

    parser.add_argument('--run-name', type=str, help='Run name')
    parser.add_argument('--n-skip-quartets', type=int, default=1,
                        help="Number of quartets to skip on either end of the duplex")
    parser.add_argument('--hi', type=int, default=500,
                        help="Maximum # thousand steps for binary search")
    parser.add_argument('--lo', type=int, default=0,
                        help="Minimum # thousand steps for binary search")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
