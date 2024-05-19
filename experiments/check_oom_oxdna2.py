import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import argparse

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.loss import pitch
from jax_dna.dna2 import model
from jax_dna import dna2, loss

from jax.config import config
config.update("jax_enable_x64", True)




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


    # Load the system
    sys_basedir = Path("data/templates/simple-helix-12bp")

    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=False)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    quartets = utils.get_all_quartets(n_nucs_per_strand=seq_oh.shape[0] // 2)
    quartets = quartets[n_skip_quartets:-n_skip_quartets]

    conf_path = sys_basedir / "init.conf"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path,
        # reverse_direction=True
        reverse_direction=False
    )
    centered_conf_info = center_configuration.center_conf(top_info, conf_info)
    init_body = conf_info.get_states()[0]
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    beta = 1 / kT
    gamma = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    gamma_scale = 2500
    gamma = rigid_body.RigidBody(
        center=gamma.center * gamma_scale,
        orientation=gamma.orientation * gamma_scale)
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))


    compute_avg_pitch, pitch_loss_fn = pitch.get_pitch_loss_fn(
        quartets, displacement_fn, model.com_to_hb)
    
    sample_every = 1000
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        losses = vmap(pitch_loss_fn)(states_to_eval)
        return losses.mean()

    def check_oom(n_steps, checkpoint_every):
        
        if checkpoint_every is None:
            scan = lax.scan
        else:
            scan = functools.partial(checkpoint.checkpoint_scan,
                                     checkpoint_every=checkpoint_every)

        def loss_fn(params, eq_body, key):

            # Run the simulation
            em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
            init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
            init_state = init_fn(key, eq_body, mass=mass, seq=seq_oh,
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
            loss = traj_loss_fn(traj)

            return loss, traj[::sample_every]
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        grad_fn = jit(grad_fn)

        params = deepcopy(model.EMPTY_BASE_PARAMS)
        params["stacking"] = model.default_base_params_seq_avg["stacking"]

        key = random.PRNGKey(0)
        try:
            start = time.time()
            (loss, traj), grads = grad_fn(params, init_body, key)
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
            
    checkpoint_freqs = [25]
    # checkpoint_freqs = [None]
    all_max_n = list()
    for checkpoint_every in tqdm(checkpoint_freqs):

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
        all_max_n.append(curr)

    print(all_max_n)
    return all_max_n

        


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
