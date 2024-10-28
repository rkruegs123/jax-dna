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
from jax_md import space, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna2 import model
from jax_dna import dna2, loss
from jax_dna import integrate

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)



checkpoint_every = 50
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def run(args):


    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_skip_quartets = args['n_skip_quartets']
    small_system = args['small_system']

    slow_diffusion = args['slow_diffusion']

    run_name = args['run_name']

    n_iters = args['n_iters']
    lr = args['lr']
    target_pitch = args['target_pitch']

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


    loss_path = log_dir / "loss.txt"
    times_path = log_dir / "times.txt"
    grads_path = log_dir / "grads.txt"
    pitch_path = log_dir / "pitch.txt"
    params_path = log_dir / "params.txt"

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    if small_system:
        sys_basedir = Path("data/templates/simple-helix-12bp")
    else:
        sys_basedir = Path("data/templates/simple-helix-60bp")
    input_template_path = sys_basedir / "input"

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
    init_body = centered_conf_info.get_states()[0]
    box_size = conf_info.box_size

    displacement_fn, shift_fn = space.free()

    # Get the loss function
    compute_avg_pitch, _ = pitch.get_pitch_loss_fn(
        quartets, displacement_fn, model.com_to_hb)


    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma_eq = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    if slow_diffusion:
        gamma_scale = 2500
    else:
        gamma_scale = 1
    gamma_opt = rigid_body.RigidBody(
        center=gamma_eq.center * gamma_scale,
        orientation=gamma_eq.orientation * gamma_scale)
    mass = rigid_body.RigidBody(center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
                                orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

    def sim_fn(params, body, n_steps, key, gamma):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        # init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_fn, step_fn = integrate.langevin.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        @jit
        def scan_fn(carry, step):
            state, total_log_prob = carry
            state, step_log_prob = step_fn(
                state,
                seq=seq_oh,
                bonded_nbrs=top_info.bonded_nbrs,
                unbonded_nbrs=top_info.unbonded_nbrs.T)
            return (state, total_log_prob + step_log_prob), state.position

        fin_carry, traj = scan(scan_fn, (init_state, 0.0), jnp.arange(n_steps))
        fin_state, total_log_prob = fin_carry
        return fin_state.position, traj, total_log_prob

    eq_fn = lambda params, key: sim_fn(params, init_body, n_eq_steps, key, gamma_eq)
    eq_fn = jit(eq_fn)

    @jit
    def loss_fn(params, eq_body, key):
        fin_pos, traj, traj_log_prob = sim_fn(params, eq_body, n_steps_per_sim, key, gamma_opt)
        states_to_eval = traj[::sample_every]
        pitches = vmap(compute_avg_pitch)(states_to_eval)
        avg_pitch = pitches.mean()
        rmse = jnp.sqrt((avg_pitch - target_pitch)**2)
        return traj_log_prob*jax.lax.stop_gradient(rmse), avg_pitch
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    batched_grad_fn = jit(vmap(grad_fn, (None, 0, 0)))

    params = deepcopy(model.EMPTY_BASE_PARAMS)
    # params["fene"] = model.default_base_params_seq_avg["fene"]
    params["stacking"] = model.default_base_params_seq_avg["stacking"]
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    key = random.PRNGKey(0)
    mapped_eq_fn = jit(vmap(eq_fn, (None, 0)))


    for i in tqdm(range(n_iters)):
        key, iter_key = random.split(key)
        iter_key, eq_key = random.split(iter_key)
        eq_keys = random.split(eq_key, n_sims)
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        eq_bodies, _, _ = mapped_eq_fn(params, eq_keys)

        batch_keys = random.split(iter_key, n_sims)
        start = time.time()
        (losses, avg_pitches), grads = batched_grad_fn(params, eq_bodies, batch_keys)
        end = time.time()
        iter_time = end - start

        avg_grads = tree_util.tree_map(jnp.mean, grads)
        avg_pitch = jnp.mean(avg_pitches)

        with open(loss_path, "a") as f:
            f.write(f"{jnp.mean(losses)}\n")

        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(avg_grads, indent=4)}\n")

        with open(params_path, "a") as f:
            f.write(f"{pprint.pformat(params, indent=4)}\n")

        with open(pitch_path, "a") as f:
            f.write(f"{avg_pitch}\n")

        updates, opt_state = optimizer.update(avg_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    return


def get_parser():
    parser = argparse.ArgumentParser(description="Optimize structural properties in oxDNA2 via differentiating through the trajectory.")

    parser.add_argument('--run-name', type=str, help='Run name')

    parser.add_argument('--n-steps-per-sim', type=int, default=100000,
                        help="Number of steps for sampling reference states per simulation")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--sample-every', type=int, default=100,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--n-sims', type=int, default=10,
                        help="Number of individual simulations, i.e. batch size")

    parser.add_argument('--target-pitch', type=float, default=pitch.TARGET_AVG_PITCH,
                        help="Target pitch in number of bps")
    parser.add_argument('--n-iters', type=int, default=200,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--n-skip-quartets', type=int, default=5,
                        help="Number of quartets to skip on either end of the duplex")

    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")

    parser.add_argument('--slow-diffusion', action='store_true')
    parser.add_argument('--small-system', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
