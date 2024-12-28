import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import argparse
import numpy as onp
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint, center_configuration, gradient_clip
from jax_dna.loss import pitch2, pitch
from jax_dna.dna1 import model as model1
from jax_dna.dna2 import model
from jax_dna import dna2, loss



def run(args):

    n_sims = args['n_sims']
    n_steps_per_sim = args['n_steps_per_sim']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    assert(n_steps_per_sim % sample_every == 0)
    n_skip_quartets = args['n_skip_quartets']
    small_system = args['small_system']
    max_norm = args['max_norm']

    clip_every = args['clip_every']
    checkpoint_every = args['checkpoint_every']
    if checkpoint_every is None:
        scan = jax.lax.scan
    else:
        scan = functools.partial(checkpoint.checkpoint_scan,
                                 checkpoint_every=checkpoint_every)

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

    grad_clip_fn = gradient_clip.get_clip_grad_fn("norm", max_norm)

    def sim_fn(params, body, n_steps, key, gamma):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        init_fn, step_fn = simulate.nvt_langevin(em.energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass, seq=seq_oh,
                             bonded_nbrs=top_info.bonded_nbrs,
                             unbonded_nbrs=top_info.unbonded_nbrs.T)

        @jit
        def scan_fn(state, step):
            state = grad_clip_fn(step % clip_every == 0, state)
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=top_info.unbonded_nbrs.T)
            return state, state.position

        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        return fin_state.position, traj

    eq_fn = lambda params, key: sim_fn(params, init_body, n_eq_steps, key, gamma_eq)
    eq_fn = jit(eq_fn)


    pitch_angles_fn = lambda body: pitch2.get_all_angles(body, quartets, displacement_fn, model.com_to_hb, model1.com_to_backbone, 0.0)
    compute_traj_angles = vmap(pitch_angles_fn)
    batch_sim_fn = vmap(sim_fn, (None, 0, None, 0, None))
    def loss_fn(params, eq_bodies, key):
        batch_keys = random.split(key, n_sims)
        all_fin_pos, all_trajs = batch_sim_fn(params, eq_bodies, n_steps_per_sim, batch_keys, gamma_opt)
        all_traj_angles = vmap(compute_traj_angles)(all_trajs)
        avg_pitch_angle = all_traj_angles.mean()
        pitch = (2*jnp.pi) / avg_pitch_angle

        rmse = jnp.sqrt((pitch - target_pitch)**2)
        return rmse, (pitch, all_traj_angles)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)




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
        eq_bodies, _ = mapped_eq_fn(params, eq_keys)

        start = time.time()
        (rmse, aux), grads = grad_fn(params, eq_bodies, iter_key)
        avg_pitch, all_traj_angles = aux
        end = time.time()
        iter_time = end - start

        all_angles = all_traj_angles.flatten()
        running_avg_angles = onp.cumsum(all_angles) / onp.arange(1, all_angles.shape[0] + 1)
        running_avg_pitches = 2*onp.pi / running_avg_angles
        plt.plot(running_avg_pitches)
        plt.savefig(img_dir / f"running_avg_i{i}.png")
        plt.close()

        with open(loss_path, "a") as f:
            f.write(f"{rmse}\n")

        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(grads, indent=4)}\n")

        with open(params_path, "a") as f:
            f.write(f"{pprint.pformat(params, indent=4)}\n")

        with open(pitch_path, "a") as f:
            f.write(f"{avg_pitch}\n")

        updates, opt_state = optimizer.update(grads, opt_state, params)
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

    parser.add_argument('--clip-every', type=int, default=100,
                        help="Frequency of gradient clipping")
    parser.add_argument('--checkpoint-every', type=int, default=10,
                        help="Frequency of gradient checkpointing")

    parser.add_argument('--max-norm', type=float, default=0.1, help="Max norm for normalization")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    run(args)
