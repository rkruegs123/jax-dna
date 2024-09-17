import webbrowser
import datetime
import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
import subprocess
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp

import jax
import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, lax
from jax_md import space, simulate, rigid_body
import tensorboardX

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna1 import model

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


checkpoint_every = None
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)

layout = {
    "custom": {
        "propeller twist": ["Multiline", ["target", "measured"]],
        "optimization": ["Multiline", ["loss"]],
    },
}
class TensorBoardLogger:
    def __init__(self, log_dir):
        self.logger = tensorboardX.SummaryWriter(str(log_dir), flush_secs=5)
        self.logger.add_custom_scalars(layout)
        self.tensorboard_proc = subprocess.Popen(
            ["tensorboard", "--logdir", str(log_dir)],
            # comment the below lines out if you want to see if tensorboard is saying things:
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open("http://localhost:6006/?darkMode=false#custom_scalars&_smoothingWeight=0")


    def add_scalar(self, tag, scalar_value, global_step):
        self.logger.add_scalar(tag, scalar_value, global_step)

    def close(self):
        self.tensorboard_proc.kill()


def run(args):

    # Load parameters
    n_iters = args['n_iters']
    n_eq_steps = args['n_eq_steps']
    sample_every = args['sample_every']
    lr = args['lr']
    min_neff_factor = args['min_neff_factor']
    n_sample_steps = args['n_sample_steps']
    assert(n_sample_steps % sample_every == 0)
    n_ref_states = n_sample_steps // sample_every
    plot_every = args['plot_every']
    run_name = args['run_name']
    target_ptwist = args['target_ptwist']
    max_approx_iters = args['max_approx_iters']

    opt_keys = args['opt_keys']

    # Setup the logging directoroy
    output_dir = Path("output/")
    run_dir = output_dir / run_name


    layout = {
        "custom": {
            "observable": ["Multiline", ["target", "measured"]],
            "optimization": ["Multiline", ["loss"]],
        },
    }

    # logger = tensorboardX.SummaryWriter(str(run_dir))
    # logger = tf.summary.create_file_writer(str(run_dir))
    # logger.add_custom_scalars(layout)
    # run_dir.mkdir(parents=False, exist_ok=False)
    logger = TensorBoardLogger(run_dir)

    ref_traj_dir = run_dir / "ref_traj"
    ref_traj_dir.mkdir(parents=False, exist_ok=False)

    params_str = ""
    for k, v in args.items():
        params_str += f"{k}: {v}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)


    # Load the system
    sys_basedir = Path("data/sys-defs/simple-helix")
    top_path = sys_basedir / "sys.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "bound_relaxed.conf"
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


    def sim_fn(params, body, n_steps, key):
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

        # Option 1: Scan
        start = time.time()
        fin_state, traj = scan(scan_fn, init_state, jnp.arange(n_steps))
        end = time.time()
        print(f"Generating reference states took {end - start} seconds")

        # Option 2: For loop
        """
        start = time.time()
        trajectory = list()
        state = init_state
        for i in tqdm(range(n_steps)):
            state = step_fn(state,
                            seq=seq_oh,
                            bonded_nbrs=top_info.bonded_nbrs,
                            unbonded_nbrs=top_info.unbonded_nbrs.T)
            trajectory.append(state.position)
        traj = utils.tree_stack(trajectory)
        end = time.time()
        print(f"Generating reference states took {end - start} seconds")
        """


        return traj


    def get_ref_states(params, init_body, key, i):

        iter_dir = ref_traj_dir / f"iter{i}"
        iter_dir.mkdir(parents=False, exist_ok=False)


        trajectory = sim_fn(params, init_body, n_eq_steps + n_sample_steps, key)
        eq_trajectory = trajectory[n_eq_steps:]
        ref_states = eq_trajectory[::sample_every]

        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        energy_fn = jit(energy_fn)

        # ref_energies = [energy_fn(body) for body in ref_states]
        # return ref_states, jnp.array(ref_energies)

        ref_energies = vmap(energy_fn)(ref_states)

        ref_ptwists = vmap(compute_avg_ptwist)(ref_states) # FIXME: this doesn't depend on params...

        n_traj_states = len(ref_ptwists)
        running_avg_ptwists = onp.cumsum(ref_ptwists) / onp.arange(1, n_traj_states + 1)
        # plt.plot(running_avg_ptwists)
        # plt.savefig(iter_dir / f"running_avg.png")
        # plt.close()

        # plt.plot(running_avg_ptwists[-int(n_traj_states // 2):])
        # plt.savefig(iter_dir / f"running_avg_second_half.png")
        # plt.close()

        return ref_states, ref_energies, ref_ptwists


    # Construct the loss function terms

    ## note: we don't include the end base pairs due to fraying
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
    compute_avg_ptwist, ptwist_loss_fn = propeller.get_propeller_loss_fn(simple_helix_bps)


    @jit
    def loss_fn(params, ref_states: rigid_body.RigidBody, ref_energies, ref_ptwists):

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

        # Compute the observable
        expected_ptwist = jnp.dot(weights, ref_ptwists)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        mse = (expected_ptwist - target_ptwist)**2
        rmse = jnp.sqrt(mse)

        return rmse, (n_eff, expected_ptwist)
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    for opt_key in opt_keys:
        params[opt_key] = deepcopy(model.DEFAULT_BASE_PARAMS[opt_key])
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    key = random.PRNGKey(0)

    init_body = conf_info.get_states()[0]
    print(f"Generating initial reference states and energies...")
    ref_states, ref_energies, ref_ptwists = get_ref_states(params, init_body, key, 0)

    min_n_eff = int(n_ref_states * min_neff_factor)
    all_losses = list()
    all_eptwists = list()
    all_ref_losses = list()
    all_ref_eptwists = list()
    all_ref_times = list()

    # loss_path = log_dir / "loss.txt"
    # neff_path = log_dir / "neff.txt"
    # ptwist_path = log_dir / "ptwist.txt"

    num_resample_iters = 0
    for i in tqdm(range(n_iters)):
        (loss, (n_eff, expected_ptwist)), grads = grad_fn(params, ref_states, ref_energies, ref_ptwists)

        if i == 0:
            logger.add_scalar("loss", loss, global_step=i)
            logger.add_scalar("target", target_ptwist, global_step=i)
            logger.add_scalar("measured", expected_ptwist, global_step=i)
            # all_ref_losses.append(loss)
            # all_ref_times.append(i)
            # all_ref_eptwists.append(expected_ptwist)

        if n_eff < min_n_eff or num_resample_iters >= max_approx_iters:
            num_resample_iters

            print(f"Resampling reference states...")
            key, split = random.split(key)
            ref_states, ref_energies, ref_ptwists = get_ref_states(params, ref_states[-1], split, i)
            (loss, (n_eff, expected_ptwist)), grads = grad_fn(params, ref_states, ref_energies, ref_ptwists)


            # all_ref_losses.append(loss)
            # all_ref_eptwists.append(expected_ptwist)
            # all_ref_times.append(i)

        # with open(loss_path, "a") as f:
        #     f.write(f"{loss}\n")
        # with open(ptwist_path, "a") as f:
        #     f.write(f"{expected_ptwist}\n")
        # with open(neff_path, "a") as f:
        #     f.write(f"{n_eff}\n")
        # all_losses.append(loss)
        # all_eptwists.append(expected_ptwist)


        print(f"Loss: {loss}")
        print(f"Effective sample size: {n_eff}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0:
            logger.add_scalar("loss", loss, global_step=i)
            logger.add_scalar("target", target_ptwist, global_step=i)
            logger.add_scalar("measured", expected_ptwist, global_step=i)

        #     # Plot the losses
        #     plt.plot(onp.arange(i+1), all_losses, linestyle="--")
        #     plt.scatter(all_ref_times, all_ref_losses, marker='o', label="Resample points")
        #     plt.xlabel("Iteration")
        #     plt.ylabel("Loss")
        #     plt.legend()
        #     plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
        #     plt.savefig(img_dir / f"losses_iter{i}.png")
        #     plt.clf()

        #     # Plot the persistence lengths
        #     plt.plot(onp.arange(i+1), all_eptwists, linestyle="--", color='blue')
        #     plt.scatter(all_ref_times, all_ref_eptwists, marker='o', label="Resample points", color='blue')
        #     plt.axhline(y=target_ptwist, linestyle='--', label="Target p. twist", color='red')
        #     plt.xlabel("Iteration")
        #     plt.ylabel("Expected Propeller Twist (deg)")
        #     plt.legend()
        #     plt.title(f"DiffTRE Propeller Twist Optimization, Neff factor={min_neff_factor}")
        #     plt.savefig(img_dir / f"eptwists_iter{i}.png")
        #     plt.clf()

    input("Press Enter to kill tensorboard and exit...")
    logger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize structural properties using differentiable trajectory reweighting")

    parser.add_argument('--n-iters', type=int, default=100,
                        help="Number of iterations of gradient descent")
    parser.add_argument('--n-eq-steps', type=int, default=10000,
                        help="Number of equilibration steps")
    parser.add_argument('--n-sample-steps', type=int, default=100000,
                        help="Number of total steps for sampling reference states")
    parser.add_argument('--sample-every', type=int, default=1000,
                        help="Frequency of sampling reference states.")
    parser.add_argument('--plot-every', type=int, default=10,
                        help="Frequency of plotting data from gradient descent epochs")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--min-neff-factor', type=float, default=0.95,
                        help="Factor for determining min Neff")
    parser.add_argument('--target-ptwist', type=float, default=propeller.TARGET_PROPELLER_TWIST,
                        help="Target persistence length in degrees")
    parser.add_argument('--run-name', type=str,
                        help='Run name', default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument('--max-approx-iters', type=int, default=5,
                        help="Maximum number of iterations before resampling")
    parser.add_argument(
        '--opt-keys',
        nargs='*',  # Accept zero or more arguments
        default=["fene", "stacking"],
        help='Parameter keys to optimize'
    )

    args = vars(parser.parse_args())

    run(args)
