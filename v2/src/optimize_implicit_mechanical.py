import pdb
import tqdm
import functools
import time
import pickle
from pathlib import Path
import datetime
import shutil
from functools import partial

import jax
import jax.debug
from jax import jit, vmap, lax, random, value_and_grad
from jax.config import config as jax_config
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt # FIXME: change to optax
from pprint import pprint

from jax_md import space, util, simulate, quantity
from jax_md.rigid_body import RigidBody, Quaternion
from jax_md.rigid_body import _quaternion_multiply, _quaternion_conjugate
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
from loss.ext_modulus import compute_dist

from jax.config import config
config.update("jax_enable_x64", True)

from jaxopt.implicit_diff import custom_root


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


checkpoint_every = None
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)

def run_simulation(params, key, steps, init_fn, step_fn):

    init_state = init_fn(key, params=params)
    @jit
    def scan_fn(state, step):
        state = step_fn(state, params=params)
        # return state, state.position
        return state, None
    # final_state, trajectory = scan(scan_fn, init_state, jnp.arange(steps))
    final_state, _ = scan(scan_fn, init_state, jnp.arange(steps))

    return final_state


def run(args, init_params,
        T=DEFAULT_TEMP, dt=5e-3,
        output_basedir="v2/data/output"):

    top_path = args['top_path']
    conf_path = args['conf_path']
    opt_steps = args['opt_steps']
    lr = args['lr']
    key = random.PRNGKey(args['key'])

    n_steps_per_force = args['n_steps_per_force']
    sample_every = args['sample_every']
    n_eq_steps = args['n_eq_steps']

    assert(n_steps_per_force % sample_every == 0)

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    output_basedir = Path(output_basedir)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"optimize_implicit_mech_{timestamp}_lr{lr}_n{n_steps_per_force}_s{sample_every}_e{n_eq_steps}"
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

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.free()

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT
    gamma = RigidBody(center=jnp.array([kT/2.5]), orientation=jnp.array([kT/7.5]))

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                             back_site, stack_site, base_site,
                                             top_info.bonded_nbrs, top_info.unbonded_nbrs)
    ext_force_magnitude = 0.025
    ext_force_bps1 = [5, 214]
    ext_force_bps2 = [104, 115]

    _, force_fn = ext_force.get_force_fn(energy_fn, top_info.n, displacement_fn,
                                         ext_force_bps1,
                                         [0, 0, ext_force_magnitude], [0, 0, 0, 0])
    _, force_fn = ext_force.get_force_fn(force_fn, top_info.n, displacement_fn,
                                             ext_force_bps2,
                                             [0, 0, -ext_force_magnitude], [0, 0, 0, 0])
    force_fn = jit(force_fn)

    def adj_force_fn(body, params):
        q = body.orientation.vec
        q_conj = _quaternion_conjugate(q)
        base_force = force_fn(body, params=params)
        base_force_q = base_force.orientation.vec
        orientation_force_adjustment = _quaternion_multiply(
            _quaternion_multiply(q_conj, base_force_q), q)
        adj_orientation_force = base_force_q - orientation_force_adjustment
        return RigidBody(center=base_force.center,
                         orientation=Quaternion(adj_orientation_force))
    batched_adj_force_fn = vmap(adj_force_fn, (0, None))
    batched_adj_force_fn = jit(batched_adj_force_fn)

    @jit
    def mean_force_fn_helper(bodies, params):
        batch_forces = batched_adj_force_fn(bodies, params)
        mean_center_force = jnp.mean(batch_forces.center, axis=0)
        mean_orientation_force = jnp.mean(batch_forces.orientation.vec, axis=0)
        return mean_center_force, mean_orientation_force

    @jit
    def mean_force_fn(bodies, params):
        n_states = bodies.center.shape[0]
        mean_center_force, mean_orientation_force = mean_force_fn_helper(bodies, params)
        mean_center_force_copied = jnp.tile(mean_center_force, (n_states, 1, 1))
        mean_orientation_force_copied = jnp.tile(mean_orientation_force, (n_states, 1, 1))
        zeros = RigidBody(center=mean_center_force_copied,
                          orientation=Quaternion(mean_orientation_force_copied))
        return zeros


    init_fn, step_fn = simulate.nvt_langevin(force_fn, shift_fn, dt, kT, gamma)
    step_fn = jit(step_fn)
    equilibrate_init_fn = Partial(init_fn, R=body, mass=mass)
    equilibrate_init_fn = jit(equilibrate_init_fn)
    run_equilibration = Partial(run_simulation, steps=n_eq_steps,
                                init_fn=equilibrate_init_fn, step_fn=step_fn)
    run_equilibration = jit(run_equilibration)


    n_sample_runs = n_steps_per_force // sample_every
    dir_end2 = (104, 115)
    dir_end1 = (5, 214)
    dir_force_axis = jnp.array([0, 0, 1])
    mapped_compute_dist = vmap(Partial(compute_dist,
                                       end1=dir_end1, end2=dir_end2,
                                       force_axis=dir_force_axis))
    mapped_compute_dist = jit(mapped_compute_dist)

    @jit
    def get_states_to_eval(key, params):
        eq_state = run_equilibration(params, key)

        @jit
        def scan_fn(sample_start_state, n_sample):
            sample_init_fn = Partial(init_fn, R=sample_start_state_state.position, mass=mass)
            end_state = run_simulation(params, key, sample_every, sample_init_fn, step_fn)
            return end_state, end_state.position # We accumulate the end states and also use the the current end state as the next start state

        _, all_samples = scan(scan_fn, eq_state, jnp.arange(n_sample_runs))
        return all_samples # a RigidBody
    get_states_to_eval = custom_root(mean_force_fn)(get_states_to_eval)

    target_pdist = 25.0
    @jit
    def eval_params(params, key):
        states_to_eval = get_states_to_eval(key, params)

        # For now, just target a particular length for a single force. If we can do this, then we can pmap
        p_dists = mapped_compute_dist(states_to_eval)
        avg_pdist = jnp.mean(p_dists)
        return (avg_pdist - target_pdist)**2

    grad_fn = value_and_grad(eval_params)

    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)

    for i in tqdm.trange(opt_steps, position=0):
        key, iter_key = random.split(key)
        curr_params = optimizer.params_fn(opt_state)
        loss, grad = grad_fn(curr_params, iter_key)
        opt_state = optimizer.update_fn(i, grad, opt_state)

        print(f"Iteration {i}: {loss}")
    fin_params = optimizer.params_fn(opt_state)
    return fin_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimizing over oxDNA parameters")
    parser.add_argument('--opt-steps', type=int, default=1,
                        help="Num. iterations of gradient descent")
    parser.add_argument('-e', '--n-eq-steps', type=int, default=100000,
                        help="Num. steps to equilibrate for a given force")
    parser.add_argument('-n', '--n-steps-per-force', type=int, default=10000000,
                        help="Total num. steps per force")
    parser.add_argument('-s', '--sample-every', type=int, default=10000,
                        help="Frequency of sampling equilibrium states")
    parser.add_argument('-k', '--key', type=int, default=0,
                        help="Random key")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="Learning rate for optimization")
    parser.add_argument('--top-path', type=str,
                        default="data/elastic-mod/generated.top",
                        help='Path to topology file')
    parser.add_argument('--conf-path', type=str,
                        default="data/elastic-mod/generated.dat",
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
