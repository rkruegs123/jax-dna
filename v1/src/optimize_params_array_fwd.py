import pdb

import tqdm
import functools
from pprint import pprint
import time

import jax
from jax import jit, vmap, lax, random
from jax.config import config as jax_config
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md.rigid_body import RigidBody, Quaternion
from jax.tree_util import Partial

from utils import DEFAULT_TEMP
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import get_one_hot

from get_params import get_default_params
from trajectory import TrajectoryInfo
from topology import TopologyInfo
# from energy import energy_fn_factory
from energy_mini import energy_fn_factory
import langevin
from checkpoint import checkpoint_scan
from gradients import value_and_jacfwd

from jax.config import config
config.update("jax_enable_x64", True)
# config.update('jax_disable_jit', True)


f64 = util.f64

mass = RigidBody(center=jnp.array([nucleotide_mass]), orientation=jnp.array([moment_of_inertia]))
gamma = RigidBody(center=jnp.array([DEFAULT_TEMP/2.5]), orientation=jnp.array([DEFAULT_TEMP/7.5]))
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


# Simulator function that returns loss
# Note: for now, we use a dummy loss
def run_simulation(params, key, steps, init_fn, step_fn):

    init_state = init_fn(key, params=params)
    # Take steps with `lax.scan`
    @jit
    def scan_fn(state, step):
        state = step_fn(state, params=params)
        log_probs = 1.0 # dummy for now; need to update rigidbody state and the integrator to return log_prob
        loss = state.position.center[0][0]
        return state, (state.position, log_probs, loss)
    final_state, (trajectory, log_probs, losses) = scan(scan_fn, init_state, jnp.arange(steps))

    avg_loss = jnp.mean(losses)
    return trajectory, log_probs, avg_loss


def single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    energy_fn = energy_fn_factory(displacement_fn,
                                  back_site, stack_site, base_site,
                                  top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq)

    # Langevin
    init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    # Nose Hoover
    # init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)

    step_fn = jit(step_fn)

    init_fn = Partial(init_fn, R=body, mass=mass)
    init_fn = jit(init_fn)

    run_single_simulation = Partial(run_simulation, steps=steps, init_fn=init_fn, step_fn=step_fn)

    @functools.partial(value_and_jacfwd)
    def _single_estimate(params, seed): # function only of the params to be differentiated w.r.t.
        trajectory, log_probs, avg_loss = run_single_simulation(params, seed)
        tot_log_prob = log_probs.sum()
        # FIXME: the naming here is confusing. `gradient_estimator` is actually not the gradient estimator -- it's gradient is the gradient estimator
        gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(avg_loss) + avg_loss)
        return gradient_estimator
    return _single_estimate


def estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):
    single_estimate_fn = single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP)

    mapped_estimate = jax.vmap(single_estimate_fn, [None, 0])

    @jit
    def _estimate_gradient(params, seed):
        seeds = jax.random.split(seed, batch_size)
        grad, gradient_estimator = mapped_estimate(params, seeds)
        avg_grad = jnp.mean(grad)
        return avg_grad, gradient_estimator
    return _estimate_gradient


def run(top_path="data/simple-helix/generated.top", conf_path="data/simple-helix/start.conf"):

    # Information for a single "test case"
    # Note: in the future, we will have multiple of these
    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)
    sim_length = 1000
    batch_size = 3
    # Note how we get one `grad_fxn` per "test case." The gradient has to be estimated *per* test case
    grad_fxn = estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info,
                                 sim_length, dt=5e-3, T=DEFAULT_TEMP)

    # Initialize values relevant for the optimization loop
    init_params = jnp.array([
        2.0, 0.25, 0.7525, # FENE
        2.0, 0.32, 0.33, 0.50, 0.515, 0.50, 0.515, # excluded volume bonded
        0.32, 0.75, 1.61, 6.0, 0.4, 0.9, # stacking
        0.0, 0.8, 1.30,
        0.0, 0.95, 0.90,
        0.0, 0.95, 0.90,
        -0.65, 2.00,
        -0.65, 2.00
    ], dtype=f64)
    opt_steps = 5
    lr = jopt.exponential_decay(0.01, opt_steps, 0.001)
    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)
    key = random.PRNGKey(0)

    # Setup some logging, some required and some not
    grads = list()
    save_every = 1

    # Do the optimization
    step_times = list()
    for i in tqdm.trange(opt_steps, position=0):
        start = time.time()
        key, split = random.split(key)

        # Get the grad for our single test case (would have to average for multiple)
        grad, grad_estimator = grad_fxn(optimizer.params_fn(opt_state), split)
        opt_state = optimizer.update_fn(i, grad, opt_state)
        grads.append(grad)
        end = time.time()
        step_times.append(end - start)
    pprint(step_times)

if __name__ == "__main__":
    import time

    start = time.time()
    run()
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
