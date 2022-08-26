import pdb

import tqdm
import functools

import jax
from jax import jit, vmap, lax, random
from jax.config import config as jax_config
import jax.numpy as jnp
from jax.example_libraries import optimizers as jopt

from jax_md import simulate
from jax_md import space
from jax_md import util
from jax_md.rigid_body import RigidBody, Quaternion

from utils import DEFAULT_TEMP
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import nucleotide_mass, get_kt, moment_of_inertia
from utils import get_one_hot

from get_params import get_default_params
from trajectory import TrajectoryInfo
from topology import TopologyInfo
from energy import energy_fn_factory

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


## simulator function that returns loss
def run_simulation(params, key, displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):
    """
    currently, this is a dummy function whose "loss" is the end-to-end distance between nucleotides
    """
    pdb.set_trace()
    body = config_info.states[0]

    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    n = top_info.n

    energy_fn = energy_fn_factory(displacement_fn,
                                  back_site, stack_site, base_site,
                                  top_info.bonded_nbrs, top_info.unbonded_nbrs)


    # Simulate with the energy function via Nose-Hoover
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)

    # step_fn = jit(step_fn)

    state = init_fn(key, body, mass=mass, seq=seq, params=params)
    E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, seq=seq, params=params)
    """
    ### lax scan version of simulation

    # @jit
    def scan_fn(state, step):
        pdb.set_trace()
        state = step_fn(state, seq=seq, params=params)
        log_probs = 1.0 # dummy for now; need to update rigidbody state and the integrator to return log_prob
        loss = state.position.center[0][0]
        return state, (state, log_probs, loss)

    # state, (_, _, _) = lax.scan(step_fn, state, eq_steps)
    state, (trajectory, log_probs, loss) = lax.scan(scan_fn, state, jnp.arange(steps))
    """
    pdb.set_trace()
    trajectory = [state.position]
    log_probs = 1.0
    loss = 1.0
    for i in range(steps):
        state = step_fn(state, seq=seq, params=params)
        trajectory.append(state.position)

    return trajectory, log_probs, loss

## single gradient estimator
def single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):
    @functools.partial(jax.value_and_grad, has_aux=True)
    def _single_estimate(params, seed): # function only of the params to be differentiated w.r.t.
        pdb.set_trace()
        trajectory, log_probs, loss = run_simulation(params, seed, displacement_fn, shift_fn, top_info,
                                                     config_info, steps, dt=5e-3, T=DEFAULT_TEMP)
        tot_log_prob = log_probs.sum()
        avg_loss = jnp.mean(loss)
        gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(avg_loss) + avg_loss)
        return gradient_estimator, avg_loss
    return _single_estimate

## mapped gradient estimate
def estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):
    # mapped_estimate = jax.vmap(single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP), [None, 0])
    my_fun = single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP)
    def mapped_estimate(params, keys):
        results = list()
        for k in keys:
            results.append(my_fun(params, k))
        return results

    # @jax.jit
    def _estimate_gradient(params, seed):
        seeds = jax.random.split(seed, batch_size)
        pdb.set_trace()
        (gradient_estimator, avg_loss), grad = mapped_estimate(params, seeds)
        pdb.set_trace()
        avg_grad = {}
        for i in grad:
            avg_grad[i] = {k:jnp.mean(v) for k,v in grad[i].items()}

        return avg_grad, (gradient_estimator, avg_loss)
    return _estimate_gradient

"""
test function

###TEST#####

key = random.PRNGKey(0)
displacement_fn, shift_fn = space.free()
conf_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/data/simple-helix/start.conf"
top_path = "/Users/megancengel/Research_apps/jaxmd-oxdna/data/simple-helix/generated.top"

top_info = TopologyInfo(top_path, reverse_direction=True)
config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
body = config_info.states[0]
print(body.center,body.orientation)
seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
n = top_info.n

energy_fn = energy_fn_factory(displacement_fn,
                              back_site, stack_site, base_site,
                              top_info.bonded_nbrs, top_info.unbonded_nbrs)


# Simulate with the energy function via Nose-Hoover
kT = get_kt(t=DEFAULT_TEMP) # 300 Kelvin = 0.1 kT
dt=5e-3
params=get_default_params()
init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)

step_fn = jit(step_fn)

state = init_fn(key, body, mass=mass, seq=seq, params=params)
E_initial = simulate.nvt_nose_hoover_invariant(energy_fn, state, kT, seq=seq, params=params)

pdb.set_trace()
key = random.PRNGKey(0)
print(state.position[0].shape)

"""

def run(top_path="data/simple-helix/generated.top", conf_path="data/simple-helix/start.conf"):
    ## optimization training loop


    ## information for a single "test case"
    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)
    sim_length = 2
    batch_size = 2
    grad_fxn = estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info,
                                 sim_length, dt=5e-3, T=DEFAULT_TEMP)

    ## initialize values relevant for the optimization loop
    init_params = get_default_params()
    opt_steps = 1
    lr = jopt.exponential_decay(0.1, opt_steps, 0.01)
    optimizer = jopt.adam(lr)
    init_state = optimizer.init_fn(init_params)
    opt_state = optimizer.init_fn(init_params)
    key = random.PRNGKey(0)


    params_ = []
    losses = []
    grads = []
    save_every = 1
    params_.append((0,) + (optimizer.params_fn(opt_state),))

    for j in tqdm.trange(opt_steps, position=0):
        key, split = random.split(key)
        grad, (_, avg_loss) = grad_fxn(optimizer.params_fn(opt_state), split)
        opt_state = optimizer.update_fn(j, grad, opt_state)
        losses.append(avg_loss)
        grads.append(grad)
        if j % save_every == 0 | j == (opt_steps-1):
            coeffs_.append(((j+1),) + (optimizer.params_fn(opt_state),))

if __name__ == "__main__":
    run()
