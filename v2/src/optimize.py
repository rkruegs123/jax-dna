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
# import langevin
from checkpoint import checkpoint_scan
from loss import geometry
from loss import structural

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


checkpoint_every = 1
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint_scan,
                             checkpoint_every=checkpoint_every)


# Simulator function that returns loss
# Note: for now, we use a dummy loss
def run_simulation(params, key, steps, init_fn, step_fn, loss_fn):

    init_state = init_fn(key, params=params)
    # Take steps with `lax.scan`
    @jit
    def scan_fn(state, step):
        state = step_fn(state, params=params)
        log_probs = 1.0 # dummy for now; need to update rigidbody state and the integrator to return log_prob
        # loss = state.position.center[0][0]
        loss = loss_fn(state.position)
        return state, (state.position, log_probs, loss)
    final_state, (trajectory, log_probs, losses) = scan(scan_fn, init_state, jnp.arange(steps))

    avg_loss = jnp.mean(losses[-100:]) # note how much repeat ocmputation we do given that we only wnat some of them
    return trajectory, log_probs, avg_loss

"""
Single gradient estimator:

For a given "test case" (i.e. a simulation configuration and a loss that is a function
of the trajectory), we want to get the gradient of the loss with respect to the
parameters. As discussed in `estimate_gradient`, this requires running multiple simulations.

`single_estimate` (or rather, _single_estimate decorated with `value_and_grad`) takes as
input the requisite information for test case and returns a function that (i) runs a
single simulation and (ii) returns the component of that simulation required for estimating
the gradient from the entire batch of trajectories.

More specifically, suppose we define a stochastic simulation S and also some function L(P)
that denotes returns a scalar value (i.e. a loss) on a trajectory generated with parameters
P. This is a "test case." We want `grad(<L(P)>)` where <L(P)> is the average of the loss
over all of phase space. However, we can't compute the loss over all of phase space.
So, we can approximate this gradient with the following expression (derivation not shown):

`grad(<L(P)>)` = `<grad(ln(pr))*L(P)> + <grad(L(P))>` = `<grad(ln(pr))*L(P) + grad(L(P))>`

We can rewrite this as

`grad(<L(P)>)` = `<A> + <B>` = `<A + B>`

where

A = grad(ln(pr))*L(P)
B = grad(L(P))

`single_estimate` returns a function, `_single_estimate` that takes as input some parameters
P, as well as a seed, and returns the value `A + B`. This is useful because given all values
`A + B` for every trajectory, we can immediately compute `<A + B>`

So, since we decorate `_single_estimate` with `value_and_grad` and supply an auxiliary value
(i.e. an additional return value for which we do not return the value and gradient),
`_single_estimate(P, seed)` will return

`(ln(pr)*L(P) + L(P), L(P)), A + B = grad(ln(pr))*L(P) + grad(L(P))`

Written differently, `_single_estimate(P, seed)` will return

`(V, L(P)), G`

where `G = A + B` and `V` is the original value for which the gradient was taken to obtain
`G = A+B`. Note that the auxiliary value get's combined with the value as a tuple.
"""
def single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):

    body = config_info.states[0]
    seq = jnp.array(get_one_hot(top_info.seq), dtype=f64)
    kT = get_kt(t=T) # 300 Kelvin = 0.1 kT

    energy_fn, _ = factory.energy_fn_factory(displacement_fn,
                                         back_site, stack_site, base_site,
                                         top_info.bonded_nbrs, top_info.unbonded_nbrs)
    energy_fn = Partial(energy_fn, seq=seq)

    # Langevin
    # init_fn, step_fn = langevin.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
    init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)

    # Nose Hoover
    # init_fn, step_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)

    step_fn = jit(step_fn)

    init_fn = Partial(init_fn, R=body, mass=mass)
    init_fn = jit(init_fn)

    # loss_fn = geometry.get_backbone_distance_loss(top_info.bonded_nbrs, displacement_fn)
    # loss_fn = jit(loss_fn)
    backbone_dist_pairs = top_info.bonded_nbrs
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
    loss_fn = structural.get_structural_loss_fn(
        backbone_dist_pairs,
        displacement_fn,
        pitch_quartets,
        propeller_base_pairs)
    loss_fn = jit(loss_fn)

    run_single_simulation = Partial(run_simulation, steps=steps, init_fn=init_fn, step_fn=step_fn, loss_fn=loss_fn)

    # Note: If has_aux is True then a tuple of ((value, auxiliary_data), gradient) is returned.
    # From https://jax.readthedocs.io/en/latest/_autosummary/jax.value_and_grad.html
    @functools.partial(jax.value_and_grad, has_aux=True)
    def _single_estimate(params, seed): # function only of the params to be differentiated w.r.t.
        trajectory, log_probs, avg_loss = run_single_simulation(params, seed)
        tot_log_prob = log_probs.sum()
        gradient_estimator = (tot_log_prob * jax.lax.stop_gradient(avg_loss) + avg_loss)
        return gradient_estimator, avg_loss
    return _single_estimate

"""
Mapped gradient estimator:

For a stochastic simulation, we have to run multiple instances of the same simulation to estimate
the true gradient. This function takes the information defining a particular stochastic simulation
(i.e. a "test case") and returns a function that estimates its gradient given a particular set of
parameters and seed. Note that the returned function will run *multiple* simulations rather than a
single one -- so, it splits the provided seed into `batch_size` seeds.
"""
def estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP):
    single_estimate_fn = single_estimate(displacement_fn, shift_fn, top_info, config_info, steps, dt=5e-3, T=DEFAULT_TEMP)

    mapped_estimate = jax.vmap(single_estimate_fn, [None, 0])

    @jit
    def _estimate_gradient(params, seed):
        seeds = jax.random.split(seed, batch_size)
        (gradient_estimator, avg_loss), grad = mapped_estimate(params, seeds)
        avg_grad = jnp.mean(grad, axis=0)
        return avg_grad, (gradient_estimator, avg_loss)
    return _estimate_gradient


def run(top_path, conf_path,
        sim_length, batch_size, opt_steps, init_params, key,
        T=DEFAULT_TEMP, dt=5e-3,
        output_basedir="v2/data/output"):

    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if not Path(conf_path).exists():
        raise RuntimeError(f"Configuration file does not exist at location: {conf_path}")

    output_basedir = Path(output_basedir)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_name = f"optimize_{timestamp}"
    run_dir = output_basedir / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    shutil.copy(top_path, run_dir)
    shutil.copy(conf_path, run_dir)
    params_str = f"topology file: {top_path}\nconfiguration file: {conf_path}\n"
    params_str += f"sim_length: {sim_length}\nbatch_size: {batch_size}\nopt_steps: {opt_steps}\n"
    params_str += f"init_params: {init_params}\nkey: {key}\ntemperature: {T}\ndt: {dt}\n"
    with open(run_dir / "params.txt", "w+") as f:
        f.write(params_str)
    print(bcolors.WARNING + f"Created directory and copied optimization information at location: {run_dir}" + bcolors.ENDC)

    print(bcolors.OKBLUE + f"Running optimization..." + bcolors.ENDC)

    # Information for a single "test case"
    # Note: in the future, we will have multiple of these
    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    displacement_fn, shift_fn = space.periodic(config_info.box_size)
    # Note how we get one `grad_fxn` per "test case." The gradient has to be estimated *per* test case
    grad_fxn = estimate_gradient(batch_size, displacement_fn, shift_fn, top_info, config_info,
                                 sim_length, dt=dt, T=T)

    # Initialize values relevant for the optimization loop
    lr = jopt.exponential_decay(0.1, opt_steps, 0.01)
    optimizer = jopt.adam(lr)
    opt_state = optimizer.init_fn(init_params)

    # Setup some logging, some required and some not
    params_ = list()
    losses = list()
    grads = list()
    save_every = 1
    # params_.append((0,) + (optimizer.params_fn(opt_state),))

    # Do the optimization
    step_times = list()
    for i in tqdm.trange(opt_steps, position=0):
        start = time.time()
        key, split = random.split(key)

        # Get the grad for our single test case (would have to average for multiple)
        grad, (_, avg_loss) = grad_fxn(optimizer.params_fn(opt_state), split)
        opt_state = optimizer.update_fn(i, grad, opt_state)

        end = time.time()
        step_times.append(end - start)
        # print(optimizer.params_fn(opt_state))
        # if i % save_every == 0 | i == (opt_steps-1):
            # coeffs_.append(((i+1),) + (optimizer.params_fn(opt_state),))

        grads.append(grad)
        params_.append(optimizer.params_fn(opt_state))
        losses.append(avg_loss)

    with open(run_dir / "params.pkl", "wb") as f:
        pickle.dump(params_, f)
    with open(run_dir / "losses.pkl", "wb") as f:
        pickle.dump(losses, f)
    with open(run_dir / "grads.pkl", "wb") as f:
        pickle.dump(grads, f)
    with open(run_dir / "step_times.pkl", "wb") as f:
        pickle.dump(step_times, f)
    return


if __name__ == "__main__":
    top_path = "data/simple-helix/generated.top"
    conf_path = "data/simple-helix/start.conf"
    init_params = jnp.array([0.60, 0.75, 1.1]) # my own hard-coded random FENE parameters
    key = random.PRNGKey(0)

    start = time.time()
    run(top_path=top_path, conf_path=conf_path,
        sim_length=1000, batch_size=20, opt_steps=200,
        init_params=init_params, key=key)
    end = time.time()
    total_time = end - start
    print(f"Execution took: {total_time}")
