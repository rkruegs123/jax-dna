import pdb
import numpy as onp
from tqdm import tqdm
import time
import pprint
from copy import deepcopy
from pathlib import Path
import functools

import jax.numpy as jnp
from jax import vmap, jit, value_and_grad, grad, random, tree_util
from jax_md import rigid_body, space, simulate
import optax

from jax_dna.cgdna import marginals, oxdna_to_cgdna
from jax_dna.common import topology, trajectory, utils, checkpoint
from jax_dna.dna1 import model

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = 10
if checkpoint_every is None:
    scan = jax.lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)

def run():
    displacement_fn, shift_fn = space.free()

    # Load the system information
    sys_basedir = Path("data/sys-defs/lb-seqs/seq1/")
    top_path = sys_basedir / "seq.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    assert(len(top_info.seq) == 24*2)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "seq.conf"
    config_info = trajectory.TrajectoryInfo(
        top_info, read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = config_info.get_states()[0]

    # Setup the simulation
    dt = 5e-3
    t_kelvin = utils.DEFAULT_TEMP
    kT = utils.get_kt(t_kelvin)
    gamma_eq = rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64))
    gamma_scale = 2500
    gamma_opt = rigid_body.RigidBody(
        center=gamma_eq.center * gamma_scale,
        orientation=gamma_eq.orientation * gamma_scale)
    mass = rigid_body.RigidBody(
        center=jnp.array([utils.nucleotide_mass], dtype=jnp.float64),
        orientation=jnp.array([utils.moment_of_inertia], dtype=jnp.float64))

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
    num_eq_steps = 10000
    eq_fn = lambda params, key: sim_fn(params, init_body, num_eq_steps, key, gamma_eq)
    eq_fn = jit(eq_fn)
    mapped_eq_fn = jit(vmap(eq_fn, (None, 0)))

    # Construct the loss function

    ## Get the cgDNA ground truth terms
    intra_coord_means, intra_coord_vars = marginals.compute_marginals(
        top_info.seq[:24], verbose=False)

    propeller_means = jnp.array(intra_coord_means["propeller"][1:-1])
    propeller_means *= 11.5 # convert to degrees
    propeller_vars = jnp.array(intra_coord_vars["propeller"][1:-1])
    propeller_vars *= 11.5 # convert to degrees

    propeller_base_pairs = list(zip(onp.arange(1, 23), onp.arange(46, 24, -1)))
    propeller_base_pairs = jnp.array(propeller_base_pairs)

    ## Construct the functions to evaluate a trajectory

    @jit
    def kl_divergence(true_mean, true_var, est_mean, est_var):
        return jnp.log(est_var / true_var) + (true_var**2 + (true_mean - est_mean)**2) / (2 * est_var**2) - 1/2

    @jit
    def compute_kl_divergences(all_prop_twists):
        prop_twists_means = jnp.mean(all_prop_twists, axis=0)
        prop_twists_vars = jnp.var(all_prop_twists, axis=0)
        all_kl_divergences = vmap(kl_divergence, (0, 0, 0, 0))(propeller_means, propeller_vars, prop_twists_means, prop_twists_vars)
        return all_kl_divergences, (prop_twists_means, prop_twists_vars)

    reader = jit(oxdna_to_cgdna.get_reader(24 * 2, 24, top_info.seq))
    num_steps = 50000
    sample_every = 100

    @jit
    def traj_loss_fn(traj):
        states_to_eval = traj[::sample_every]
        traj_ptwists = reader(states_to_eval)[:, 1:23]
        kl_divergences, (ptwist_means, ptwist_vars) = compute_kl_divergences(traj_ptwists)
        return jnp.sum(kl_divergences), (ptwist_means, ptwist_vars)

    @jit
    def loss_fn(params, eq_body, key):
        fin_pos, traj = sim_fn(params, eq_body, num_steps, key, gamma_opt)
        loss, metadata = traj_loss_fn(traj)
        return loss, metadata

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    batched_grad_fn = jit(vmap(grad_fn, (None, 0, 0)))

    # Setup the optimization
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    # params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
    # params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]
    params["hydrogen_bonding"] = model.DEFAULT_BASE_PARAMS["hydrogen_bonding"]
    lr = 0.01
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    n_epochs = 100
    batch_size = 10
    key = random.PRNGKey(0)

    times_path = "test_times.txt"
    loss_path = "test_loss.txt"
    params_path = "test_params.txt"
    grads_path = "test_grads.txt"
    
    for i in tqdm(range(n_epochs)):
        key, iter_key = random.split(key)
        iter_key, eq_key = random.split(iter_key)
        eq_keys = random.split(eq_key, batch_size)
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        eq_bodies, _ = mapped_eq_fn(params, eq_keys)

        batch_keys = random.split(iter_key, batch_size)
        start = time.time()
        (losses, batched_metadata), grads = batched_grad_fn(params, eq_bodies, batch_keys)
        end = time.time()
        iter_time = end - start

        avg_grads = tree_util.tree_map(jnp.mean, grads)

        with open(times_path, "a") as f:
            f.write(f"{iter_time}\n")
        with open(loss_path, "a") as f:
            f.write(f"{jnp.mean(losses)}\n")
        with open(grads_path, "a") as f:
            f.write(f"{pprint.pformat(avg_grads, indent=4)}\n")
        with open(params_path, "a") as f:
            f.write(f"{pprint.pformat(params, indent=4)}\n")

        updates, opt_state = optimizer.update(avg_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

if __name__ == "__main__":
    run()
