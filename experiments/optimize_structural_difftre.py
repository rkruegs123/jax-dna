import pdb
from pathlib import Path
from copy import deepcopy
import functools
import pprint
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as onp

import optax
import jax.numpy as jnp
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util, lax
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import geometry, pitch, propeller
from jax_dna.dna1 import model
from jax_dna import dna1, loss

from jax.config import config
config.update("jax_enable_x64", True)


checkpoint_every = None
if checkpoint_every is None:
    scan = lax.scan
else:
    scan = functools.partial(checkpoint.checkpoint_scan,
                             checkpoint_every=checkpoint_every)


def tree_stack(trees):
    return tree_util.tree_map(lambda *v: jnp.stack(v), *trees)

def run():

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
        traj = tree_stack(trajectory)
        end = time.time()
        print(f"Generating reference states took {end - start} seconds")
        """


        return traj


    n_eq_steps = 5000
    n_sample_steps = 50000
    sample_every = 500
    assert(n_sample_steps % sample_every == 0)
    n_ref_states = n_sample_steps // sample_every

    def get_ref_states(params, init_body, key):
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
        return ref_states, ref_energies




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
    def loss_fn(params, ref_states: rigid_body.RigidBody, ref_energies):

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
        # FIXME: only considering ptwist for now. Have to extend to multiple observables
        # FIXME: since none of the observables actually depend on theta (maybe they will for melting temperature), we could just precompute them and map over them...
        unweighted_ptwists = vmap(compute_avg_ptwist)(ref_states) # FIXME: this doesn't depend on params...
        weighted_ptwists = weights * unweighted_ptwists # element-wise multiplication
        expected_ptwist = jnp.sum(weighted_ptwists)

        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return (expected_ptwist - propeller.TARGET_PROPELLER_TWIST)**2, n_eff
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    grad_fn = jit(grad_fn)

    # Setup the optimization
    params = deepcopy(model.EMPTY_BASE_PARAMS)
    params["fene"] = model.DEFAULT_BASE_PARAMS["fene"]
    params["stacking"] = model.DEFAULT_BASE_PARAMS["stacking"]
    lr = 0.001
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    n_epochs = 100
    key = random.PRNGKey(0)

    init_body = conf_info.get_states()[0]
    print(f"Generating initial reference states and energies...")
    ref_states, ref_energies = get_ref_states(params, init_body, key)

    min_n_eff = int(n_ref_states * 0.95)
    all_losses = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_times = list()
    plot_every = 10
    for i in tqdm(range(n_epochs)):
        (loss, n_eff), grads = grad_fn(params, ref_states, ref_energies)

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)

        if n_eff < min_n_eff:
            print(f"Resampling reference states...")
            key, split = random.split(key)
            ref_states, ref_energies = get_ref_states(params, ref_states[-1], split)
            (loss, n_eff), grads = grad_fn(params, ref_states, ref_energies)

            all_ref_losses.append(loss)
            all_ref_times.append(i)


        all_losses.append(loss)
        all_n_effs.append(n_eff)


        print(f"Loss: {loss}")
        print(f"Effective sample size: {n_eff}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if i % plot_every == 0:
            plt.plot(onp.arange(i+1), all_losses, linestyle="--")
            plt.scatter(all_ref_times, all_ref_losses, marker='o')
            plt.savefig(f"difftre_iter{i}.png")


if __name__ == "__main__":
    run()
