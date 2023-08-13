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
from jax import jit, vmap, random, grad, value_and_grad, lax, tree_util
from jax_md import space, simulate, rigid_body

from jax_dna.common import utils, topology, trajectory, checkpoint
from jax_dna.loss import persistence_length
from jax_dna.dna1 import model

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

def get_all_quartets(n_nucs_per_strand):
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    bps = list(zip(s1_nucs, s2_nucs))
    n_bps = len(s1_nucs)
    all_quartets = list()
    for i in range(n_bps-1):
        bp1 = bps[i]
        bp2 = bps[i+1]
        all_quartets.append(bp1 + bp2)
    return jnp.array(all_quartets, dtype=jnp.int32)

def run():

    # Load the system
    sys_basedir = Path("data/sys-defs/persistence-length")
    top_path = sys_basedir / "init.top"
    top_info = topology.TopologyInfo(top_path, reverse_direction=True)
    seq_oh = jnp.array(utils.get_one_hot(top_info.seq), dtype=jnp.float64)

    conf_path = sys_basedir / "relaxed.dat"
    conf_info = trajectory.TrajectoryInfo(
        top_info,
        read_from_file=True, traj_path=conf_path, reverse_direction=True
    )
    init_body = conf_info.get_states()[0]

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


    n_eq_steps = 10000
    def eq_fn(params, body, key):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, body, mass=mass)

        fori_step_fn = lambda t, state: step_fn(state)
        fori_step_fn = jit(fori_step_fn)

        eq_state = lax.fori_loop(0, n_eq_steps, fori_step_fn, init_state)
        return eq_state.position
    


    # n_sample_steps = int(1e4) # For testing
    # sample_every = int(1e1) # For testing
    # n_sample_steps = int(1e7)
    # sample_every = int(1e4)
    n_sample_steps = int(1e7)
    sample_every = int(1e4)
    assert(n_sample_steps % sample_every == 0)
    n_ref_states = n_sample_steps // sample_every
    
    def get_ref_states(params, eq_init_body, key):
        em = model.EnergyModel(displacement_fn, params, t_kelvin=t_kelvin)
        energy_fn = lambda body: em.energy_fn(body,
                                              seq=seq_oh,
                                              bonded_nbrs=top_info.bonded_nbrs,
                                              unbonded_nbrs=top_info.unbonded_nbrs.T)
        
        init_fn, step_fn = simulate.nvt_langevin(energy_fn, shift_fn, dt, kT, gamma)
        init_state = init_fn(key, eq_init_body, mass=mass)

        step_fn = jit(step_fn)

        fori_step_fn = lambda t, state: step_fn(state)
        fori_step_fn = jit(fori_step_fn)

        state = deepcopy(init_state)
        trajectory = list()
        for i in tqdm(range(n_ref_states)):
            state = lax.fori_loop(0, sample_every, fori_step_fn, state)
            trajectory.append(state.position)

        ref_states = tree_stack(trajectory)
        ref_energies = vmap(energy_fn)(ref_states)
            
        return ref_states, ref_energies


    # Construct the loss function

    quartets = get_all_quartets(n_nucs_per_strand=init_body.center.shape[0] // 2)
    quartets = quartets[25:]
    quartets = quartets[:-25]
    base_site = jnp.array([model.com_to_hb, 0.0, 0.0])
    compute_all_curves = vmap(persistence_length.get_correlation_curve, (0, None, None))

    target_lp_nm = 37.5 # FIXME: some dummy value to optimize to

    @jit
    def loss_fn(params, ref_states, ref_energies):
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


        unweighted_corr_curves, unweighted_l0_avgs = compute_all_curves(ref_states, quartets, base_site)
        weighted_corr_curves = vmap(lambda v, w: v * w)(unweighted_corr_curves, weights)
        weighted_l0_avgs = vmap(lambda l0, w: l0 * w)(unweighted_l0_avgs, weights)
        expected_corr_curv = jnp.sum(weighted_corr_curves, axis=0)
        expected_l0_avg = jnp.sum(weighted_l0_avgs)
        expected_lp = persistence_length.persistence_length_fit(expected_corr_curv, expected_l0_avg)


        # Compute effective sample size
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return (expected_lp - target_lp_nm)**2, (n_eff, expected_lp, expected_corr_curv)
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
    print(f"Finished generating initial reference states and energies")

    min_n_eff = int(n_ref_states * 0.90)
    all_losses = list()
    all_n_effs = list()
    all_ref_losses = list()
    all_ref_times = list()
    plot_every = 2

    for i in tqdm(range(n_epochs)):
        (loss, (n_eff, curr_lp, expected_corr_curv)), grads = grad_fn(params, ref_states, ref_energies)

        if i == 0:
            all_ref_losses.append(loss)
            all_ref_times.append(i)

        if n_eff < min_n_eff:
            print(f"n_eff was {n_eff}... resampling reference states...")
            key, split = random.split(key)

            eq_key, ref_key = random.split(split)

            start = time.time()
            eq_pos = eq_fn(params, ref_states[-1], eq_key)
            end = time.time()
            print(f"Generating equilibrium state took {end - start} seconds")
            
            start = time.time()
            ref_states, ref_energies = get_ref_states(params, eq_pos, ref_key)
            end = time.time()
            print(f"Generating reference states took {end - start} seconds")
            (loss, (n_eff, curr_lp, expected_corr_curv)), grads = grad_fn(params, ref_states, ref_energies)

            all_ref_losses.append(loss)
            all_ref_times.append(i)


        all_losses.append(loss)
        all_n_effs.append(n_eff)


        print(f"Loss: {loss}")
        print(f"Effective sample size: {n_eff}")
        print(f"Current expected Lp: {curr_lp}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        plt.plot(expected_corr_curv)
        plt.savefig(f"difftre_iter{i}_1e7v2_corr.png")
        plt.clf()


        if i % plot_every == 0:
            plt.plot(onp.arange(i+1), all_losses, linestyle="--")
            plt.scatter(all_ref_times, all_ref_losses, marker='o')
            plt.savefig(f"difftre_iter{i}_1e7v2.png")



if __name__ == "__main__":
    run()
