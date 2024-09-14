from collections.abc import Callable
import functools
import os
import time

import jax
import jax.numpy as jnp
import jax_md
import matplotlib.pyplot as plt
import numpy as np
import optax

import jax_dna
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.gradient_estimators as grad_est
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.toml as toml_reader
import jax_dna.simulators.jax_md as jmd
import jax_dna.simulators.oxdna as oxdna
import jax_dna.losses.observable_wrappers as loss_wrapper
import jax_dna.observables as obs
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

def main():
    start = time.time()
    topology_fname = "data/templates/simple-helix/sys.top"
    traj_fname = "data/templates/simple-helix/init.conf"
    simulation_config = "jax_dna/input/dna1/default_simulation.toml"
    energy_config = "jax_dna/input/dna1/default_energy.toml"

    n_eq_steps = 100
    n_samples_steps = 100000
    sample_every = 100
    lr  = 0.0005
    min_n_eff_factor = 0.95
    seed = 0
    n_epochs = 1000

    key = jax.random.PRNGKey(seed)
    n_ref_states = n_samples_steps // sample_every

    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )
    experiment_config = toml_reader.parse_toml(simulation_config)
    energy_config = toml_reader.parse_toml(energy_config)

    kT = experiment_config["kT"]

    energy_configs = [
        # single param
        dna1_energy.FeneConfiguration.from_dict(energy_config["fene"]),
        # multiple params
        dna1_energy.BondedExcludedVolumeConfiguration.from_dict(energy_config["bonded_excluded_volume"]),
        # shortcut for all params, though you could list them all too
        dna1_energy.StackingConfiguration.from_dict(energy_config["stacking"] | {"kt": kT }, ("*",)),
        dna1_energy.UnbondedExcludedVolumeConfiguration.from_dict(energy_config["unbonded_excluded_volume"]),
        dna1_energy.HydrogenBondingConfiguration.from_dict(energy_config["hydrogen_bonding"]),
        dna1_energy.CrossStackingConfiguration.from_dict(energy_config["cross_stacking"]),
        dna1_energy.CoaxialStackingConfiguration.from_dict(energy_config["coaxial_stacking"]),
    ]

    opt_params = [c.opt_params for c in energy_configs]

    # configs and energy functions should be in the same order
    # though maybe we can be more clever in the future
    energy_fns = [
        dna1_energy.Fene,
        dna1_energy.BondedExcludedVolume,
        dna1_energy.Stacking,
        dna1_energy.UnbondedExcludedVolume,
        dna1_energy.HydrogenBonding,
        dna1_energy.CrossStacking,
        dna1_energy.CoaxialStacking,
    ]

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    init_body = traj.states[0].to_rigid_body()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT/diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT/rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    loss_fns = [
        functools.partial(
            loss_wrapper.ObservableLossFn(
                observable=obs.propeller.PropellerTwist(
                    rigid_body_transform_fn=transform_fn,
                    h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
                ),
                loss_fn=loss_wrapper.SquaredError(),
            ),
            target=obs.propeller.TARGETS["oxDNA"]
        ),
    ]

    space = jax_md.space.free()

    energy_fn_builder = functools.partial(
        grad_est.difftre2.build_energy_function,
        displacement_fn=space[0],
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        rigid_body_transform_fn=transform_fn,
        seq_one_hot=seq,
        bonded_neighbors=top.bonded_neighbors,
        unbonded_neighbors=top.unbonded_neighbors.T,
    )

    sim_init_fn = functools.partial(
        jmd.JaxMDSimulator,
        simulator_params=jmd.StaticSimulatorParams(
            seq=seq,
            mass=mass,
            bonded_neighbors=top.bonded_neighbors,
            n_steps=n_samples_steps,
            checkpoint_every=0,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=space,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jmd.NoNeighborList(unbonded_nbrs=top.unbonded_neighbors),
        transform_fn=transform_fn,
        topology=top,
    )


    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(opt_params)

    key, split = jax.random.split(key)
    ge = grad_est.difftre2.DiffTRe(
        energy_fn_builder=energy_fn_builder,
        beta=jnp.array(1/kT),
        min_n_eff=jnp.array(int(n_ref_states * min_n_eff_factor)),
        loss_fns=tuple(loss_fns),
        losses_reduce_fn=jnp.mean,
        sim_init_fn = sim_init_fn,
        energy_configs=tuple(energy_configs),
        energy_fns=tuple(energy_fns),
        init_state=init_body,
        n_steps=n_samples_steps,
        n_eq_steps=n_eq_steps,
        sample_every=sample_every,
    ).initialize(opt_params, split)

    loss_vals, resets = [], []
    for i in tqdm(range(n_epochs)):
        ge, grads, loss, losses, regenerated = ge(opt_params, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        opt_params = optax.apply_updates(opt_params, updates)
        loss_vals.append(loss)
        if regenerated:
            resets.append(i)

    import matplotlib.pyplot as plt
    plt.plot(loss_vals)
    plt.vlines(resets, min(loss_vals), max(loss_vals), color="red")
    plt.show()





if __name__ == "__main__":
    main()