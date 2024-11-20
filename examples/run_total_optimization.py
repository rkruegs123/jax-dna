import dataclasses as dc
import functools
import itertools
import typing
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax_md
import ray


import examples.simulator_actor as sim_actor

import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.tree as jdna_tree
import jax_dna.input.toml as toml_reader
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.losses.observable_wrappers as jdna_losses
import jax_dna.optimization.base as jdna_optimization
import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.simulators.jax_md as jaxmd
import jax_dna.utils.types as jdna_types



def main():
    logger = None

    optimization_config = {
        "n_steps": 100,
        "batch_size": 1,
    }

    simulation_config = toml_reader.parse_toml("jax_dna/input/dna1/default_simulation.toml")
    energy_config = toml_reader.parse_toml("jax_dna/input/dna1/default_energy.toml")

    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = dna1_energy.default_configs()
    opt_params = [c.opt_params for c in energy_configs]

    obs_proptwist = "proptwist"
    obs_dproptwist_dparams = "dproptwist_dparams"


    # Simulator ================================================================
    topology_fname = "data/sys-defs/simple-helix/sys.top"
    traj_fname = "data/sys-defs/simple-helix/bound_relaxed.conf"
    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )

    displacement_fn, shift_fn = jax_md.space.free()
    key = jax.random.PRNGKey(0)

    kT = simulation_config["kT"]
    dt = simulation_config["dt"]
    diff_coef = simulation_config["diff_coef"]
    rot_diff_coef = simulation_config["rot_diff_coef"]

    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT/diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT/rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([simulation_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([simulation_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    sampler = jaxmd.JaxMDSimulator(
        energy_configs=energy_configs,
        energy_fns=energy_fns,
        topology=top,
        simulator_params=jaxmd.StaticSimulatorParams(
            seq=seq,
            mass=mass,
            bonded_neighbors=top.bonded_neighbors,
            n_steps=5_000,
            checkpoint_every=0,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=(displacement_fn, shift_fn),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jaxmd.NoNeighborList(unbonded_nbrs=top.unbonded_neighbors),
    )
    # ==========================================================================







    # Objective ================================================================
    # this is function process the return of the proptwist simulation actors
    def proptwist_gradfn(
        proptwist_loc: str,
        dproptwist_dparams_loc: str,
        meta_data: str,
    ) -> jdna_types.Grads:
        TARGET_PROPELLER_TWIST = 21.7

        # we need to calculate the gradient of the loss with respect
        # to the proptwist and then multiply by the gradient of the proptwist
        # with respect to the parameters to get the gradient we need.
        # δproptwist     δLoss
        # ---------- * ----------
        #  δparams     δproptwist

        proptwist = jdna_tree.load_pytree(proptwist_loc) # array of [batch_size, time]

        def loss(obs:jnp.ndarray) -> jnp.ndarray:
            return (obs.mean(axis=1).mean() - TARGET_PROPELLER_TWIST)**2

        # array of [batch_size, time]
        dloss_dproptwist = jax.grad(loss)(proptwist)
        # pytree with vals param -> [batch_size, time]
        dproptwist_dparams = jdna_tree.load_pytree(dproptwist_dparams_loc)

        dloss_dopts = jax.tree_map(
            lambda dpt_dparam: (dpt_dparam * dloss_dproptwist).sum(),
            dproptwist_dparams,
        )

        return dloss_dopts


    propeller_twist_objective = sim_actor.Objective.remote(
        required_observables=[obs_proptwist, obs_dproptwist_dparams],
        needed_observables=[obs_proptwist, obs_dproptwist_dparams],
        grad_fn=proptwist_gradfn,
    )
    # ==========================================================================




    objectives = [propeller_twist_objective]
    simulators = []

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
    )

    for i in range(optimization_config["n_steps"]):
        opt, grads = opt.step(params)
        params = opt.update_params(params, grads)

        for objective in opt.objectives:
            log_values = objective.get_latest_values()
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)



if __name__=="__main__":
    main()