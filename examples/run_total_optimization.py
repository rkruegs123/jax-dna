import dataclasses as dc
import functools
import itertools
import operator
import os
from pathlib import Path
import typing
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray


jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


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
import jax_dna.observables as jd_obs
import jax_dna.simulators.base as jdna_simulators
import jax_dna.simulators.io as jdna_sio
import jax_dna.simulators.jax_md as jaxmd
import jax_dna.utils.types as jdna_types


class MockLogger:

    def log_metric(self, name:str, value:float, step:int):
        print(f"Step {step}: {name} = {value}")



def main():
    logger = MockLogger()

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",
                "JAX_PLATFORM_NAME": "cpu",
                # "RAY_DEBUG": "legacy",
            }
        }
    )


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

    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    )


    cwd = Path(os.getcwd())
    out_dir = cwd / "proptwist_store"
    dproptwist_dopt_loc =  out_dir / "dproptwist_dparams.pkl"
    proptwist_loc = out_dir / "proptwist.pkl"

    def proptwist_writer_fn(
        outs: jdna_types.Grads,
        aux: tuple[jdna_types.Arr_N, jdna_types.MetaData],
        meta_data: jdna_types.MetaData,
    )-> None:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        jdna_tree.save_pytree(aux[0], proptwist_loc)
        jdna_tree.save_pytree(outs, dproptwist_dopt_loc)
        return proptwist_loc, dproptwist_dopt_loc


    init_body = traj.states[0].to_rigid_body()
    run_steps = 1000
    def simulator_fn(
        params: jdna_types.Params,
        meta:jdna_types.MetaData
    ) -> tuple[str, str]:
        sim_traj, _ = sampler.run(params, init_body, run_steps, key)
        obs = prop_twist_fn(sim_traj)

        return obs, (obs, meta)

    grad_and_obs_fn = jax.jit(jax.jacfwd(simulator_fn, has_aux=True))



    proptwist_simulator = sim_actor.SimulatorActor.remote(
        fn=grad_and_obs_fn,
        exposes=[obs_proptwist, obs_dproptwist_dparams],
        meta_data={},
        writer_fn=proptwist_writer_fn,
    )

    # ==========================================================================





    # Objective ================================================================
    # this is function process the return of the proptwist simulation actors
    def proptwist_gradfn(
        proptwist_loc: str,
        dproptwist_dparams_loc: str,
    ) -> jdna_types.Grads:
        TARGET_PROPELLER_TWIST = 21.7

        # we need to calculate the gradient of the loss with respect
        # to the proptwist and then multiply by the gradient of the proptwist
        # with respect to the parameters to get the gradient we need.
        # δproptwist     δLoss
        # ---------- * ----------
        #  δparams     δproptwist

        proptwist = jdna_tree.load_pytree(proptwist_loc) # array of [batch_size, time]

        def loss_fn(obs:jnp.ndarray) -> jnp.ndarray:
            return (obs.mean(axis=-1).mean() - TARGET_PROPELLER_TWIST)**2

        # array of [batch_size, time]
        loss, dloss_dproptwist = jax.value_and_grad(loss_fn)(proptwist)
        # pytree with vals param -> [batch_size, time]
        dproptwist_dparams = jdna_tree.load_pytree(dproptwist_dparams_loc)

        dloss_dopts = jax.tree.map(
            lambda dpt_dparam: (dpt_dparam * dloss_dproptwist).sum(),
            dproptwist_dparams,
        )

        return dloss_dopts, loss

    def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
        if len(trees) <= 1:
            return trees[0]
        summed = jax.tree.map(operator.add, *trees)
        return jax.tree.map(lambda x: x / len(trees), summed)


    propeller_twist_objective = sim_actor.Objective.remote(
        required_observables=[obs_proptwist, obs_dproptwist_dparams],
        needed_observables=[obs_proptwist, obs_dproptwist_dparams],
        logging_observables=["loss", obs_proptwist],
        grad_fn=proptwist_gradfn,
    )
    # ==========================================================================

    objectives = [propeller_twist_objective]
    simulators = [proptwist_simulator]

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
        optimizer = optax.adam(learning_rate=1e-5),
        aggregate_grad_fn=tree_mean,
    )

    for i in range(optimization_config["n_steps"]):
        print("Step", i, "Starting")
        opt_state, opt_params = opt.step(opt_params)

        for objective in opt.objectives:
            log_values = ray.get(objective.logging_observables.remote())
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)

        opt = opt.post_step(optimizer_state=opt_state)
        print("Step", i, "Completed")



if __name__=="__main__":
    main()