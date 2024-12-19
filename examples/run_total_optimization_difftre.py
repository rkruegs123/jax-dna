
import functools
import os
from pathlib import Path
import typing
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray

import examples.simulator_actor as sim_actor

import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.toml as toml_reader
import jax_dna.input.tree as jdna_tree
import jax_dna.observables as jd_obs
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.base as jdna_optimization
import jax_dna.simulators.jax_md as jaxmd
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types
from jax_dna.input import topology, trajectory

jax.config.update("jax_enable_x64", True)

class MockLogger:
    def log_metric(self, name:str, value:float, step:int):
        print(f"Step {step}: {name} = {value}")


def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
    if len(trees) <= 1:
        return trees[0]
    summed = jax.tree.map(operator.add, *trees)
    return jax.tree.map(lambda x: x / len(trees), summed)



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

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_configs,
        transform_fn=transform_fn,
    )

    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq_one_hot),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    # energy_fn_builder = lambda rigid_body: energy_fn_builder_fn(
    #         rigid_body,
    #         seq=jnp.array(top.seq_one_hot),
    #         bonded_neighbors=top.bonded_neighbors,
    #         unbonded_neighbors=top.unbonded_neighbors.T,
    # )

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

    cwd = Path(os.getcwd())
    output_dir = cwd / "basic_trajectory"
    trajectory_loc = output_dir / "trajectory.pkl"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    init_body = traj.states[0].to_rigid_body()
    run_steps = 1000
    n_equilibration_steps = run_steps // 10

    def simulator_fn(
        params: jdna_types.Params,
        meta: jdna_types.MetaData,
    ) -> tuple[str, str]:
        traj, _ = sampler.run(params, init_body, run_steps, key)
        return traj, meta

    def trajectory_writer_fn(
        traj: jdna_sio.SimulatorTrajectory,
        aux: dict[str, typing.Any],
        meta: jdna_types.MetaData,
    ) -> list[str]:
        jdna_tree.save_pytree(traj, trajectory_loc)
        return [trajectory_loc]

    obs_trajectory = "trajectory"

    trajectory_simulator = sim_actor.SimulatorActor.remote(
        fn=simulator_fn,
        exposes = [obs_trajectory],
        meta_data = {},
        writer_fn = trajectory_writer_fn,
    )
    # ==========================================================================


    # Objective ================================================================
    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    )

    def prop_twist_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
    ) -> tuple[float, tuple[str, typing.Any]]:
        obs = prop_twist_fn(traj)
        loss = (jnp.dot(weights, obs) - jd_obs.propeller.TARGETS["oxDNA"])**2
        loss = jnp.sqrt(loss)
        return loss, (obs, {})


    propeller_twist_objective = jdna_objective.DiffTReObjective.remote(
        required_observables = [obs_trajectory],
        needed_observables = [obs_trajectory],
        logging_observables = ["loss", "prop_twist"],
        grad_or_loss_fn = prop_twist_loss_fn,
        energy_fn_builder = energy_fn_builder,
        opt_params = opt_params,
        trajectory_key = obs_trajectory,
        beta = jnp.array(1/kT),
        n_equilibration_steps = n_equilibration_steps,
    )
    # ==========================================================================

    objectives = [propeller_twist_objective]
    simulators = [trajectory_simulator]

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
        optimizer = optax.adam(learning_rate=1e-3),
        aggregate_grad_fn=tree_mean,
    )


    for i in range(optimization_config["n_steps"]):
        print("Step", i, "Starting")
        opt_state, opt_params = opt.step(opt_params)

        for objective in opt.objectives:
            log_values = ray.get(objective.logging_observables.remote())
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )
        print("Step", i, "Completed")


if __name__ == "__main__":
    main()