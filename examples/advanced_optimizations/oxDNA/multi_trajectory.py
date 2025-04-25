"""An example of running a multi trajectory simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.oxdna.oxDNA``
"""
import functools
import logging
import os
from pathlib import Path
import shutil
import typing
import warnings
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray
from tqdm import tqdm

import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.toml as toml_reader
import jax_dna.input.topology as jdna_top
import jax_dna.input.trajectory as jdna_traj
import jax_dna.input.tree as jdna_tree
import jax_dna.observables as jd_obs
import jax_dna.optimization.simulator as jdna_simulator
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.simulators.oxdna as oxdna
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types
import jax_dna.ui.loggers.console as console_logger


jax.config.update("jax_enable_x64", True)


# Logging configurations =======================================================
logging.basicConfig(level=logging.DEBUG, filename="opt.log", filemode="w")
objective_logging_config = {
    "level":logging.DEBUG,
    "filename":"objective.log",
    "filemode":"w",
}
simulator_logging_config = objective_logging_config | {"filename": "simulator.log"}
# ==============================================================================


# To combine the gradients of multiple objectives, we can use a mean, however
# this example only has one objective, so it will remain unchanged.
def tree_mean(trees:tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
    if len(trees) <= 1:
        return trees[0]
    summed = jax.tree.map(operator.add, *trees)
    return jax.tree.map(lambda x: x / len(trees), summed)




def main():

    # The coordination of objectives and simulators is done through Ray actors.
    # So we need to initialize a ray server
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=True,
        runtime_env={
            "env_vars": {
                "JAX_ENABLE_X64": "True",
                "JAX_PLATFORM_NAME": "cpu",
            }
        }
    )


    # Input configuration ======================================================
    optimization_config = {
        "n_steps": 20,
        "oxdna_build_threads": 4,
        "log_every": 10,
        "n_oxdna_runs": 3,
    }

    simulator_logging_config = {
        "filename": "simulator.log",
        "level": logging.DEBUG,
        "filemode": "w",
    }

    kT = toml_reader.parse_toml("jax_dna/input/dna1/default_simulation.toml")["kT"]
    geometry = toml_reader.parse_toml("jax_dna/input/dna1/default_energy.toml")["geometry"]

    template_dir = Path("data/templates/simple-helix")
    topology_fname = template_dir / "sys.top"

    cwd = Path(os.getcwd())
    # ==========================================================================


    # Energy Function ==========================================================
    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = []
    opt_params = []

    for ec in dna1_energy.default_energy_configs():
        # We are only interested in the stacking configuration
        # However we don't want to optimize ss_stack_weights and kt
        if isinstance(ec, dna1_energy.StackingConfiguration):
            ec = ec.replace(
                non_optimizable_required_params=(
                    "ss_stack_weights",
                    "kt",
                )
            )
            opt_params.append(ec.opt_params)
            energy_configs.append(ec)
        else:
            energy_configs.append(ec)
            opt_params.append({})

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

    top = jdna_top.from_oxdna_file(topology_fname)
    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
            / top.n_nucleotides
        )
    # ==========================================================================

    run_flag = oxdna.oxDNABinarySemaphoreActor.remote()


    # Simulators ================================================================
    sim_outputs_dir = cwd / "sim_outputs"
    sim_outputs_dir.mkdir(parents=True, exist_ok=True)

    def make_simulator(id:str, disable_build:bool) -> jdna_simulator.BaseSimulator:
        sim_dir = sim_outputs_dir / id

        if sim_dir.exists():
            warnings.warn(f"Directory {sim_dir} already exists. Assuming that's fine.")

        sim_dir.mkdir(parents=True, exist_ok=True)

        for f in template_dir.iterdir():
            shutil.copy(f, sim_dir)

        simulator = oxdna.oxDNASimulator(
            input_dir=sim_dir,
            sim_type=jdna_types.oxDNASimulatorType.DNA1,
            energy_configs=energy_configs,
            n_build_threads=optimization_config["oxdna_build_threads"],
            logger_config=simulator_logging_config | {"filename": f"simulator_{id}.log"},
            disable_build=disable_build,
            check_build_ready=lambda: ray.get(run_flag.check.remote()),
            set_build_ready=run_flag.set.remote,
        )

        output_dir = sim_dir / "trajectory"
        trajectory_loc = output_dir / "trajectory.pkl"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        def simulator_fn(
            params: jdna_types.Params,
            meta: jdna_types.MetaData,
        ) -> tuple[str, str]:
            simulator.run(params)

            ox_traj = jdna_traj.from_file(
                sim_dir / "output.dat",
                strand_lengths=top.strand_counts,
            )
            traj = jdna_sio.SimulatorTrajectory(
                rigid_body=ox_traj.state_rigid_body,
            )

            jdna_tree.save_pytree(traj, trajectory_loc)
            return [trajectory_loc]

        return jdna_simulator.SimulatorActor.options(
            runtime_env={
                "env_vars": {
                    oxdna.BIN_PATH_ENV_VAR: str(Path("../oxDNA/build/bin/oxDNA").resolve()),
                    oxdna.BUILD_PATH_ENV_VAR: str(Path("../oxDNA/build").resolve()),
                },
            },
        ).remote(
            name=id,
            fn=simulator_fn,
            exposes=[f"traj-{id}",],
            meta_data={},
        )


    sim_ids = [f"sim{i}" for i in range(optimization_config["n_oxdna_runs"])]
    traj_ids = [f"traj-{id}" for id in sim_ids]

    simulators = [make_simulator(*id_db) for id_db in zip(sim_ids, [False] + [True]*(len(sim_ids)-1))]
    # ==========================================================================



    # Objective ================================================================
    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]]),
    )

    def prop_twist_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
    ) -> tuple[float, tuple[str, typing.Any]]:
        obs = prop_twist_fn(traj)
        expected_prop_twist = jnp.dot(weights, obs)
        loss = (expected_prop_twist - jd_obs.propeller.TARGETS["oxDNA"]) ** 2
        loss = jnp.sqrt(loss)
        return loss, (("prop_twist", expected_prop_twist), {})

    propeller_twist_objective = jdna_objective.DiffTReObjectiveActor.remote(
        name="prop_twist",
        required_observables=traj_ids,
        needed_observables=traj_ids,
        logging_observables=["loss", "prop_twist", "neff"],
        grad_or_loss_fn=prop_twist_loss_fn,
        energy_fn_builder=energy_fn_builder,
        opt_params=opt_params,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
        max_valid_opt_steps=10,
    )
    # ==========================================================================



    # Logger ===================================================================
    logger = console_logger.ConsoleLogger(
        log_dir="logs",
    )
    # ==========================================================================


    # Optimization =============================================================
    objectives = [propeller_twist_objective]

    opt = jdna_optimization.Optimization(
        objectives=objectives,
        simulators=simulators,
        optimizer = optax.adam(learning_rate=1e-3),
        aggregate_grad_fn=tree_mean,
        logger=logger,
    )
    # ==========================================================================



    # Run optimization =========================================================
    for i in tqdm(range(optimization_config["n_steps"]), desc="Optimizing"):
        opt_state, opt_params, grads = opt.step(opt_params)

        for objective in opt.objectives:
            log_values = ray.get(objective.logging_observables.remote())
            for (name, value) in log_values:
                logger.log_metric(name, value, step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )
        # block the oxdna builds so that the simulator that builds can do so
        run_flag.set_bin_status.remote(False)
