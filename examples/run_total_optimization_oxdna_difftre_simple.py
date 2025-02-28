import functools
import itertools
import logging
import os
import numpy as np
from pathlib import Path
import typing
import jax
import jax.numpy as jnp
import jax_md
import optax
from tqdm import tqdm


import jax_dna
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.toml as toml_reader
import jax_dna.input.tree as jdna_tree
import jax_dna.input.trajectory as jdna_traj
import jax_dna.observables as jd_obs
import jax_dna.optimization.simulator as jdna_simulator
import jax_dna.optimization.objective as jdna_objective
import jax_dna.optimization.optimization as jdna_optimization
import jax_dna.simulators.oxdna as oxdna
import jax_dna.simulators.io as jdna_sio
import jax_dna.utils.types as jdna_types
import jax_dna.ui.loggers.jupyter as jupyter_logger
import jax_dna.ui.loggers.console as console_logger
from jax_dna.input import topology, trajectory

os.environ[oxdna.BIN_PATH_ENV_VAR] = str(Path("../oxDNA/build/bin/oxDNA").resolve())
os.environ[oxdna.BUILD_PATH_ENV_VAR] = str(Path("../oxDNA/build").resolve())

jax.config.update("jax_enable_x64", True)


def tree_mean(trees: tuple[jdna_types.PyTree]) -> jdna_types.PyTree:
    if len(trees) <= 1:
        return trees[0]
    summed = jax.tree.map(operator.add, *trees)
    return jax.tree.map(lambda x: x / len(trees), summed)


def main():

    # Input configuration ======================================================
    optimization_config = {
        "n_steps": 1000,
        "oxdna_build_threads": 4,
        "log_every": 10,
    }

    simulator_logging_config = {
        "filename": "simulator.log",
        "level": logging.DEBUG,
        "filename": "objective.log",
        "filemode": "w",
    }

    kT = toml_reader.parse_toml("jax_dna/input/dna1/default_simulation.toml")["kT"]
    geometry = toml_reader.parse_toml("jax_dna/input/dna1/default_energy.toml")["geometry"]

    input_dir = Path("data/templates/simple-helix")
    topology_fname = input_dir / "sys.top"

    cwd = Path(os.getcwd())
    # ==========================================================================


    # Energy Function ==========================================================
    energy_fns = dna1_energy.default_energy_fns()
    energy_configs = []
    opt_params = []

    for ec in dna1_energy.default_configs():
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

    top = topology.from_oxdna_file(topology_fname)
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

    # Simulator ================================================================
    simulator = oxdna.oxDNASimulator(
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_configs=energy_configs,
        n_build_threads=optimization_config["oxdna_build_threads"],
        logger_config=simulator_logging_config,
    )

    output_dir = cwd / "basic_trajectory"
    trajectory_loc = output_dir / "trajectory.pkl"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    def simulator_fn(
        params: jdna_types.Params,
        meta: jdna_types.MetaData,
    ) -> tuple[str, str]:
        simulator.run(params)

        ox_traj = jdna_traj.from_file(
            "data/templates/simple-helix/output.dat",
            strand_lengths=top.strand_counts,
        )
        traj = jdna_sio.SimulatorTrajectory(
            rigid_body=ox_traj.state_rigid_body,
        )

        jdna_tree.save_pytree(traj, trajectory_loc)
        return [trajectory_loc]

    obs_trajectory = "trajectory"

    trajectory_simulator = jdna_simulator.BaseSimulator(
        name="oxdna-sim",
        fn=simulator_fn,
        exposes=[obs_trajectory],
        meta_data={},
    )
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

    propeller_twist_objective = jdna_objective.DiffTReObjective(
        name="DiffTRe",
        required_observables=[obs_trajectory],
        needed_observables=[obs_trajectory],
        logging_observables=["loss", "prop_twist", "neff"],
        grad_or_loss_fn=prop_twist_loss_fn,
        energy_fn_builder=energy_fn_builder,
        opt_params=opt_params,
        trajectory_key=obs_trajectory,
        min_n_eff_factor=0.95,
        beta=jnp.array(1 / kT, dtype=jnp.float64),
        n_equilibration_steps=0,
    )
    # ==========================================================================

    # Logger ===================================================================
    params_to_log = [
        "eps_stack_base",
        "eps_stack_kt_coeff",
        [
            "dr_low_stack",
            "dr_high_stack",
            "a_stack",
            "dr0_stack",
            "dr_c_stack",
        ],
        [
            "theta0_stack_4",
            "delta_theta_star_stack_4",
            "a_stack_4",
        ],
        [
            "theta0_stack_5",
            "delta_theta_star_stack_5",
            "a_stack_5",
        ],
        [
            "theta0_stack_6",
            "delta_theta_star_stack_6",
            "a_stack_6",
        ],
        [
            "neg_cos_phi1_star_stack",
            "a_stack_1",
        ],
        [
            "neg_cos_phi2_star_stack",
            "a_stack_2",
        ],
    ]
    params_list_flat = list(
        itertools.chain.from_iterable(
            [
                [
                    p,
                ]
                if isinstance(p, str)
                else p
                for p in params_to_log
            ]
        )
    )

    logger = console_logger.ConsoleLogger(
        log_dir="logs",
    )
    # ==========================================================================

    # Optimizer ================================================================
    opt = jdna_optimization.SimpleOptimizer(
        objective=propeller_twist_objective,
        simulator=trajectory_simulator,
        optimizer=optax.adam(learning_rate=1e-3),
        logger=logger,
    )
    # ==========================================================================


    # Optimization Loop ========================================================
    for i in range(optimization_config["n_steps"]):
        opt_state, opt_params = opt.step(opt_params)
        log_values = propeller_twist_objective.logging_observables()

        if i % optimization_config["log_every"] == 0:
            for name, value in log_values:
                logger.log_metric(name, value, step=i)

            for params in filter(lambda op: len(op) > 0, opt_params):
                for name in filter(lambda name: name in params_list_flat, params):
                    logger.log_metric(name, params[name], i)

            logger.log_metric("target_ptwist", jd_obs.propeller.TARGETS["oxDNA"], step=i)

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )


if __name__ == "__main__":
    main()
