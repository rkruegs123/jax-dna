"""An example of running a simulation using oxDNA.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simple_optimizations.oxdna.oxDNA``
"""


import functools
import itertools
import logging
import os
from pathlib import Path
import typing
import jax
import jax.numpy as jnp
import jax_md
import optax
import ray
import sys
from tqdm import tqdm


import jax_dna
import jax_dna.energy as jdna_energy
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.input.toml as toml_reader
import jax_dna.input.tree as jdna_tree
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

jax.config.update("jax_enable_x64", True)

# You need to either set the `oxdna` executable/build paths here or somewhere else
os.environ[oxdna.BIN_PATH_ENV_VAR] = str(Path("../oxDNA/build/bin/oxDNA").resolve())
os.environ[oxdna.BUILD_PATH_ENV_VAR] =  str(Path("../oxDNA/build").resolve())

def main():

    optimization_config = {
        "n_steps": 1000,
        "batch_size": 1,
    }

    simulation_config, energy_config = dna1_energy.default_configs()
    kT = simulation_config["kT"]

    energy_fns = dna1_energy.default_energy_fns()
    energy_fn_configs = dna1_energy.default_energy_configs()
    opt_params = []
    for ec in energy_fn_configs:
        opt_params.append(
            ec.opt_params if isinstance(ec, dna1_energy.StackingConfiguration) else {}
        )

    for op in opt_params:
        if "ss_stack_weights" in op:
            del op["ss_stack_weights"]

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    energy_fn_builder_fn = jdna_energy.energy_fn_builder(
        energy_fns=energy_fns,
        energy_configs=energy_fn_configs,
        transform_fn=transform_fn,
    )

    topology_fname = "data/templates/simple-helix/sys.top"
    top = topology.from_oxdna_file(topology_fname)

    def energy_fn_builder(params: jdna_types.Params) -> callable:
        return jax.vmap(
            lambda trajectory: energy_fn_builder_fn(params)(
                trajectory.rigid_body,
                seq=jnp.array(top.seq),
                bonded_neighbors=top.bonded_neighbors,
                unbonded_neighbors=top.unbonded_neighbors.T,
            )
        )

    # setup the simulator
    input_dir = "data/templates/simple-helix"
    simulator = oxdna.oxDNASimulator(
        input_dir=input_dir,
        sim_type=jdna_types.oxDNASimulatorType.DNA1,
        energy_configs=energy_fn_configs,
        n_build_threads=4,
        disable_build=False,
    )

    cwd = Path(os.getcwd())
    output_dir = cwd / "basic_trajectory"
    trajectory_loc = output_dir / "trajectory.pkl"
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    def simulator_fn(
        params: jdna_types.Params,
        meta: jdna_types.MetaData,
    ) -> tuple[str, str]:
        traj = simulator.run(params)
        p = Path("energies")
        p.mkdir(parents=True, exist_ok=True)
        n = len(list(p.glob("*.npy")))
        jnp.save(f"energies-{n}.npy", energy_fn_builder(params)(traj))
        jdna_tree.save_pytree(traj, trajectory_loc)
        return [trajectory_loc]

    obs_trajectory = "trajectory"

    trajectory_simulator = jdna_simulator.BaseSimulator(
        name="oxdna-sim",
        fn=simulator_fn,
        exposes = [obs_trajectory],
        meta_data = {},
    )

    prop_twist_fn = jd_obs.propeller.PropellerTwist(
        rigid_body_transform_fn=transform_fn,
        h_bonded_base_pairs=jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    )

    def prop_twist_loss_fn(
        traj: jax_md.rigid_body.RigidBody,
        weights: jnp.ndarray,
        energy_model: jdna_energy.base.ComposedEnergyFunction,
    ) -> tuple[float, tuple[str, typing.Any]]:
        obs = prop_twist_fn(traj)
        expected_prop_twist = jnp.dot(weights, obs)
        loss = (expected_prop_twist - jd_obs.propeller.TARGETS["oxDNA"])**2
        loss = jnp.sqrt(loss)
        return loss, (("prop_twist", expected_prop_twist), {})


    propeller_twist_objective = jdna_objective.DiffTReObjective(
        name = "DiffTRe",
        required_observables = [obs_trajectory],
        needed_observables = [obs_trajectory],
        logging_observables = ["loss", "prop_twist"],
        grad_or_loss_fn = prop_twist_loss_fn,
        energy_fn_builder = energy_fn_builder,
        opt_params = opt_params,
        min_n_eff_factor = 0.95,
        beta = jnp.array(1/kT),
        n_equilibration_steps = 0, # periodic steps are already in oxdna
    )

    opt = jdna_optimization.SimpleOptimizer(
        objective=propeller_twist_objective,
        simulator=trajectory_simulator,
        optimizer = optax.adam(learning_rate=1e-3),
    )


    for i in range(optimization_config["n_steps"]):
        opt_state, opt_params = opt.step(opt_params)

        if i % 5 == 0:
            log_values = propeller_twist_objective.logging_observables()
            for (name, value) in log_values:
                print(f"{i}::{name}={value}")

        opt = opt.post_step(
            optimizer_state=opt_state,
            opt_params=opt_params,
        )

if __name__=="__main__":
    main()