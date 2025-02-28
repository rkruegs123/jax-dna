"""An example of running a simulation using jax_md.

Important: This assumes that the current working is the root directory of the
repository. i.e. this file was invoked using:

``python -m examples.simulations.jaxmd.jaxmd``
"""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy.dna1 as jdna_energy
import jax_dna.input.topology as jdna_top
import jax_dna.input.trajectory as jdna_traj
import jax_dna.simulators.jax_md as jdna_jaxmd

# the default precision for jax is float32
jax.config.update("jax_enable_x64", True)


def main():
    # configs specific to this file
    run_config = {
        "n_steps": 5_000,
    }

    experiment_dir = Path("data/sys-defs/simple-helix")

    topology = jdna_top.from_oxdna_file(experiment_dir / "sys.top")
    initial_positions = (
        jdna_traj.from_file(
            experiment_dir / "bound_relaxed.conf",
            topology.strand_counts,
        )
        .states[0]
        .to_rigid_body()
    )

    experiment_config, energy_config = jdna_energy.default_configs()

    dt = experiment_config["dt"]
    kT = experiment_config["kT"]
    diff_coef = experiment_config["diff_coef"]
    rot_diff_coef = experiment_config["rot_diff_coef"]

    # These are special values for the jax_md simulator
    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT / diff_coef], dtype=jnp.float64),
        orientation=jnp.array([kT / rot_diff_coef], dtype=jnp.float64),
    )
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([experiment_config["nucleotide_mass"]], dtype=jnp.float64),
        orientation=jnp.array([experiment_config["moment_of_inertia"]], dtype=jnp.float64),
    )

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        jdna_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    # The jax_md simulator needs an energy function. We can use the default
    # energy functions and configurations for dna1 simulations. For more
    # information on energy functions and configurations, see the documentation.
    energy_fn_configs = jdna_energy.default_energy_configs()
    params = [{} for _ in range(len(energy_fn_configs))]
    energy_fns = jdna_energy.default_energy_fns()

    simulator = jdna_jaxmd.JaxMDSimulator(
        energy_configs=energy_fn_configs,
        energy_fns=energy_fns,
        topology=topology,
        simulator_params=jdna_jaxmd.StaticSimulatorParams(
            seq=jnp.array(topology.seq),
            mass=mass,
            bonded_neighbors=topology.bonded_neighbors,
            # this is gradient checkpointing which isn't used in this examples
            checkpoint_every=100,
            dt=dt,
            kT=kT,
            gamma=gamma,
        ),
        space=jax_md.space.free(),
        transform_fn=transform_fn,
        simulator_init=jax_md.simulate.nvt_langevin,
        neighbors=jdna_jaxmd.NoNeighborList(unbonded_nbrs=topology.unbonded_neighbors),
    )

    key = jax.random.PRNGKey(0)
    sim_fn = jax.jit(lambda opts: simulator.run(opts, initial_positions, run_config["n_steps"], key))

    print("Running simulation...")
    trajectory = sim_fn(params)
    print("Simulation Complete! ✅ Trajectory length:", trajectory.rigid_body.center.shape[0])


if __name__ == "__main__":
    main()
