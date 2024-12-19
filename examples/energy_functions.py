import functools

import jax
import jax.numpy as jnp
import jax_md
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.toml as toml_reader
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.base as jdna_energy
import jax_dna.energy.configuration as jdna_energy_config
import jax_dna.utils.types as jdt

jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    topology_fname = "data/templates/simple-helix/sys.top"
    traj_fname = "data/templates/simple-helix/init.conf"
    simulation_config = "jax_dna/input/dna1/default_simulation.toml"
    energy_config = "jax_dna/input/dna1/default_energy.toml"

    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )
    experiment_config = toml_reader.parse_toml(simulation_config)
    energy_config = toml_reader.parse_toml(energy_config)

    displacement_fn, shift_fn = jax_md.space.free()
    key = jax.random.PRNGKey(0)

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

    geometry = energy_config["geometry"]
    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=geometry["com_to_backbone"],
        com_to_hb=geometry["com_to_hb"],
        com_to_stacking=geometry["com_to_stacking"],
    )

    configs = [
        # single param
        dna1_energy.FeneConfiguration.from_dict(energy_config["fene"], ("eps_backbone",)),
        # multiple params
        dna1_energy.BondedExcludedVolumeConfiguration.from_dict(energy_config["bonded_excluded_volume"], ("eps_exc","dr_star_base")),
        # shortcut for all params, though you could list them all too
        dna1_energy.StackingConfiguration.from_dict(energy_config["stacking"] | {"kt": kT }, ("*",)),
        dna1_energy.UnbondedExcludedVolumeConfiguration.from_dict(energy_config["unbonded_excluded_volume"]),
        dna1_energy.HydrogenBondingConfiguration.from_dict(energy_config["hydrogen_bonding"]),
        dna1_energy.CrossStackingConfiguration.from_dict(energy_config["cross_stacking"]),
        dna1_energy.CoaxialStackingConfiguration.from_dict(energy_config["coaxial_stacking"]),
    ]

    opt_params = [c.opt_params for c in configs]

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

    def test_energy_fn(
        params: list[dict[str, jdt.ARR_OR_SCALAR]],
    ) -> float:

        transformed_fns = [
            energy_fn(
                displacement_fn=displacement_fn,
                params=(config | param).init_params(),
            )
            for param, config, energy_fn in zip(params, configs, energy_fns)
        ]

        energy_fn = jdna_energy.ComposedEnergyFunction(
            energy_fns=transformed_fns,
            rigid_body_transform_fn=transform_fn,
        )

        return lambda R: energy_fn(
            R,
            seq=seq,
            bonded_neighbors=top.bonded_neighbors,
            unbonded_neighbors=top.unbonded_neighbors.T,
        )

    print("init_body", init_body.center.shape, init_body.orientation.vec.shape)
    print("initial energy", test_energy_fn(opt_params)(init_body))

    force_fn = jax.grad(test_energy_fn(opt_params))
    force = force_fn(init_body)

    print("force", force.center.shape, force.orientation.vec.shape)

    jit_force_fn = jax.jit(force_fn)
    force = jit_force_fn(init_body)

    print("force", force.center.shape, force.orientation.vec.shape)
