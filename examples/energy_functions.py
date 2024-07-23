import functools

import jax
import jax.numpy as jnp
import jax_md
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.input.configuration as config
import jax_dna.input.dna1.bonded as  dna1_bonded_config
import jax_dna.input.dna1.unbonded as dna1_unbonded_config
import jax_dna.energy.dna1 as dna1_energy
import jax_dna.energy.base as jdna_energy
import jax_dna.utils.types as jdt

jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    topology_fname = "data/templates/simple-helix/sys.top"
    traj_fname = "data/templates/simple-helix/init.conf"
    config_fname = "jax_dna/input/dna1/defaults.toml"



    displacement_fn, shift_fn = jax_md.space.free()

    top = topology.from_oxdna_file(topology_fname)
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        traj_fname,
        top.strand_counts,
    )
    experiment_config = config.BaseConfiguration.parse_toml(config_fname)

    dt = experiment_config["dt"]
    kT = experiment_config["t_kelvin"]

    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64),
    )
    beta = 1 / kT
    gamma = jax_md.rigid_body.RigidBody(
        center=jnp.array([kT/2.5], dtype=jnp.float64),
        orientation=jnp.array([kT/7.5], dtype=jnp.float64),
    )
    nucleotide_mass = 1.0
    moment_of_inertia = [1.0, 1.0, 1.0]
    mass = jax_md.rigid_body.RigidBody(
        center=jnp.array([nucleotide_mass], dtype=jnp.float64),
        orientation=jnp.array([moment_of_inertia], dtype=jnp.float64),
    )

    init_body = traj.states[0].to_rigid_body()

    key = jax.random.PRNGKey(0)

    transform_fn = functools.partial(
        dna1_energy.Nucleotide.from_rigid_body,
        com_to_backbone=experiment_config["com_to_backbone"],
        com_to_hb=experiment_config["com_to_hb"],
        com_to_stacking=experiment_config["com_to_stacking"],
    )


    config_file = "jax_dna/input/dna1/defaults.toml"
    configs = [
        # single param
        dna1_bonded_config.FeneConfiguration.from_toml(config_file, ("eps_backbone",)),
        # multiple params
        dna1_bonded_config.ExcludedVolumeConfiguration.from_toml(config_file, ("eps_exc","dr_star_base")),
        # shortcut for all params, though you could list them all too
        dna1_bonded_config.StackingConfiguration.from_toml(config_file, ("*",)),
        dna1_unbonded_config.ExcludedVolumeConfiguration.from_toml(config_file),
        dna1_unbonded_config.HydrogenBondingConfiguration.from_toml(config_file),
        dna1_unbonded_config.CrossStackingConfiguration.from_toml(config_file),
        dna1_unbonded_config.CoaxialStackingConfiguration.from_toml(config_file),
    ]



    opt_params = [c.opt_params for c in configs]

    # configs and energy functions should be in the same order
    # though maybe we can be more clever in the future
    energy_fns = [
        dna1_energy.bonded.Fene,
        dna1_energy.bonded.ExcludedVolume,
        dna1_energy.bonded.Stacking,
        dna1_energy.unbonded.ExcludedVolume,
        dna1_energy.unbonded.HydrogenBonding,
        # dna1_energy.unbonded.CrossStacking,
        # dna1_energy.unbonded.CoaxialStacking,
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
            transformed_fns,
            rigid_body_transform_fn=transform_fn,
        )

        energy = energy_fn(
            init_body,
            seq=seq,
            bonded_neighbors=top.bonded_neighbors,
            unbonded_neighbors=top.unbonded_neighbors.T,
        )

        return energy


    def test_energy_grad_fn(
        params: list[config.BaseConfiguration],
    ) -> float:
        transformed_fns = [
            energy_fn(
                displacement_fn=displacement_fn,
                params=(config | param).init_params(),
            )
            for param, config, energy_fn in zip(params, configs, energy_fns)
        ]

        energy_fn = jdna_energy.ComposedEnergyFunction(
            transformed_fns,
            rigid_body_transform_fn=transform_fn,
        )

        init_fn, step_fn = jax_md.simulate.nvt_langevin(
            energy_fn,
            shift_fn,
            dt,
            kT,
            gamma,
        )

        init_state = init_fn(
            key,
            init_body,
            mass=mass,
            seq=seq,
            bonded_neighbors=top.bonded_neighbors,
            unbonded_neighbors=top.unbonded_neighbors.T,
        )

        next_state = step_fn(
            init_state,
            seq=seq,
            bonded_neighbors=top.bonded_neighbors,
            unbonded_neighbors=top.unbonded_neighbors.T,
        )

        return next_state.position.center.sum()



    print("jitting test_energy_fn")
    f = jax.jit(test_energy_fn)
    print("running test_energy_fn")
    print(f(opt_params))

    print("gradding test_energy_grad_fn")
    f = jax.grad(test_energy_grad_fn)
    print("jitting test_energy_grad_fn")
    f = jax.jit(jax.value_and_grad(test_energy_grad_fn))
    print("running test_energy_grad_fn")
    print(f(opt_params))




