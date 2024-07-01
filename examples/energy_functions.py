import functools

import jax
import jax.numpy as jnp
import jax_md
import jax_dna.input.topology as topology
import jax_dna.input.trajectory as trajectory
import jax_dna.energy.dna1 as dna1
import jax_dna.energy.base as jdna_energy

jax.config.update("jax_enable_x64", True)

if __name__=="__main__":

    displacement_fn, shift_fn = jax_md.space.free()

    structure_defaults = dna1.defaults.STRUCTURE

    top = topology.from_oxdna_file("data/templates/simple-helix/sys.top")
    seq = jnp.array(top.seq_one_hot)
    traj = trajectory.from_file(
        "data/templates/simple-helix/init.conf",
        top.strand_counts,
    )

    dt = 5e-3
    kT = 296.15 * 0.1 / 300.0

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
        dna1.Nucleotide.from_rigid_body,
        com_to_backbone=structure_defaults["com_to_backbone"],
        com_to_hb=structure_defaults["com_to_hb"],
        com_to_stacking=structure_defaults["com_to_stacking"],
    )

    def test_energy_fn(
        opt_params: dict[str, dict[str, float]],
        params: dict[str, dict[str, float]],
        energy_fns: list[jdna_energy.BaseEnergyFunction],
    ) -> float:
        pass

        lambda fn: fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))

        transformed_fns = [
            fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))
            for fn in energy_fns
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
        opt_params: dict[str, dict[str, float]],
        params: dict[str, dict[str, float]],
        energy_fns: list[jdna_energy.BaseEnergyFunction],
    ) -> float:

        lambda fn: fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))

        transformed_fns = [
            fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))
            for fn in energy_fns
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

    w_defaults = lambda cls: functools.partial(cls, displacement_fn=displacement_fn, params={})

    opt_params = {
        dna1.bonded.Fene.__name__: {"eps_backbone": 1.0},
    }

    params = {
        dna1.bonded.Stacking.__name__: {"ss_stack_weights": dna1.defaults.STRUCTURE["stack_weights_sa"]},
        dna1.unbonded.HydrogenBonding.__name__: {"hb_weights": dna1.defaults.STRUCTURE["hb_weights_sa"]},
    }

    energy_fns = [
        dna1.bonded.Fene,
        dna1.bonded.ExcludedVolume,
        dna1.bonded.Stacking,
        dna1.unbonded.ExcludedVolume,
        dna1.unbonded.HydrogenBonding,
        dna1.unbonded.CrossStacking,
        dna1.unbonded.CoaxialStacking,
    ]

    print(test_energy_fn(opt_params, params, energy_fns))

    f = jax.value_and_grad(test_energy_grad_fn)
    value, grad = f(opt_params, params, energy_fns)
    print(value)
    print(grad)

