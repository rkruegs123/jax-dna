import functools

import jax
import jax.numpy as jnp
import jax_md
import jax_dna.energy.dna1 as dna1
import jax_dna.energy.base as energy

displacement_fn ,_ = jax_md.space.free()

structure_defaults = dna1.defaults.STRUCTURE


def test_energy_fn(
    opt_params: dict[str, dict[str, float]],
    params: dict[str, dict[str, float]],
    energy_fns: list[energy.BaseEnergyFunction],
) -> float:

    lambda fn: fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))

    transformed_fns = [
        fn(displacement_fn, params.get(fn.__name__, {}), opt_params.get(fn.__name__, {}))
        for fn in energy_fns
    ]

    energy_fn = sum(transformed_fns[1:], start=transformed_fns[0])

    sample_rigid_body = jax_md.rigid_body.RigidBody(
        center = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.5],
            [1.5, 1.5, 1.5],
            [2.5, 2.5, 2.5],
        ]),
        orientation = jax_md.rigid_body.Quaternion(jnp.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ])),
    )

    seq = jnp.array([
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ])

    bonded_neighbors = jnp.array([
        (0, 3),
        (1, 4),
        (2, 5),
    ])

    unbonded_neighbors = jnp.array([
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
        (3, 5),
        (4, 5),
    ])

    energy = energy_fn(
        dna1.Nucleotide.from_rigid_body(
            sample_rigid_body,
            structure_defaults["com_to_backbone"],
            structure_defaults["com_to_hb"],
            structure_defaults["com_to_stacking"],
        ),
        seq,
        bonded_neighbors,
        unbonded_neighbors,
    )

    return energy

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

f = jax.value_and_grad(test_energy_fn)

value, grad = f(opt_params, params, energy_fns)

print(value)
print(grad)

