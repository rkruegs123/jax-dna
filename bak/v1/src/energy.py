import numpy as onp

from jax_md import util
from jax_md.rigid_body import RigidBody
from jax_md import rigid_body
from jax_md import energy

from nn_energy import nn_energy_fn_factory
from other_pairs_energy import other_pairs_energy_fn_factory_fixed


f32 = util.f32

def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors):
    nn_energy_fn, _ = nn_energy_fn_factory(
        displacement_fn,
        back_site=back_site,
        stack_site=stack_site,
        base_site=base_site,
        neighbors=bonded_neighbors)
    other_pairs_energy_fn, _ = other_pairs_energy_fn_factory_fixed(
        displacement_fn,
        back_site=back_site,
        stack_site=stack_site,
        base_site=base_site,
        neighbors=unbonded_neighbors
    )

    def energy_fn(body: RigidBody, seq: util.Array, params, **kwargs) -> float:
        return nn_energy_fn(body, seq, params, **kwargs) + other_pairs_energy_fn(body, seq, params, **kwargs)

    # monomer = rigid_body.point_union_shape(onp.array([[0.0, 0.0, 0.0]], f32), f32(1.0))
    # energy_fn = rigid_body.point_energy(energy.soft_sphere_pair(displacement_fn),
                                        # monomer)

    return energy_fn
