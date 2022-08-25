from jax_md import util
from jax_md.rigid_body import RigidBody

from nn_energy import nn_energy_fn_factory
from other_pairs_energy import other_pairs_energy_fn_factory_fixed

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

    def energy_fn(body: RigidBody, seq: util.Array, params) -> float:
        return nn_energy_fn(body, seq, params) + other_pairs_energy_fn(body, seq, params)

    return energy_fn
