from jax_md import util
from jax_md.rigid_body import RigidBody

from bonded_energy import static_energy_fn_factory
from unbonded_energy import dynamic_energy_fn_factory_fixed

def energy_fn_factory(displacement_fn,
                      back_site, stack_site, base_site,
                      bonded_neighbors, unbonded_neighbors):
    static_energy_fn, _ = static_energy_fn_factory(displacement,
                                                   back_site=back_site,
                                                   stack_site=stack_site,
                                                   base_site=base_site,
                                                   neighbors=bonded_neighbors)
    dynamic_energy_fn, _ = dynamic_energy_fn_factory_fixed(
        displacement,
        back_site=back_site,
        stack_site=stack_site,
        base_site=base_site,
        neighbors=unbonded_neighbors
    )

    def energy_fn(body: RigidBody, seq: util.Array, params) -> float:
        return static_energy_fn(body, seq, params) + dynamic_energy_fn(body, seq, params)

    return energy_fn
