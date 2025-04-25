import pdb

import jax.numpy as jnp
from jax import jit
from jax_md import quantity, rigid_body


# FIXME: not differentiable. Would be nice if it was.
def get_force_fn(base_energy_or_force_fn, n, displacement_fn, nuc_ids,
                 center_force, orientation_force):

    center_force = jnp.array(center_force)
    assert(center_force.shape == (3,)) # note differentiable
    zero_center_force = jnp.zeros(3)

    orientation_force = jnp.array(orientation_force)
    assert(orientation_force.shape == (4,)) # note differentiable
    zero_orientation_force = jnp.zeros(4)

    base_force_fn = quantity.canonicalize_force(base_energy_or_force_fn)

    # FIXME: make this differentiable with a where
    nuc_ids = set(nuc_ids)
    center_ext_force = jnp.array([center_force if idx in nuc_ids else zero_center_force
                                  for idx in range(n)])
    orientation_ext_force = jnp.array([orientation_force if idx in nuc_ids else zero_orientation_force
                                       for idx in range(n)])

    @jit
    def force_fn(body, **kwargs):
        base_force = base_force_fn(body, **kwargs)
        center = base_force.center + center_ext_force
        orientation_vec = base_force.orientation.vec + orientation_ext_force
        orientation = rigid_body.Quaternion(orientation_vec)
        return rigid_body.RigidBody(center=center, orientation=orientation)

    return base_force_fn, force_fn


if __name__ == "__main__":
    pass
