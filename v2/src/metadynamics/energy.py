import pdb
import jax.numpy as jnp

from jax_md import util
from jax_md.rigid_body import RigidBody

from metadynamics.utils import sum_of_gaussians


def factory(energy_fn, cv_fn):
    def metad_energy_fn(body: RigidBody,
                        heights: util.Array, centers: util.Array, widths: util.Array,
                        **kwargs) -> float:
        cv = cv_fn(body)
        # return energy_fn(body, **kwargs) + sum_of_gaussians(heights, centers, widths, cv)
        return sum_of_gaussians(heights, centers, widths, cv)
    return metad_energy_fn


if __name__ == "__main__":
    pass
