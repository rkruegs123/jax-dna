"""Propeller twist observable."""

import dataclasses as dc

import chex
import jax
import jax.numpy as jnp

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types

TARGETS = {
    "oxDNA": 21.7, # degrees
}


def single_propeller_twist_rad(
    bp: jnp.ndarray,  # this is a array of shape (2,) containing the indices of the h-bonded nucleotides
    base_normals: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the propeller twist of a base pair."""
    # get the normal vectors of the h-bonded bases
    bp1, bp2 = bp
    nv1 = base_normals[bp1]
    nv2 = base_normals[bp2]

    # compute angle between base normal vectors
    return jnp.arccos(jd_math.clamp(jnp.dot(nv1, nv2)))


propeller_twist_rad = jax.vmap(single_propeller_twist_rad, in_axes=(0, None))


@chex.dataclass(frozen=True)
class PropellerTwist(jd_obs.BaseObservable):
    """Computes the propeller twist of a base pair.

    The propeller twist is defined as the angle between the normal vectors
    of h-bonded bases

    Args:
    - bp: a 2-dimensional array containing the indices of the h-bonded nucleotides
    - base_normals: the base normal vectors of the entire body
    """

    h_bonded_base_pairs: jnp.ndarray = dc.field(
        hash=False
    )  # a 2-dimensional array containing the indices of the h-bonded nucleotides

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the twist of the propeller in degrees.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the propeller twist for

        Returns:
            jd_types.ARR_OR_SCALAR: the propeller twist in degrees for each state , so expect a size
            of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        base_normals = nucleotides.base_normals
        ptwist = jax.vmap(lambda bn: 180.0 - (propeller_twist_rad(self.h_bonded_base_pairs, bn) * 180.0 / jnp.pi))
        return jnp.mean(ptwist(base_normals), axis=1)
