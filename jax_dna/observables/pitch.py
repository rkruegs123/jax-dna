"""Pitch observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types

TARGETS = {
    "oxDNA": 10.5,  # bp/turn
}


def compute_pitch(avg_pitch_angle: float) -> float:
    """Computes the pitch given an average pitch angle in radians.

    Args:
        avg_pitch_angle (float): a value in radians specifying the pitch value
            averaged over a trajectory

    Returns:
        float: the pitch value in base pairs per turn
    """
    return jnp.pi / avg_pitch_angle


def single_pitch_angle(
    quartet: jnp.ndarray, base_sites: jnp.ndarray, back_sites: jnp.ndarray, displacement_fn: Callable
) -> jd_types.ARR_OR_SCALAR:
    """Computes the pitch angle between adjacent base pairs."""
    # Extract the base pairs
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2

    # Compute the local helical axis
    local_helix_dir = jd_obs.local_helical_axis(quartet, base_sites, displacement_fn)

    # Compute the vector between backbone sites for each base pair
    bb1 = displacement_fn(back_sites[b1], back_sites[a1])
    # Do we need this? bb1_dir = bb1 / jnp.linalg.norm(bb1)
    bb2 = displacement_fn(back_sites[b2], back_sites[a2])
    # Do we need this? bb2_dir = bb2 / jnp.linalg.norm(bb2)

    # Project each vector onto the local helical axis
    bb1_proj = displacement_fn(bb1, jnp.dot(local_helix_dir, bb1) * local_helix_dir)
    bb1_proj_dir = bb1_proj / jnp.linalg.norm(bb1_proj)
    bb2_proj = displacement_fn(bb2, jnp.dot(local_helix_dir, bb2) * local_helix_dir)
    bb2_proj_dir = bb2_proj / jnp.linalg.norm(bb2_proj)

    # Compute the angle between these projections
    return jnp.arccos(jd_math.clamp(jnp.dot(bb1_proj_dir, bb2_proj_dir)))


single_pitch_angle_mapped = jax.vmap(single_pitch_angle, in_axes=(0, None, None, None))


@chex.dataclass(frozen=True, kw_only=True)
class PitchAngle(jd_obs.BaseObservable):
    """Computes the average pitch angle in radians for each state.

    The pitch is defined by (2*pi) / <angle> where <angle> is the average angle
    between adjacent base pairs across states

    Args:
    - quartets: a (n_quartets, 2, 2) array containing the pairs of adjacent base pairs
      for which to compute pitch angles
    - displacement_fn: a function for computing displacements between two positions
    """

    quartets: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the average pitch angle in radians.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the pitch for

        Returns:
            jd_types.ARR_OR_SCALAR: the average pitch angle in radians for each state,
            so expect a size of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        base_sites = nucleotides.base_sites
        back_sites = nucleotides.back_sites

        angles = jax.vmap(single_pitch_angle_mapped, (None, 0, 0, None))(
            self.quartets, base_sites, back_sites, self.displacement_fn
        )
        return jnp.mean(angles, axis=1)
