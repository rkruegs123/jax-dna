"""Utility functions for computing stretch-torsion moduli."""

import dataclasses as dc
import functools
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types


@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def single_angle_xy(quartet: jnp.ndarray, base_sites: jnp.ndarray, displacement_fn: Callable) -> jd_types.ARR_OR_SCALAR:
    """Computes the angle in the X-Y plane between adjacent base pairs."""
    # Extract the base pairs
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2

    # Compute the vector between base sites for each base pair
    bb1 = displacement_fn(base_sites[b1], base_sites[a1])
    bb2 = displacement_fn(base_sites[b2], base_sites[a2])

    # Omit the z-direction from normalization
    bb1 = bb1[:2]
    bb2 = bb2[:2]

    # Normalize
    bb1 = bb1 / jnp.linalg.norm(bb1)
    bb2 = bb2 / jnp.linalg.norm(bb2)

    # Compute
    return jnp.arccos(jd_math.clamp(jnp.dot(bb1, bb2)))


@chex.dataclass(frozen=True, kw_only=True)
class TwistXY(jd_obs.BaseObservable):
    """Computes the total twist of a duplex in the X-Y plane in radians.

    The total twist of a duplex is defined as the sum of angles in the X-Y plane between
    adjacent base pairs.

    Args:
    - quartets: a (n_quartets, 2, 2) array containing the pairs of adjacent base pairs
    - displacement_fn: a function for computing displacements between two positions
    """

    quartets: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the total twist in the X-Y plane in radians.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory

        Returns:
            jd_types.ARR_OR_SCALAR: the total twist in radians for each state, so expect
            a size of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        base_sites = nucleotides.base_sites

        angles = jax.vmap(single_angle_xy, (None, 0, 0, None))(self.quartets, base_sites, self.displacement_fn)
        return jnp.sum(angles, axis=1)


def single_extension_z(
    center: jd_types.Arr_Nucleotide_3,
    bp1: jnp.ndarray,
    bp2: jnp.ndarray,
) -> jd_types.ARR_OR_SCALAR:
    """Computes the distance between the midpoints of two base pairs."""
    # Extract the base pair indices
    a1, b1 = bp1
    a2, b2 = bp2

    # Compute the midpoints of each base pair
    bp1_midp = (center[a1] + center[b1]) / 2
    bp2_midp = (center[a2] + center[b2]) / 2

    # Compute the extension between the two base pairs in the Z-direction
    return jnp.abs(bp1_midp[2] - bp2_midp[2])


@chex.dataclass(frozen=True, kw_only=True)
class ExtensionZ(jd_obs.BaseObservable):
    """Computes the total extension of a duplex in the Z-direction in simulation units.

    The total extension of a duplex is defined as the distance between the midpoints of
    two pre-specified base pairs in the Z-direction.

    Args:
    - bp1: a (2,) array specifying the indices of the first base pair
    - bp2: a (2,) array specifying the indices of the second base pair
    - displacement_fn: a function for computing displacements between two positions
    """

    bp1: jnp.ndarray = dc.field(hash=False)
    bp2: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the total extension in simulation units.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory

        Returns:
            jd_types.ARR_OR_SCALAR: the total extension for each state, so expect a size
            of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        center = nucleotides.center

        # return the extensions
        return jax.vmap(single_extension_z, (None, 0, 0, None))(center, self.bp1, self.bp2, self.displacement_fn)


def stretch(forces: jnp.ndarray, extensions: jnp.ndarray) -> tuple[float, float, float]:
    r"""Computes the effective stretch modulus and relevant summary statistics from stretch experiments.

    Following Assenza and Perez (JCTC 2022), the effective stretch modulus can be computed as

    .. math::
      \tilde{S} = \frac{L_0}{A_1}

    where `A_1` and `L_0` are the slope and offset, respectively, of a linear force-extension fit.

    Args:
        forces (jnp.ndarray): the forces applied to the polymer
        extensions (jnp.ndarray): the equilibrium extensions under the applied forces

    Returns:
        Tuple[float, float, float]: the slope and offset of the linear fit, and the effective stretch modulus
    """
    # Format the forces for line-fitting
    forces_ = jnp.stack([jnp.ones_like(forces), forces], axis=1)

    # Fit a line
    # Note: we do not fix l0 to be the extension under 0 force. We fit it as a parameter.
    fit_ = jnp.linalg.lstsq(forces_, extensions)

    # Extract statistics
    a1 = fit_[0][1]
    l0 = fit_[0][0]  # Note: this is the equilibrium extension at 0 force and torque, *not* the contour length

    # Compute effective stretch modulus
    s_eff = l0 / a1
    return a1, l0, s_eff


def torsion(torques: jnp.ndarray, extensions: jnp.ndarray, twists: jnp.ndarray) -> tuple[float, float]:
    """Computes the relevant summary statistics from torsion experiments.

    Following Assenza and Perez (JCTC 2022), the torsional modulus and twist-stretch coupling can be
    computed via linear fits to the extension and twist of a duplex under torque (when combined with
    similar statistics from stretching experiments). This function computes the slopes of these
    linear fits

    Args:
        torques (jnp.ndarray): the torques applied to the polymer
        extensions (jnp.ndarray): the equilibrium extensions under the applied torques
        twists (jnp.ndarray): the equilibrium twists under the applied torques

    Returns:
        Tuple[float, float]: the slopes of the linear fits to the extensions and twists, respectively
    """
    # Format the torques for line-fitting
    torques_ = jnp.stack([jnp.ones_like(torques), torques], axis=1)

    # Fit a line to the extensions
    fit_ = jnp.linalg.lstsq(torques_, extensions)
    a3 = fit_[0][1]

    # Fit a line to the twists
    fit_ = jnp.linalg.lstsq(torques_, twists)
    a4 = fit_[0][1]

    return a3, a4


def stretch_torsion(
    forces: jnp.ndarray,
    force_extensions: jnp.ndarray,
    torques: jnp.ndarray,
    torque_extensions: jnp.ndarray,
    torque_twists: jnp.ndarray,
) -> tuple[float, float, float]:
    """Computes the effective stretch and torsion moduli, and twist-stretch coupling from stretch-torsion experiments.

    Args:
        forces (jnp.ndarray): the forces applied to the polymer
        force_extensions (jnp.ndarray): the equilibrium extensions under the applied forces
        torques (jnp.ndarray): the torques applied to the polymer
        torque_extensions (jnp.ndarray): the equilibrium extensions under the applied torques
        torque_twists (jnp.ndarray): the equilibrium twists under the applied torques

    Returns:
        Tuple[float, float, float]: the effective stretch modulus, torsional modulus, and twist-stretch coupling
    """
    # Compute the effective stretch modulus and relevant summary statistics from stretching experiments
    a1, l0, s_eff = stretch(forces, force_extensions)

    # Compute the relevant summary statistics from torsion experiments
    a3, a4 = torsion(torques, torque_extensions, torque_twists)

    # Compute the torsional modulus and twist-stretch coupling
    c = a1 * l0 / (a4 * a1 - a3**2)
    g = -(a3 * l0) / (a4 * a1 - a3**2)

    return s_eff, c, g
