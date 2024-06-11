"""Math utilities for DNA sequence analysis."""

import jax.numpy as jnp
import numpy as np

import jax_dna.utils.types as typ


def principal_axes_to_euler_angles(
    x: typ.Arr_Nucleotide_3,
    y: typ.Arr_Nucleotide_3,
    z: typ.Arr_Nucleotide_3,
) -> tuple[typ.Arr_Nucleotide, typ.Arr_Nucleotide, typ.Arr_Nucleotide]:
    """Convert principal axes to Tait-Bryan Euler angles.

    A utility function for converting a set of principal axes
    (that define a rotation matrix) to a commonly used set of
    Tait-Bryan Euler angles.

    There are two options to compute the Tait-Bryan angles. Each can be seen at the respective links:
    (1) From wikipedia (under Tait-Bryan angles): https://en.wikipedia.org/wiki/Euler_angles
    (2) Equation 10A-C: https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    However, note that the definition from Wikipedia (i.e. the one using arcsin) has numerical stability issues,
    so we use the definition from (2) (i.e. the one using arctan2)

    Note that if we were following (1), we would do:
    psi = onp.arcsin(x[1] / onp.sqrt(1 - x[2]**2))
    theta = onp.arcsin(-x[2])
    phi = onp.arcsin(y[2] / onp.sqrt(1 - x[2]**2))

    Note that Tait-Bryan (i.e. Cardan) angles are *not* proper euler angles
    """
    psi = np.arctan2(x[:, 1], x[:, 0])
    theta = np.arcsin(-np.clip(x[:, 2], -1, 1))
    phi = np.arctan2(y[:, 2], z[:, 2])

    return (psi, theta, phi)


def euler_angles_to_quaternion(
    psi: typ.Arr_Nucleotide,
    theta: typ.Arr_Nucleotide,
    phi: typ.Arr_Nucleotide,
) -> typ.Arr_Nucleotide_4:
    """Convert Euler angles to quaternions.

    A utility function for converting euler angles to quaternions.
    Used when converting a trajectory DataFrame to a set of states.

    We follow the ZYX convention. For details, see page A-11 in
    https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    from the following set of documentation:
    https://ntrs.nasa.gov/citations/19770024290
    """
    sin_psi, cos_psi = np.sin(0.5 * psi), np.cos(0.5 * psi)
    sin_theta, cos_theta = np.sin(0.5 * theta), np.cos(0.5 * theta)
    sin_phi, cos_phi = np.sin(0.5 * phi), np.cos(0.5 * phi)

    q0 = sin_psi * sin_theta * sin_phi + cos_psi * cos_theta * cos_phi
    q1 = -sin_psi * sin_theta * cos_phi + sin_phi * cos_psi * cos_theta
    q2 = sin_psi * cos_theta * sin_phi + cos_psi * sin_theta * cos_phi
    q3 = sin_psi * cos_theta * cos_phi - cos_psi * sin_theta * sin_phi

    return np.array([q0, q1, q2, q3]).T


def smooth_abs(x: typ.ARR_OR_SCALAR, eps=1e-10) -> typ.ARR_OR_SCALAR:
    """A smooth absolute value function.
    Note that a non-zero eps gives continuous first dervatives.

    https://math.stackexchange.com/questions/1172472/differentiable-approximation-of-the-absolute-value-function
    """
    return jnp.sqrt(x**2 + eps)


def clamp(x, lo=-1.0, hi=1.0):
    """
    correction = 1e-10
    min_ = jnp.where(x + 1e-10 > hi, hi, x)
    max_ = jnp.where(min_ - 1e-10 < lo, lo, min_)
    """

    min_ = jnp.where(x >= hi, hi, x)
    max_ = jnp.where(min_ <= lo, lo, min_)
    return max_
    # return jnp.clip(x, lo, hi)


def mult(a: typ.Arr_N, b: typ.Arr_N) -> typ.Arr_N:
    return jnp.einsum("ij, ij->i", a, b)
