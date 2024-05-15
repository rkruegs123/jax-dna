"""Math utilities for DNA sequence analysis."""

import jax.numpy as jnp

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
    psi = jnp.arctan2(x[:, 1], x[:, 0])
    theta = jnp.arcsin(-jnp.clip(x[:, 2], -1, 1))
    phi = jnp.arctan2(y[:, 2], z[:, 2])

    return (psi, theta, phi)


def euler_angles_to_quaternion(
    t1: typ.Arr_Nucleotide,
    t2: typ.Arr_Nucleotide,
    t3: typ.Arr_Nucleotide,
) -> typ.Arr_Nucleotide_4:
    """Convert Euler angles to quaternions.

    A utility function for converting euler angles to quaternions.
    Used when converting a trajectory DataFrame to a set of states.

    We follow the ZYX convention. For details, see page A-11 in
    https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
    from the following set of documentation:
    https://ntrs.nasa.gov/citations/19770024290
    """
    sin_t1, cos_t1 = jnp.sin(0.5 * t1), jnp.cos(0.5 * t1)
    sin_t2, cos_t2 = jnp.sin(0.5 * t2), jnp.cos(0.5 * t2)
    sin_t3, cos_t3 = jnp.sin(0.5 * t3), jnp.cos(0.5 * t3)

    q0 = sin_t1 * sin_t2 * sin_t3 + cos_t1 * cos_t2 * cos_t3
    q1 = -sin_t1 * sin_t2 * cos_t3 + sin_t3 * cos_t1 * cos_t2
    q2 = sin_t1 * cos_t2 * sin_t3 + cos_t1 * sin_t2 * cos_t3
    q3 = sin_t1 * cos_t2 * cos_t3 - cos_t1 * sin_t2 * sin_t3

    return jnp.array([q0, q1, q2, q3]).T
