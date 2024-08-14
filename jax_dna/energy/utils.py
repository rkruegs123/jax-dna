"""Utility functions for energy calculations for DNA1."""

import functools

import jax
import jax.numpy as jnp
import jax_md


@jax.vmap
def q_to_back_base(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the vector from the center to the base of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)])


@jax.vmap
def q_to_base_normal(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the normal vector to the base of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0**2 - q1**2 - q2**2 + q3**2])


@jax.vmap
def q_to_cross_prod(q: jax_md.rigid_body.Quaternion) -> jnp.ndarray:
    """Get the cross product vector of the nucleotide."""
    q0, q1, q2, q3 = q.vec
    return jnp.array([2 * (q1 * q2 - q0 * q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2 * q3 + q0 * q1)])


@functools.partial(jax.vmap, in_axes=(None, 0, 0), out_axes=0)
def get_pair_probs(seq: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
    """Get the pair probabilities for a sequence."""
    return jnp.kron(seq[i], seq[j])
