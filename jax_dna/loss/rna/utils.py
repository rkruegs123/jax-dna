# ruff: noqa
import functools
import pdb

import jax.numpy as jnp
from jax import jit, vmap


@jit
def fit_plane(points):
    """Fits plane through points, return normal to the plane."""

    n_points = points.shape[0]
    rc = jnp.sum(points, axis=0) / n_points
    points_centered = points - rc

    As = vmap(lambda point: jnp.kron(point, point).reshape(3, 3))(points_centered)
    A = jnp.sum(As, axis=0)
    vals, vecs = jnp.linalg.eigh(A)
    return vecs[:, 0]


@functools.partial(jit, static_argnums=(5, 6))
def get_localized_axis(body, back_sites, stack_sites, base_sites, base_id, down_length, up_length):
    n = body.center.shape[0]

    def get_base_plane_nuc_info(i):
        midpoint_A = 0.5 * (stack_sites[i] + stack_sites[n - i - 1])
        midpoint_B = 0.5 * (stack_sites[n - i - 1 - 1] + stack_sites[i + 1])

        guess = -midpoint_A + midpoint_B

        p1 = back_sites[i] - back_sites[i + 1]
        p2 = back_sites[n - i - 1] - back_sites[n - i - 1 - 1]
        p3 = midpoint_A - midpoint_B
        return guess, jnp.array([p1, p2, p3])

    n_base_plane_idxs = 1 + down_length + up_length
    base_plane_idxs = base_id - down_length + jnp.arange(n_base_plane_idxs)

    # base_plane_idxs = jnp.arange(base_id-down_length, base_id+up_length+1)
    guesses, plane_points = vmap(get_base_plane_nuc_info)(base_plane_idxs)
    plane_points = plane_points.reshape(-1, 3)

    mean_guess = jnp.mean(guesses, axis=0)
    mean_guess = mean_guess / jnp.linalg.norm(mean_guess)
    plane_vector = fit_plane(plane_points)

    plane_vector = jnp.where(jnp.dot(mean_guess, plane_vector) < 0, -1.0 * plane_vector, plane_vector)

    return plane_vector / jnp.linalg.norm(plane_vector)
