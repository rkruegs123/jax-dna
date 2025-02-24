import functools
import pdb

from jax import jit, vmap
import jax.numpy as jnp

from jax_dna.common import utils


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


@jit
def get_site_positions(body):
    Q = body.orientation
    back_base_vectors = utils.Q_to_back_base(Q) # space frame, normalized
    base_normals = utils.Q_to_base_normal(Q) # space frame, normalized
    cross_prods = utils.Q_to_cross_prod(Q) # space frame, normalized

    # RNA values
    com_to_backbone_x = -0.4
    com_to_backbone_y = 0.2
    com_to_stacking = 0.34
    com_to_hb = 0.4
    back_sites = body.center + com_to_backbone_x*back_base_vectors + com_to_backbone_y*base_normals
    stack_sites = body.center + com_to_stacking * back_base_vectors
    base_sites = body.center + com_to_hb * back_base_vectors

    return back_sites, stack_sites, base_sites


def get_plane_vector(stack_sites, n, first_base, last_base):

    def get_back_poses(i):
        a_idx = i
        ac_idx = i+1
        b_idx = n-i-1
        bc_idx = n-i-1-1

        pose1 = stack_sites[a_idx] - stack_sites[ac_idx]
        pose2 = stack_sites[b_idx] - stack_sites[bc_idx]
        return jnp.array([pose1, pose2])
    back_poses = vmap(get_back_poses)(jnp.arange(first_base, last_base)).reshape(-1, 3)

    plane_vector = fit_plane(back_poses)

    i = first_base
    a_idx = i
    b_idx = n-i-1
    mid_a = (stack_sites[a_idx] + stack_sites[b_idx]) / 2.0

    i = last_base
    ac_idx = i
    bc_idx = n-i-1
    mid_ac = (stack_sites[ac_idx] + stack_sites[bc_idx]) / 2.0

    guess = (-mid_a + mid_ac)
    guess = guess / jnp.linalg.norm(guess)

    flip_cond = (jnp.dot(guess, plane_vector) < 0)
    plane_vector = jnp.where(flip_cond, -1*plane_vector, plane_vector)

    return plane_vector

def get_pos_abs(stack_sites, n, first_base, last_base):

    def get_ith_positions(i):
        a_idx = i
        ac_idx = i+1
        b_idx = n-i-1
        bc_idx = n-i-1-1

        pos_a = stack_sites[a_idx]
        pos_b = stack_sites[b_idx]

        return pos_a, pos_b
    pos_as, pos_bs = vmap(get_ith_positions)(jnp.arange(first_base, last_base))
    pos_as = pos_as.reshape(-1, 3)
    pos_bs = pos_bs.reshape(-1, 3)

    i = last_base-1
    ac_idx = i+1
    bc_idx = n-i-1-1

    last_pos_a = stack_sites[ac_idx]
    pos_as = jnp.concatenate([pos_as, last_pos_a.reshape(-1, 3)])

    last_pos_b = stack_sites[bc_idx]
    pos_bs = jnp.concatenate([pos_bs, last_pos_b.reshape(-1, 3)])

    return pos_as, pos_bs
