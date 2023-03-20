import pdb
from functools import partial
from typing import Callable

from jax import vmap
from jax import random

import jax.numpy as jnp
from jax_md.partition import NeighborList, NeighborListFormat
from jax_md import smap, partition, space, energy, quantity
from jax_md.util import *


i32 = jnp.int32


def dense_to_sparse(idx):
    N = idx.shape[0]
    sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape) # (N, N_neigh)
    sender_idx = jnp.reshape(sender_idx, (-1,))
    receiver_idx = jnp.reshape(idx, (-1,))
    return jnp.stack((sender_idx, receiver_idx), axis=0)

def sparse_to_dense(N, max_count, idx):
    senders, receivers = idx

    offset = jnp.tile(jnp.arange(max_count), N)[:len(senders)]
    hashes = senders * max_count + offset
    dense_idx = N * jnp.ones(((N + 1) * max_count,), i32)
    dense_idx = dense_idx.at[hashes].set(receivers).reshape((N + 1, max_count))
    return dense_idx[:-1]

def sparse_mask_to_dense_mask(sparse_mask_fn):
    def dense_mask_fn(idx, **kwargs): # idx shape (N, N_neigh)
        N, max_count = idx.shape
        sparse_idx = dense_to_sparse(idx)
        pdb.set_trace()
        sparse_masked_idx = sparse_mask_fn(sparse_idx, **kwargs)
        return sparse_to_dense(N, max_count, sparse_masked_idx)
    return dense_mask_fn


# FIXME: could clean up existing, could test with OrderedSparse, then could do the more streaightofraward dense mask function. Need to also confirm that we're doing it right -- e.g. talk about occupancy, and the pruning -- because at face value, :max_occupancy looks like it'd be doing the wrong thing... shouldn't be a problem as we follow how self-masking is done. Also, put in actual dynamic_nbrs
def get_rkk_sparse_custom(pairs, mask_val=5):
    check_pairs = partial(vmap(jnp.array_equal, in_axes=(0, None)), pairs)
    def mask_single(nbr_pr):
        is_prohibited_pair = check_pairs(nbr_pr).any()
        return jnp.where(is_prohibited_pair, jnp.array([nbr_pr[0], mask_val], dtype=jnp.int32), nbr_pr)


    check_sparse = vmap(mask_single, in_axes=1)
    check_sparse_trans = lambda sparse_idx: check_sparse(sparse_idx).T # FIXME: better way to do this?
    return check_sparse_trans


# pairs must be a list of tuples
# e.g. pairs = [(0, 1), (1, 2)]
def get_rkk_dense_custom(pairs, mask_val):
    pairs = tuple(jnp.array(pairs).T) # will be a tuple of jnp arrays
    def custom_mask_dense(dense_idx):
        return dense_idx.at[pairs].set(mask_val)
    return custom_mask_dense



# For testing
def test_custom_mask_function():
    displacement_fn, shift_fn = space.free()

    box_size = 1.0
    r_cutoff = 3.0
    dr_threshold = 0.0
    n_particles = 5
    R = jnp.broadcast_to(jnp.zeros(3), (n_particles,3))

    mask_val = n_particles


    to_mask = jnp.array([(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)], dtype=jnp.int32)
    # custom_mask_function = sparse_mask_to_dense_mask(get_rkk_sparse_custom(to_mask, mask_val=mask_val))
    custom_mask_function = get_rkk_dense_custom(to_mask, mask_val=mask_val)

    neighbor_list_fn = partition.neighbor_list(
      displacement_fn,
      box=box_size,
      r_cutoff=r_cutoff,
      dr_threshold=dr_threshold,
      custom_mask_function=custom_mask_function,
      # format=NeighborListFormat.Sparse
      format=NeighborListFormat.Sparse
    )

    neighbors = neighbor_list_fn.allocate(R)
    neighbors = neighbors.update(R)

    pdb.set_trace()

    return


if __name__ == "__main__":

    test_custom_mask_function()

    pdb.set_trace()

    print("done")
