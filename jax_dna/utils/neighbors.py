"""Utilities for using neighbor lists."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax_md.partition import NeighborListFormat, neighbor_list


ERR_NEIGHBORS_INVALID_BONDED_NEIGHBORS = "Indices of bonded neighbors must be bewteen 0 and n_nucleotides-1"

def get_neighbor_list_fn(
    bonded_neighbors: np.ndarray,
    n_nucleotides: int,
    displacement_fn: Callable,
    box_size: float,
    r_cutoff: float = 10.0,
    dr_threshold: float = 0.2,
) -> Callable:
    """Construct a neighbor list function for unbonded pairs.

    This function constructs a neighbor list function for identifying neighboring
    particles that are not bonded. To achieve this, it uses the provided list of
    bonded neighbors to explicitly exclude bonded pairs from the neighbor list
    by creating a mask.
    """
    bonded_neighbors_in_range = (bonded_neighbors >= 0) & (bonded_neighbors < n_nucleotides)
    if not bonded_neighbors_in_range.all():
        raise ValueError(ERR_NEIGHBORS_INVALID_BONDED_NEIGHBORS)

    # Construct nx2 mask specifying bonded pairs
    dense_mask = np.full((n_nucleotides, 2), n_nucleotides, dtype=np.int32)
    counter = np.zeros(n_nucleotides, dtype=np.int32)
    for bp1, bp2 in bonded_neighbors:
        dense_mask[bp1, counter[bp1]] = bp2
        counter[bp1] += 1

        dense_mask[bp2, counter[bp2]] = bp1
        counter[bp2] += 1
    dense_mask = jnp.array(dense_mask, dtype=jnp.int32)

    # Specify a mask function for the neighbor list construction
    def bonded_nbrs_mask_fn(dense_idx: np.ndarray) -> np.ndarray:
        nbr_mask1 = dense_idx == dense_mask[:, 0].reshape(n_nucleotides, 1)
        dense_idx = jnp.where(nbr_mask1, n_nucleotides, dense_idx)

        nbr_mask2 = dense_idx == dense_mask[:, 1].reshape(n_nucleotides, 1)
        return jnp.where(nbr_mask2, n_nucleotides, dense_idx)

    # Construct a neighbor list function via JAX-MD
    return neighbor_list(
        displacement_fn,
        box=box_size,
        r_cutoff=r_cutoff,
        dr_threshold=dr_threshold,
        custom_mask_function=bonded_nbrs_mask_fn,
        format=NeighborListFormat.OrderedSparse,
        disable_cell_list=True,
    )
