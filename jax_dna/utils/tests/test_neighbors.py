import jax.numpy as jnp
import pytest
from jax_md import space

import jax_dna.utils.neighbors as jd_nb


def test_valid_bonded_nbrs():
    bonded_nbrs = jnp.array([[0, 1], [2, 3]])
    n_nucleotides = 3
    box_size = 10.0
    displacement_fn = space.periodic(box_size)

    with pytest.raises(ValueError, match=jd_nb.ERR_NEIGHBORS_INVALID_BONDED_NEIGHBORS):
        jd_nb.get_neighbor_list_fn(bonded_nbrs, n_nucleotides, displacement_fn, box_size)


def test_no_bonded_nbrs_returned():
    bonded_nbrs = jnp.array(
        [
            [0, 1],
            [1, 2],
        ]
    )
    n_nucleotides = 3
    box_size = 10.0
    displacement_fn, _ = space.periodic(box_size)

    neighbor_fn = jd_nb.get_neighbor_list_fn(bonded_nbrs, n_nucleotides, displacement_fn, box_size)

    body_center = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    neighbors = neighbor_fn.allocate(body_center)
    neighbors = neighbors.update(body_center)
    neighbors_idx = neighbors.idx

    neighbors_set = {(int(i), int(j)) for i, j in neighbors_idx.T}

    for i, j in bonded_nbrs:
        bonded_nbr_tuple = (int(i), int(j))
        assert bonded_nbr_tuple not in neighbors_set
