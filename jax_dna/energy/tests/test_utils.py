"""Test the jax_dna.energy.utils module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_dna.energy.utils as je_utils
import jax_dna.utils.types as typ

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 - ignore boolean positional value


@pytest.mark.parametrize(
    ("pseq", "nt1", "nt2", "weights_table", "is_unpaired", "idx_to_unpaired_idx", "idx_to_bp_idx", "expected_weight"),
    [
        (
            (
                jnp.array(
                    [
                        [0.27, 0.03, 0.68, 0.02],
                        [0.04, 0.56, 0.22, 0.18],
                    ]
                ),
                jnp.array(
                    [
                        [0.66, 0.14, 0.01, 0.19],
                    ]
                ),
            ),
            0,
            3,
            jnp.array([[0.2, 0.1, 0.3, 0.4], [0.05, 0.25, 0.1, 0.6], [0.55, 0.15, 0.2, 0.1], [0.1, 0.15, 0.6, 0.15]]),
            jnp.array([0, 1, 1, 0]),  # is unpaired
            jnp.array([-1, 0, 1, -1]),  # idx_to_unpaired_idx
            jnp.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),  # idx_to_bp_idx
            0.66 * 0.4 + 0.14 * 0.1 + 0.01 * 0.15 + 0.19 * 0.1,  # expected_weight
        ),
        (
            (
                jnp.array(
                    [
                        [0.27, 0.03, 0.68, 0.02],
                        [0.04, 0.56, 0.22, 0.18],
                    ]
                ),
                jnp.array(
                    [
                        [0.66, 0.14, 0.01, 0.19],
                    ]
                ),
            ),
            0,
            1,
            jnp.array([[0.2, 0.1, 0.3, 0.4], [0.05, 0.25, 0.1, 0.6], [0.55, 0.15, 0.2, 0.1], [0.1, 0.15, 0.6, 0.15]]),
            jnp.array([0, 1, 1, 0]),  # is unpaired
            jnp.array([-1, 0, 1, -1]),  # idx_to_unpaired_idx
            jnp.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),  # idx_to_bp_idx
            0.66 * 0.27 * 0.2
            + 0.66 * 0.03 * 0.1
            + 0.66 * 0.68 * 0.3
            + 0.66 * 0.02 * 0.4
            + 0.14 * 0.27 * 0.1
            + 0.14 * 0.03 * 0.15
            + 0.14 * 0.68 * 0.6
            + 0.14 * 0.02 * 0.15
            + 0.01 * 0.27 * 0.55
            + 0.01 * 0.03 * 0.15
            + 0.01 * 0.68 * 0.2
            + 0.01 * 0.02 * 0.1
            + 0.19 * 0.27 * 0.05
            + 0.19 * 0.03 * 0.25
            + 0.19 * 0.68 * 0.1
            + 0.19 * 0.02 * 0.6,  # expected_weight
        ),
        (
            (
                jnp.array(
                    [
                        [0.27, 0.03, 0.68, 0.02],
                        [0.04, 0.56, 0.22, 0.18],
                    ]
                ),
                jnp.array(
                    [
                        [0.66, 0.14, 0.01, 0.19],
                    ]
                ),
            ),
            1,
            2,
            jnp.array([[0.2, 0.1, 0.3, 0.4], [0.05, 0.25, 0.1, 0.6], [0.55, 0.15, 0.2, 0.1], [0.1, 0.15, 0.6, 0.15]]),
            jnp.array([0, 1, 1, 0]),  # is unpaired
            jnp.array([-1, 0, 1, -1]),  # idx_to_unpaired_idx
            jnp.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),  # idx_to_bp_idx
            0.27 * 0.04 * 0.2
            + 0.27 * 0.56 * 0.1
            + 0.27 * 0.22 * 0.3
            + 0.27 * 0.18 * 0.4
            + 0.03 * 0.04 * 0.05
            + 0.03 * 0.56 * 0.25
            + 0.03 * 0.22 * 0.1
            + 0.03 * 0.18 * 0.6
            + 0.68 * 0.04 * 0.55
            + 0.68 * 0.56 * 0.15
            + 0.68 * 0.22 * 0.2
            + 0.68 * 0.18 * 0.1
            + 0.02 * 0.04 * 0.1
            + 0.02 * 0.56 * 0.15
            + 0.02 * 0.22 * 0.6
            + 0.02 * 0.18 * 0.15,
        ),
        (
            (
                jnp.array(
                    [
                        [0.27, 0.03, 0.68, 0.02],
                        [0.04, 0.56, 0.22, 0.18],
                    ]
                ),
                jnp.array(
                    [
                        [0.66, 0.14, 0.01, 0.19],
                    ]
                ),
            ),
            2,
            3,
            jnp.array([[0.2, 0.1, 0.3, 0.4], [0.05, 0.25, 0.1, 0.6], [0.55, 0.15, 0.2, 0.1], [0.1, 0.15, 0.6, 0.15]]),
            jnp.array([0, 1, 1, 0]),  # is unpaired
            jnp.array([-1, 0, 1, -1]),  # idx_to_unpaired_idx
            jnp.array([[0, 0], [-1, -1], [-1, -1], [0, 1]]),  # idx_to_bp_idx
            0.04 * 0.14 * 0.2
            + 0.04 * 0.01 * 0.1
            + 0.04 * 0.19 * 0.3
            + 0.04 * 0.66 * 0.4
            + 0.56 * 0.14 * 0.05
            + 0.56 * 0.01 * 0.25
            + 0.56 * 0.19 * 0.1
            + 0.56 * 0.66 * 0.6
            + 0.22 * 0.14 * 0.55
            + 0.22 * 0.01 * 0.15
            + 0.22 * 0.19 * 0.2
            + 0.22 * 0.66 * 0.1
            + 0.18 * 0.14 * 0.1
            + 0.18 * 0.01 * 0.15
            + 0.18 * 0.19 * 0.6
            + 0.18 * 0.66 * 0.15,
        ),
    ],
)
def test_compute_seq_dep_weight(
    pseq: typ.Probabilistic_Sequence,
    nt1: int,
    nt2: int,
    weights_table: np.ndarray,
    is_unpaired: typ.Arr_Nucleotide_Int,
    idx_to_unpaired_idx: typ.Arr_Nucleotide_Int,
    idx_to_bp_idx: typ.Arr_Nucleotide_2_Int,
    expected_weight: float,
):
    calc_weight = je_utils.compute_seq_dep_weight(
        pseq, nt1, nt2, weights_table, is_unpaired, idx_to_unpaired_idx, idx_to_bp_idx
    )
    assert np.isclose(calc_weight, expected_weight)
