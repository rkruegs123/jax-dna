import jax
import jax.numpy as jnp
import jaxtyping as jaxtyp
import pytest

import jax_dna.utils.helpers as jdh


def pytree_equal(tree1, tree2):
    """Check if two PyTrees have the same structure and values."""
    # Check if the structures match
    if jax.tree.structure(tree1) != jax.tree.structure(tree2):
        return False

    # Check if the values match
    def values_equal(x, y):
        return jnp.array_equal(x, y)

    return all(jax.tree.flatten(jax.tree.map(values_equal, tree1, tree2))[0])


@pytest.mark.parametrize(
    ("in_iter", "n", "out_iter"),
    [
        (
            "ABCDEFG",
            3,
            [("A", "B", "C"), ("D", "E", "F"), ("G",)],
        ),
    ],
)
def test_batched(in_iter, n, out_iter):
    assert list(jdh.batched(in_iter, n)) == out_iter


def test_batched_raises_value_error():
    with pytest.raises(ValueError, match=jdh.ERR_BATCHED_N):
        list(jdh.batched("ABCDEFG", 0))


@pytest.mark.parametrize(
    ("trees", "expected_pytree"),
    [
        (
            [
                {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])},
                {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}
            ],
            {"a": jnp.array([[1, 2], [5, 6]]), "b": jnp.array([[3, 4], [7, 8]])},
        ),
    ]
)
def test_tree_stack(trees: list[jaxtyp.PyTree], expected_pytree: jaxtyp.PyTree):
    stacked_pytree = jdh.tree_stack(trees)
    assert pytree_equal(stacked_pytree, expected_pytree)


@pytest.mark.parametrize(
    ("trees", "expected_pytree"),
    [
        (
            [
                {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])},
                {"a": jnp.array([5, 6]), "b": jnp.array([7, 8])}
            ],
            {"a": jnp.array([1, 2, 5, 6]), "b": jnp.array([3, 4, 7, 8])},
        ),
    ]
)
def test_tree_concatenate(trees: list[jaxtyp.PyTree], expected_pytree: jaxtyp.PyTree):
    concatenated_pytree = jdh.tree_concatenate(trees)
    assert pytree_equal(concatenated_pytree, expected_pytree)
