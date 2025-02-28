from pathlib import Path

import jax
import jax.numpy as jnp
import jax_md

from jax_dna.input import tree

TEST_FILES_DIR = Path(__file__).parent / "test_files"


def test_save_load_pytree():
    pytree = [
        {"test": [1, 2, 3]},
        {"val2": [{"test": [4, 5, 6]}]},
        {"val3": [7, 8, 9]},
        {
            "vals": jax_md.rigid_body.RigidBody(
                center=jnp.array([1, 2, 3], dtype=jnp.float32), orientation=jnp.array([4, 5, 6, 8], dtype=jnp.float32)
            )
        },
    ]

    save_path = TEST_FILES_DIR / "test_save_load_pytree.pkl"
    tree.save_pytree(pytree, save_path)

    loaded_pytree = tree.load_pytree(save_path)

    save_path.unlink()

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda x, y: jnp.allclose(x, y) if isinstance(x, jnp.ndarray) else x == y, pytree, loaded_pytree
        )
    )
