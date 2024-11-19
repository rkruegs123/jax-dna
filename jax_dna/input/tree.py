import pickle
from pathlib import Path

import jax

import jax_dna.utils.types as jdna_types


def save_pytree(data: jdna_types.PyTree, filename: jdna_types.PathOrStr):
    """Save a pytree to a file."""
    save_path = Path(filename)
    leaves, treedef = jax.tree_util.tree_flatten(data)
    with save_path.open("wb") as f:
        pickle.dump((leaves, treedef), f)


def load_pytree(filename: jdna_types.PathOrStr) -> jdna_types.PyTree:
    """Load a pytree to a file."""
    save_path = Path(filename)
    with save_path.open("rb") as f:
        leaves, treedef = pickle.load(f)
    return jax.tree_util.tree_unflatten(treedef, leaves)
