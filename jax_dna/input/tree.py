"""Functions for saving and loading pytrees."""

import pickle
from pathlib import Path

import jax

import jax_dna.utils.types as jdna_types


def save_pytree(data: jdna_types.PyTree, filename: jdna_types.PathOrStr) -> None:
    """Save a pytree to a file."""
    save_path = Path(filename)
    leaves, treedef = jax.tree_util.tree_flatten(data)
    with save_path.open("wb") as f:
        pickle.dump((leaves, treedef), f)


def load_pytree(filename: jdna_types.PathOrStr) -> jdna_types.PyTree:
    """Load a pytree to a file."""
    save_path = Path(filename)
    with save_path.open("rb") as f:
        # Though this is labeled as a security issue by Bandit we only open
        # files that we write. So we can ignore this for now, but if there
        # another way we should consider switching to that.
        # TODO(ryanhausen): Investigate a more secure way to load the file.
        # https://github.com/rkruegs123/jax-dna/issues/7
        leaves, treedef = pickle.load(f)  # nosec B301 # noqa: S301
    return jax.tree_util.tree_unflatten(treedef, leaves)
