"""Utilities for computing gradients."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pdb

from typing import Callable


def _first_arg_partial(f, *args, **kwargs):
    def f_(x):
        return f(x, *args, **kwargs)
    return f_

def _split_and_pack_like(j, x):
    leaves, structure = jax.tree_util.tree_flatten(x)
    sizes = [leaf.size for leaf in leaves]
    split = jnp.split(j, np.cumsum(sizes), axis=-1)
    reshaped = [s.reshape(s.shape[:-1] + y.shape) for s, y in zip(split, leaves)]
    return jax.tree_util.tree_unflatten(structure, reshaped)

def _tangents_like(x):
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(x)]
    eye = np.eye(sum([leaf.size for leaf in jax.tree_util.tree_leaves(x)]))
    return _split_and_pack_like(eye, x)

def value_and_jacfwd(f: Callable) -> Callable:
    """Returns a function that computes the Jacobian for the first argument, along with the value of the function."""
    def val_and_jac(*args, **kwargs):
        partial_f = _first_arg_partial(f, *args[1:], **kwargs)
        tangents = _tangents_like(args[0])
        jvp = functools.partial(jax.jvp, partial_f, (args[0],))
        y, jac = jax.vmap(jvp, out_axes=-1)((tangents,))
        y = jax.tree_map(lambda x: x[..., 0], y)
        jac = jax.tree_map(lambda j: _split_and_pack_like(j, args[0]), jac)
        return y, jac
    return val_and_jac
