"""Helper functions for the JAX-DNA package."""

import itertools
import sys
from typing import Any


def batched(iterable: iter[Any], n: int) -> iter[Any]:
    """Batch an iterable into chunks of size n.

    Args:
        iterable (iter[Any]): iterable to batch
        n (int): batch size

    Returns:
        iter[Any]: batched iterable
    """
    if sys.version_info >= (3, 12):
        batch_f = itertools.batched
    else:
        # taken from https://docs.python.org/3/library/itertools.html#itertools.batched
        def batch_f(iterable: iter[Any], n: int) -> iter[Any]:
            # batched('ABCDEFG', 3) → ABC DEF G
            if n < 1:
                raise ValueError("n must be at least one")
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, n)):
                yield batch

    return batch_f(iterable, n)
