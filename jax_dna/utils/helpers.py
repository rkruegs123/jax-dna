"""Helper functions for the JAX-DNA package."""

import itertools
import sys
from collections.abc import Iterable
from typing import Any

ERR_BATCHED_N = "n must be at least one"


def batched(iterable: Iterable[Any], n: int) -> Iterable[Any]:
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
        def batch_f(iterable: Iterable[Any], n: int) -> Iterable[Any]:
            # batched('ABCDEFG', 3) → ABC DEF G
            if n < 1:
                raise ValueError(ERR_BATCHED_N)
            it = iter(iterable)
            while batch := tuple(itertools.islice(it, n)):
                yield batch

    return batch_f(iterable, n)
