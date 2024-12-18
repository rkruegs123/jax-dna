"""Base class for observables."""

from collections.abc import Callable

import chex
import jax.numpy as jnp

import jax_dna.simulators.io as jd_sio

ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED = "rigid_body_transform_fn must be provided"

@chex.dataclass(frozen=True)
class BaseObservable:
    """Base class for observables."""

    rigid_body_transform_fn: Callable

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jnp.ndarray:
        """Calculate the observable."""


def local_helical_axis(
        quartet: jnp.ndarray,
        base_sites: jnp.ndarray,
        displacement_fn: Callable
) -> jnp.ndarray:
    """Computes the normalized local helical axis defined by two base pairs"""

    # Extract the two base pairs. a1 is h-bonded to b1, a2 is h-bonded to b2
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2

    # Compute the midpoints of each base pair
    midp_a1b1 = (base_sites[a1] + base_sites[b1]) / 2.0
    midp_a2b2 = (base_sites[a2] + base_sites[b2]) / 2.0

    # Compute the normalized direction between the midpoints
    dr = displacement_fn(midp_a2b2, midp_a1b1)
    return dr / jnp.linalg.norm(dr)

def get_duplex_quartets(n_nucs_per_strand: int) -> jnp.ndarray:
    """Computes all quartets (i.e. pairs of adjacent base pairs) for a duplex of a given size"""

    # Construct the indices of the nucleotides on each strand
    s1_nucs = list(range(n_nucs_per_strand))
    s2_nucs = list(range(n_nucs_per_strand, n_nucs_per_strand*2))
    s2_nucs.reverse()

    # Record all pairs of adjacent base pairs
    bps = list(zip(s1_nucs, s2_nucs))
    n_bps = len(s1_nucs)
    all_quartets = list()
    for i in range(n_bps-1):
        bp1 = bps[i]
        bp2 = bps[i+1]
        all_quartets.append([bp1, bp2])
    return jnp.array(all_quartets, dtype=jnp.int32)
