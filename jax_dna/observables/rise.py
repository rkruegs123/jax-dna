"""Rise observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units

TARGETS = {
    "oxDNA": 3.4,  # Angstroms
}


def single_rise(quartet: jnp.ndarray, base_sites: jnp.ndarray, displacement_fn: Callable) -> jd_types.ARR_OR_SCALAR:
    """Computes the rise between adjacent base pairs."""
    # Extract the base pairs
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2

    # Compute the local helical axis
    local_helix_dir = jd_obs.local_helical_axis(quartet, base_sites, displacement_fn)

    # Compute the midpoints of each base pair
    midp1 = (base_sites[a1] + base_sites[b1]) / 2.0
    midp2 = (base_sites[a2] + base_sites[b2]) / 2.0

    # Project the displacement between the midpoints onto the local helical axis
    dr = displacement_fn(midp2, midp1)

    rise = jnp.dot(dr, local_helix_dir)

    return rise * jd_units.ANGSTROMS_PER_OXDNA_LENGTH


single_rise_mapped = jax.vmap(single_rise, (0, None, None))


@chex.dataclass(frozen=True, kw_only=True)
class Rise(jd_obs.BaseObservable):
    """Computes the rise for each state.

    The rise between two adjacent base pairs is defined as the distance of the displacement
    vector between their midpoints projected onto the local helical axis. The rise per state
    is the average rise over all (specified) pairs of adjacent base pairs (i.e. quartets)

    Args:
    - quartets: a (n_bp, 2, 2) array containing the pairs of adjacent base pairs
      for which to compute the rise
    - displacement_fn: a function for computing displacements between two positions
    """

    quartets: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the average rise in Angstroms.

        Args:
            trajectory (jd_sio.Trajectory): the trajectory to calculate the rise for

        Returns:
            jd_types.ARR_OR_SCALAR: the average rise in Angstroms for each state, so expect a
            size of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)
        base_sites = nucleotides.base_sites

        rises = jax.vmap(single_rise_mapped, (None, 0, None))(self.quartets, base_sites, self.displacement_fn)
        return jnp.mean(rises, axis=1)
