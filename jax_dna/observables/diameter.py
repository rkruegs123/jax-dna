"""Diameter observable."""

import dataclasses as dc
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax_md import space

import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units

TARGETS = {
    "oxDNA": 23.0,  # Angstroms. Experimental value for helical radius is 11.5-12 A
}

ERR_DISPLACEMENT_FN_REQUIRED = "A displacement function is required for computing the helical diameter."


def single_diameter(
    bp: jnp.ndarray, back_sites: jnp.ndarray, displacement_fn: Callable, sigma_backbone: float
) -> jd_types.ARR_OR_SCALAR:
    """Computes the helical diameter of a base pair.

    Args:
        bp (jnp.ndarray): a 2-dimensional array containing the indices of the h-bonded nucleotides
        back_sites (jnp.ndarray): a 2-dimensional array containing the positions of the backbone sites
        displacement_fn (Callable): a function for computing displacements between two positions
        sigma_backbone (float): the excluded volume distance between backbone sites

    Returns:
        jd_types.ARR_OR_SCALAR: the helical diameter in Angstroms for each base pair
    """
    bp1, bp2 = bp

    # Compute the distance between the backbone sites
    dr = displacement_fn(back_sites[bp1], back_sites[bp2])
    r = space.distance(dr)

    # Add the excluded volume distance
    r += sigma_backbone

    return r * jd_units.ANGSTROMS_PER_OXDNA_LENGTH


single_diameter_mapped = jax.vmap(single_diameter, (0, None, None, None))


@chex.dataclass(frozen=True, kw_only=True)
class Diameter(jd_obs.BaseObservable):
    """Computes the helical diameter for each state.

    The helical diameter for a given base pair is defined as the furthest extent of the
    excluded volume. The diameter per state is the average over all (specified) base pairs.

    Args:
    - bp: a 2-dimensional array containing the indices of the h-bonded nucleotides
    - displacement_fn: a function for computing displacements between two positions
    """

    h_bonded_base_pairs: jnp.ndarray = dc.field(hash=False)
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)
        if self.displacement_fn is None:
            raise ValueError(ERR_DISPLACEMENT_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory, sigma_backbone: float) -> jd_types.ARR_OR_SCALAR:
        """Calculate the average helical diameter in Angstroms.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the helical diameter for
            sigma_backbone (float): the excluded volume distance between backbone sites

        Returns:
            jd_types.ARR_OR_SCALAR: the average helical diameter in Angstroms for each state, so
            expect a size of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)
        back_sites = nucleotides.back_sites

        diameters = jax.vmap(single_diameter_mapped, (None, 0, None, None))(
            self.h_bonded_base_pairs, back_sites, self.displacement_fn, sigma_backbone
        )
        return jnp.mean(diameters, axis=1)

