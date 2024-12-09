"""Pitch observable."""

import dataclasses as dc
import functools
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax_md import space

import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.trajectory as jd_traj
import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units

TARGETS = {
    "oxDNA": 23.0, # Angstroms. Experimental value for helical radius is 11.5-12 A
}


@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def single_diameter(
        bp: jnp.ndarray,
        back_sites: jnp.ndarray,
        displacement_fn: Callable,
        sigma_backbone: float
) -> jd_types.ARR_OR_SCALAR:
    """Computes the helical diameter of a base pair."""

    bp1, bp2 = bp

    # Compute the distance between the backbone sites
    dr = displacement_fn(back_sites[bp1], back_sites[bp2])
    r = space.distance(dr)

    # Add the excluded volume distance
    r += sigma_backbone

    return r * jd_units.ANGSTROMS_PER_OXDNA_LENGTH


@chex.dataclass(frozen=True, kw_only=True)
class Diameter(jd_obs.BaseObservable):
    """Computes the helical diameter for each state.

    The helical diameter for a given base pair is defined as the furthest extent of the
    excluded volume. The diameter per state is the average over all (specified) base pairs.

    Args:
    - bp: a 2-dimensional array containing the indices of the h-bonded nucleotides
    - displacement_fn: a function for computing displacements between two positions
    """

    h_bonded_base_pairs: jnp.ndarray = dc.field(
        hash=False
    )
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

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

        diameters = jax.vmap(single_diameter, (None, 0, None, None))(
            self.h_bonded_base_pairs, back_sites, self.displacement_fn, sigma_backbone
        )
        return jnp.mean(diameters, axis=1)


if __name__ == "__main__":
    import jax_md

    import jax_dna.input.topology as jd_top

    test_geometry = jd_toml.parse_toml("jax_dna/input/dna1/default_energy.toml")["geometry"]
    tranform_fn = functools.partial(
        jd_energy.Nucleotide.from_rigid_body,
        com_to_backbone=test_geometry["com_to_backbone"],
        com_to_hb=test_geometry["com_to_hb"],
        com_to_stacking=test_geometry["com_to_stacking"],
    )

    top = jd_top.from_oxdna_file("data/templates/simple-helix/sys.top")
    test_traj = jd_traj.from_file(
        path="data/templates/simple-helix/init.conf",
        strand_lengths=top.strand_counts,
    )

    sim_traj = jd_sio.SimulatorTrajectory(
        seq_oh=jnp.array(top.seq_one_hot),
        strand_lengths=top.strand_counts,
        rigid_body=test_traj.state_rigid_body,
    )

    bps = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    displacement_fn, _ = space.free()
    diameter = Diameter(rigid_body_transform_fn=tranform_fn, h_bonded_base_pairs=bps, displacement_fn=displacement_fn)
    sigma_backbone = 0.7
    output_diameters = diameter(sim_traj, sigma_backbone)
