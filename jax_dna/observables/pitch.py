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

TARGETS = {
    "oxDNA": 10.5, # bp/turn
}


def compute_pitch(avg_pitch_angle: float) -> float:
    """Computes the pitch given an average pitch angle in radians.

    Args:
    - avg_pitch_angle: a value in radians specifying the pitch value averaged over a trajectory
    """
    return jnp.pi / avg_pitch_angle

@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def single_pitch_angle(
        quartet: jnp.ndarray,
        base_sites: jnp.ndarray,
        back_sites: jnp.ndarray,
        displacement_fn: Callable
) -> jd_types.ARR_OR_SCALAR:
    """Computes the pitch angle between adjacent base pairs."""

    # Extract the base pairs
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2

    # Compute the local helical axis
    local_helix_dir = jd_obs.local_helical_axis(quartet, base_sites, displacement_fn)

    # Compute the vector between backbone sites for each base pair
    bb1 = displacement_fn(back_sites[b1], back_sites[a1])
    bb1_dir = bb1 / jnp.linalg.norm(bb1)
    bb2 = displacement_fn(back_sites[b2], back_sites[a2])
    bb2_dir = bb2 / jnp.linalg.norm(bb2)

    # Project each vector onto the local helical axis
    bb1_proj = displacement_fn(bb1, jnp.dot(local_helix_dir, bb1) * local_helix_dir)
    bb1_proj_dir = bb1_proj / jnp.linalg.norm(bb1_proj)
    bb2_proj = displacement_fn(bb2, jnp.dot(local_helix_dir, bb2) * local_helix_dir)
    bb2_proj_dir = bb2_proj / jnp.linalg.norm(bb2_proj)

    # Compute the angle between these projections
    theta = jnp.arccos(jd_math.clamp(jnp.dot(bb1_proj_dir, bb2_proj_dir)))
    return theta


@chex.dataclass(frozen=True, kw_only=True)
class PitchAngle(jd_obs.BaseObservable):
    """Computes the average pitch angle in radians for each state.

    The pitch is defined by (2*pi) / <angle> where <angle> is the average angle
    between adjacent base pairs across states

    Args:
    - quartets: a (n_quartets, 2, 2) array containing the pairs of adjacent base pairs
      for which to compute pitch angles
    - displacement_fn: a function for computing displacements between two positions
    """

    quartets: jnp.ndarray = dc.field(
        hash=False
    )
    displacement_fn: Callable

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the average pitch angle in radians.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the pitch for

        Returns:
            jd_types.ARR_OR_SCALAR: the average pitch angle in radians for each state,
            so expect a size of (n_states,)
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        base_sites = nucleotides.base_sites
        back_sites = nucleotides.back_sites

        angles = jax.vmap(single_pitch_angle, (None, 0, 0, None))(
            self.quartets, base_sites, back_sites, self.displacement_fn
        )
        return jnp.mean(angles, axis=1)


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

    quartets = jd_obs.get_duplex_quartets(8)
    displacement_fn, _ = space.free()
    pitch_angle = PitchAngle(rigid_body_transform_fn=tranform_fn, quartets=quartets, displacement_fn=displacement_fn)
    output_pitch_angles = pitch_angle(sim_traj)
