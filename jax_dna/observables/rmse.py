"""RMSE observable."""

import dataclasses as dc
import functools
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from jax_md import space, rigid_body

import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.trajectory as jd_traj
import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types
import jax_dna.utils.units as jd_units


def svd_align(ref_coords: jnp.ndarray, coords: jnp.ndarray):
    """Aligns a set of 3D coordinates to a reference configuration via SVD."""

    n_nt = coords.shape[1]
    indexes = jnp.arange(n_nt) # Note: this could be a subset of the structure if desired

    # Set the origin of the reference configuration
    ref_center = jnp.zeros(3) # Reference structure is assumed to be centered at the origin

    # Calculate centroids of the reference and input structures
    av1 = ref_center
    av2 = jnp.mean(coords[0][indexes], axis=0)
    coords = coords.at[0].set(coords[0] - av2) # Shift the first input structure to be centered

    # Compute the correlation matrix between the reference and input coordinates
    a = jnp.dot(jnp.transpose(coords[0][indexes]), ref_coords - av1)

    # Perform Singular Value Decomposition (SVD) to obtain rotation components
    u, _, vt = jnp.linalg.svd(a)

    # Calculate the rotation matrix
    rot = jnp.transpose(jnp.dot(jnp.transpose(vt), jnp.transpose(u)))

    # Check for a reflection
    found_reflection = (jnp.linalg.det(rot) < 0)
    vt = jnp.where(found_reflection, vt.at[2].set(-vt[2]), vt)
    rot = jnp.where(found_reflection,
                    jnp.transpose(jnp.dot(jnp.transpose(vt), jnp.transpose(u))),
                    rot)

    # Translation is trivial here as `tran` is effectively the center of reference (set to `av1`)
    tran = av1

    # Apply the computed rotation to the coordinates, back-base vectors, and base normals
    return (jnp.dot(coords[0], rot) + tran,
            jnp.dot(coords[1], rot),
            jnp.dot(coords[2], rot))



def single_rmse(
        target: rigid_body.RigidBody,
        state_nts: jd_energy.nucleotide.Nucleotide,
) -> jd_types.ARR_OR_SCALAR:
    """Computes the RMSE between a state and a target configuration."""

    conf = jnp.asarray([state_nts.center, state_nts.back_base_vectors, state_nts.base_normals])
    aligned_conf = svd_align(target.center, conf)[0]
    fluc_sq = jnp.power(jnp.linalg.norm(aligned_conf - target.center, axis=1), 2)

    rmse = jnp.sqrt(jnp.mean(fluc_sq))

    return rmse * jd_units.ANGSTROMS_PER_OXDNA_LENGTH

ERR_SINGLE_TARGET_STATE_REQUIRED = "the target state must be a single conformation"
ERR_TARGET_STATE_DIM = "the target state must have center positions in (x, y, z) format"

@chex.dataclass(frozen=True, kw_only=True)
class RMSE(jd_obs.BaseObservable):
    """Computes the RMSE with respect to a target configuration for each state.

    Args:
    - target_state: a rigid body specifying the target configuration
    """

    target_state: rigid_body.RigidBody

    def __post_init__(self) -> None:
        """Validate the input."""
        if self.rigid_body_transform_fn is None:
            raise ValueError(jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

        if len(target_state.center.shape) != 2:
            raise ValueError(ERR_SINGLE_TARGET_STATE_REQUIRED)

        if target_state.center.shape[1] != 3:
            raise ValueError(ERR_TARGET_STATE_DIM)


    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the RMSE in Angstroms.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the RMSE for

        Returns:
            jd_types.ARR_OR_SCALAR: the RMSE in Angstroms for each state, so expect a
            size of (n_states,)
        """

        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        # Center the target state
        centered_target_state = self.target_state.set(
            center=self.target_state.center - jnp.mean(self.target_state.center, axis=0))

        # Compute the RMSE per state
        rmses = jax.vmap(single_rmse, (None, 0))(centered_target_state, nucleotides)
        return rmses


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

    target_state = rigid_body.RigidBody(
        center=test_traj.state_rigid_body.center[0],
        orientation=rigid_body.Quaternion(test_traj.state_rigid_body.orientation.vec[0])
    )
    rmse = RMSE(rigid_body_transform_fn=tranform_fn, target_state=target_state)
    output_rmses = rmse(sim_traj)
