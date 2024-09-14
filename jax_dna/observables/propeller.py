import dataclasses as dc
import functools

import chex
import jax
import jax.numpy as jnp

import jax_dna.energy.dna1 as jd_energy
import jax_dna.input.toml as jd_toml
import jax_dna.input.trajectory as jd_traj
import jax_dna.observables.base as jd_obs
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types

ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED = "rigid_body_transform_fn must be provided"

TARGETS = {
    "oxDNA": 21.7,
}


# TODO: this could probably be done using broadcasting rather than vmap
@functools.partial(jax.vmap, in_axes=(0, None))
def single_propeller_twist_rad(
    bp: jnp.ndarray,  # this is a array of shape (2,) containing the indices of the h-bonded nucleotides
    base_normals: jnp.ndarray,
) -> jnp.ndarray:
    """Computes the propeller twist of a base pair."""

    # get the normal vectors of the h-bonded bases
    bp1, bp2 = bp
    nv1 = base_normals[bp1]
    nv2 = base_normals[bp2]

    # compute angle between base normal vectors
    return jnp.arccos(jd_math.clamp(jnp.dot(nv1, nv2)))


@chex.dataclass(frozen=True, kw_only=True)
class PropellerTwist(jd_obs.BaseObservable):
    """Computes the propeller twist of a base pair.

    The propeller twist is defined as the angle between the normal vectors
    of h-bonded bases

    Args:
    - bp: a 2-dimensional array containing the indices of the h-bonded nucleotides
    - base_normals: the base normal vectors of the entire body
    """

    h_bonded_base_pairs: jnp.ndarray = dc.field(
        hash=False
    )  # a 2-dimensional array containing the indices of the h-bonded nucleotides

    def __post_init__(self):
        if self.rigid_body_transform_fn is None:
            raise ValueError(ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED)

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the twist of the propeller in degrees.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the propeller twist for

        Returns:
            jd_types.ARR_OR_SCALAR: the propeller twist in degrees for each state and for each
            base pair, so expect a size of (n_states, n_base_pairs)
        """
        # TODO(ryanhausen): move this out to be jitted
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)

        base_normals = nucleotides.base_normals
        # ptwist_rad = single_propeller_twist_rad(self.h_bonded_base_pairs, base_normals)
        ptwist = jax.vmap(
            lambda bn: 180.0 - (single_propeller_twist_rad(self.h_bonded_base_pairs, bn) * 180.0 / jnp.pi)
        )
        print("outs", jnp.mean(ptwist(base_normals), axis=1))
        return jnp.mean(ptwist(base_normals), axis=1)


if __name__ == "__main__":
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

    simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    print("input rigid body", sim_traj.rigid_body.center.shape, sim_traj.rigid_body.orientation.vec.shape)
    prop_twist = PropellerTwist(rigid_body_transform_fn=tranform_fn, h_bonded_base_pairs=simple_helix_bps)
    output_prop_twist = prop_twist(sim_traj)
    print("output prop twist shape", output_prop_twist.shape)
    import jax_md
    # concate multiple states together to simulate a longer trajectory

    sim_traj = jd_sio.SimulatorTrajectory(
        seq_oh=jnp.array(top.seq_one_hot),
        strand_lengths=top.strand_counts,
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.concatenate([sim_traj.rigid_body.center, sim_traj.rigid_body.center], axis=0),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.concatenate([sim_traj.rigid_body.orientation.vec, sim_traj.rigid_body.orientation.vec], axis=0),
            ),
        ),
    )

    print("input rigid body", sim_traj.rigid_body.center.shape, sim_traj.rigid_body.orientation.vec.shape)

    print(prop_twist(sim_traj))
