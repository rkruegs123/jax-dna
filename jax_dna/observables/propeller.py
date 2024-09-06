import dataclasses as dc
import functools

import chex
import jax
import jax.numpy as jnp
import jax_dna.input.trajectory as jd_traj
import jax_dna.observables.base as jd_obs
import jax_dna.utils.math as jd_math
import jax_dna.utils.types as jd_types

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

    def __call__(self, trajectory: jd_traj.Trajectory) -> jd_types.ARR_OR_SCALAR:
        """Calculate the twist of the propeller in degrees.

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the propeller twist for

        Returns:
            jd_types.ARR_OR_SCALAR: the propeller twist in degrees for each state and for each
            base pair, so expect a size of (n_states, n_base_pairs)
        """

        base_normals = jnp.stack(list(map(lambda body: body.base_normal, trajectory.states)))
        ptwist_rad = jax.vmap(lambda normals: single_propeller_twist_rad(self.h_bonded_base_pairs, normals))(
            base_normals
        )

        return 180.0 - (ptwist_rad * 180.0 / jnp.pi)


if __name__ == "__main__":
    import jax_dna.input.topology as jd_top

    top = jd_top.from_oxdna_file("data/sys-defs/simple-helix/sys.top")
    test_traj = jd_traj.from_file(
        path="data/sys-defs/simple-helix/bound_relaxed.conf",
        strand_lengths=top.strand_counts,
    )
    simple_helix_bps = jnp.array([[1, 14], [2, 13], [3, 12], [4, 11], [5, 10], [6, 9]])
    prop_twist = PropellerTwist(h_bonded_base_pairs=simple_helix_bps)
    print(prop_twist(test_traj))  # expect a 2D array of shape (n_states, n_base_pairs)
