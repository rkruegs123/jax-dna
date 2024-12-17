"""Persistence length observable."""

import dataclasses as dc
import functools
from collections.abc import Callable
import pdb
from typing import Tuple

import chex
import jax
from jax import vmap
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
    "oxDNA": 47.5, # nm
}


def persistence_length_fit(correlations: jnp.ndarray, l0_av: float) -> Tuple[float, float]:
    """Computes the Lp given correlations in alignment decay and average distance between base pairs.

    Lp obeys the following equality: `<l_n * l_0> = exp(-n<l_0> / Lp)`, where `<l_n * l_0>` represents the
    average correlation between adjacent base pairs (`l_0`) and base pairs separated by a distance of
    `n` base pairs (`l_n`). This relationship is linear in log space, `log(<l_n * l_0>) = -n<l_0> / Lp`.
    So, given the average correlations across distances and the average distance between adjacent base pairs,
    we compute Lp via a linear fit.

    Args:
    - correlations: a (max_dist,) array containing the average correlation between base pairs separated by
      distances up to `max_dist`
    - l0_av: the average distance between adjacent base pairs
    """

    # Format the correlations for a linear fit
    y = jnp.log(correlations)
    x = jnp.arange(correlations.shape[0])
    x = jnp.stack([jnp.ones_like(x), x], axis=1)

    # Fit a line
    fit_ = jnp.linalg.lstsq(x, y)

    # Extract slope and offset, and compute Lp
    offset = fit_[0][0]
    slope = fit_[0][1] # slope = -l0_av / Lp
    Lp = -l0_av / slope

    return Lp, offset


def compute_l_vector(base_sites: jnp.ndarray, quartet: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Computes the distance between two adjacent base pairs"""

    # Extract the two base pairs defined by a quartet
    bp1, bp2 = quartet
    (a1, b1), (a2, b2) = bp1, bp2 # a1 and b1, and a2 and b2 are base paired

    # Compute midpoints for each base pair
    mp1 = (base_sites[b1] + base_sites[a1]) / 2.
    mp2 = (base_sites[b2] + base_sites[a2]) / 2.

    # Compute vector between midpoints
    l = mp2 - mp1
    l0 = jnp.linalg.norm(l)
    l /= l0

    # Return vector and its norm
    return l, l0
get_all_l_vectors = vmap(compute_l_vector, in_axes=(None, 0))


def vector_autocorrelate(vecs: jnp.ndarray) -> jnp.ndarray:
    """Computes the average correlations in alignment decay between a list of vector.

    Given an ordered list of n vectors (representing vectors between adjacent base pairs),
    computes the average correlation between all pairs of vectors separated by a distance `d`
    for all distances `d < n`. Note that multiple pairs of vectors are included for all
    values < n-1.

    Args:
    - vecs: a (n, 3) array of vectors corresponding to displacements between midpoints of adjacent
      base pairs.

    """

    max_dist = vecs.shape[0]

    def window_correlations(i):
        li = vecs[i]
        i_correlation_fn = lambda j: jnp.where(j >= i, jnp.dot(li, vecs[j]), 0.0)
        i_correlations = vmap(i_correlation_fn)(jnp.arange(max_dist))
        i_correlations = jnp.roll(i_correlations, -i)
        return i_correlations

        all_correlations += i_correlations

    all_correlations = vmap(window_correlations)(jnp.arange(max_dist))
    all_correlations = jnp.sum(all_correlations, axis=0)

    all_correlations /= jnp.arange(max_dist, 0, -1)
    return all_correlations


def compute_metadata(base_sites: jnp.ndarray, quartets: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
    """Computes (i) average correlations in alignment decay and (ii) average distance between base pairs"""

    all_l_vectors, l0_vals = get_all_l_vectors(base_sites, quartets)
    autocorr = vector_autocorrelate(all_l_vectors)
    return autocorr, jnp.mean(l0_vals)



@chex.dataclass(frozen=True, kw_only=True)
class LpMetadata(jd_obs.BaseObservable):
    """Computes the metadata relevant for computing the persistence length (Lp) for each state.

    To model Lp, we assume an infinitely long, semi-flexible polymer, in which correlations in
    alignment decay exponentially with separation. So, to compute Lp, we need the average correlations
    across many states, as well as the average distance between adjacent base pairs. This observable
    computes these two quantities for a single state, and the average of these quantities across
    a trajectory can be postprocessed to compute a value for Lp.

    Args:
    - quartets: a (n_bp, 2, 2) array containing the pairs of adjacent base pairs
      for which to compute the Lp
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

    def __call__(self, trajectory: jd_sio.SimulatorTrajectory) -> Tuple[jnp.ndarray, jd_types.ARR_OR_SCALAR]:
        """Calculate the correlations in alignment decay and average distance between adjacent
        base pairs for each state in a trajectory..

        Args:
            trajectory (jd_traj.Trajectory): the trajectory to calculate the rise for

        Returns:
            Tuple[jnp.ndarray, jd_types.ARR_OR_SCALAR]: the correlations in alignment decay and the the average
            distance between adjacent base pairs for each state. The former will have shape (n_states, n_quartets-1)
            and the latter will have shape (n_states,).
        """
        nucleotides = jax.vmap(self.rigid_body_transform_fn)(trajectory.rigid_body)
        base_sites = nucleotides.base_sites

        all_corrs, all_l0_vals = vmap(compute_metadata, (0, None))(base_sites, self.quartets)

        return all_corrs, all_l0_vals


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import jax_md
    import jax_dna.input.topology as jd_top


    test_geometry = jd_toml.parse_toml("jax_dna/input/dna1/default_energy.toml")["geometry"]
    tranform_fn = functools.partial(
        jd_energy.Nucleotide.from_rigid_body,
        com_to_backbone=test_geometry["com_to_backbone"],
        com_to_hb=test_geometry["com_to_hb"],
        com_to_stacking=test_geometry["com_to_stacking"],
    )

    top = jd_top.from_oxdna_file("data/templates/persistence-length/sys.top")
    test_traj = jd_traj.from_file(
        path="data/templates/persistence-length/init.conf",
        strand_lengths=top.strand_counts,
    )

    sim_traj = jd_sio.SimulatorTrajectory(
        seq_oh=jnp.array(top.seq_one_hot),
        strand_lengths=top.strand_counts,
        rigid_body=test_traj.state_rigid_body,
    )

    quartets = jd_obs.get_duplex_quartets(202)
    displacement_fn, _ = space.free()
    lp_metadata = LpMetadata(rigid_body_transform_fn=tranform_fn, quartets=quartets, displacement_fn=displacement_fn)
    output_all_corrs, output_all_l0_vals = lp_metadata(sim_traj)


    mean_all_corrs = jnp.mean(output_all_corrs, axis=0)
    mean_l0_val = jnp.mean(output_all_l0_vals, axis=0)

    truncation = 40
    fit_lp, fit_offset = persistence_length_fit(mean_all_corrs[:truncation], mean_l0_val)

    log_corr_fn = lambda n: -n * mean_l0_val / (fit_lp) + fit_offset
    plt.plot(jnp.log(mean_all_corrs[:truncation]))
    plt.plot(log_corr_fn(jnp.arange(mean_all_corrs[:truncation].shape[0])), linestyle='--')
    plt.xlabel("Distance")
    plt.ylabel("Log-Correlation")
    plt.show()
