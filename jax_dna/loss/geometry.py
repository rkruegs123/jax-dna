import pdb
from functools import partial

import jax.numpy as jnp
from jax_md import space

from jax_dna.common import utils


# DNA Structure and Function, R. Sinden, 1st ed
# Table 1.3, Pg 27
TARGET_HELICAL_DIAMETER = utils.angstroms_to_oxdna_length(20.0)

def get_helical_diameter_loss_fn(bp_pairs, displacement_fn, com_to_backbone,
                                 target_distance=TARGET_HELICAL_DIAMETER):
    """
    Constructs two functions:
    1) A function that computes the helical distance for each base pair
    in a rigid body
    2) A function that uses these computed helical distances to compute
    a loss for a given rigid body

    We treat the helical diameter as the distance between the edges of
    the backbone sides between the two phosphate backbone atoms in a
    base pair. So, to compute the helical diameter for a given base pair,
    we first compute the distance between the COMs of the phosphates
    and then add 2*(phosphate radius)

    Note that the phosphate radius is a parameter over which we could
    optimize (i.e. it is an excluded volume parameter). So, we make
    the loss function a function of this radius.
    """

    d = space.map_bond(partial(displacement_fn))
    bp_i = bp_pairs[:, 0]
    bp_j = bp_pairs[:, 1]

    def compute_helical_diameters(body, backbone_radius=0.675):
        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[bp_i], back_sites[bp_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)
        return r_back + 2*backbone_radius

    def loss_fn(body, backbone_radius=0.675):
        helical_distances = compute_helical_diameters(body, backbone_radius)
        return jnp.mean((helical_distances - target_distance)**2)

    return compute_helical_diameters, loss_fn


# DNA Structure and Function, R. Sinden, 1st ed
# Table 1.3, Pg 27
TARGET_PHOS_PHOS_DIST = angstroms_to_oxdna_length(7.0)

def get_backbone_distance_loss_fn(bonded_nbrs, displacement_fn, com_to_backbone,
                                  target_distance=TARGET_PHOS_PHOS_DIST):

    d = space.map_bond(partial(displacement_fn))
    bonded_nbrs_i = bonded_nbrs[:, 0]
    bonded_nbrs_j = bonded_nbrs[:, 1]

    def compute_backbone_distances(body):
        Q = body.orientation
        back_base_vectors = utils.Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[bonded_nbrs_i], back_sites[bonded_nbrs_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)
        return r_back

    def loss_fn(body):
        backbone_distances = compute_backbone_distances(body)
        return jnp.sum(((backbone_distances - target_distance))**2)

    return compute_backbone_distances, loss_fn


if __name__ == "__main__":
    pass
