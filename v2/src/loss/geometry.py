import pdb
from functools import partial

import jax.numpy as jnp
from jax_md import space

# import sys
# sys.path.append("/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/v2/src")
# pdb.set_trace()
from utils import Q_to_back_base
from utils import com_to_backbone, com_to_stacking, com_to_hb
from utils import angstroms_to_oxdna_length

from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo



# DNA Structure and Function, R. Sinden, 1st ed
# Table 1.3, Pg 27
TARGET_PHOS_PHOS_DIST = angstroms_to_oxdna_length(7.0)

# DNA Structure and Function, R. Sinden, 1st ed
# Table 1.3, Pg 27
TARGET_HELICAL_DIAMETER = angstroms_to_oxdna_length(20.0)



"""
Strategy for the helical diameter:
- We treat the helical diameter as the distance between the edges of the backbone sides between the two phosphate backbone atoms in a base pair.
- From our current implementation, we can only compute the distance between the COMs of both phosphates
- So, we can correct this distance by adding 2*(phosphate radius)
- However, the phosphate radius (i.e. the backbone radius) is effectively a parameter over which we will optimize, as we optimize over the excluded volume parameters.
  - By the definition of Tom's parameters, this parameter is `dr_star_backbone`
- So, the loss function will have to take in the current parameter set.
- This will require that the energy factory does not load the parameters, but instead accepts them from elsewhere (e.g. the optimization code)
- For now, until we optimize over the excluded volume parameters, we fix the target value to be Tom's value
"""
backbone_radius = 0.675 # FIXME: when we optimize over excluded volume parameters, this will have to be made variable
def get_helical_diameter_loss(bp_pairs, displacement_fn, target_distance=TARGET_HELICAL_DIAMETER):
    d = space.map_bond(partial(displacement_fn))
    bp_i = bp_pairs[:, 0]
    bp_j = bp_pairs[:, 1]
    def helical_diameter_loss(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[bp_i], back_sites[bp_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)
        r_back += 2*backbone_radius
        # Note: maybe to decide between sum and mean, we determine whether or not the target property is a bulk proeprty or one that should hold for each individual term (e.g. bond lengtH)
        # return jnp.sum(((r_back - HELICAL_RADIUS))**2)
        return jnp.mean(((r_back - target_distance))**2)
    return helical_diameter_loss


def get_backbone_distance_loss(pairs, displacement_fn,
                               target_distance=TARGET_PHOS_PHOS_DIST):

    d = space.map_bond(partial(displacement_fn))
    nbrs_i = pairs[:, 0]
    nbrs_j = pairs[:, 1]

    # note: so much repeat computatoin given that we've already computed this...
    def backbone_distance_loss(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        back_sites = body.center + com_to_backbone * back_base_vectors
        dr_back = d(back_sites[nbrs_i], back_sites[nbrs_j])
        r_back = jnp.linalg.norm(dr_back, axis=1)

        return jnp.sum(((r_back - target_distance))**2)
        # return jnp.mean(((r_back - target_distance))**2)
        # return r_back, jnp.sum((100*(r_back - target_distance))**2)
        # avg_r_back = jnp.mean(r_back)
        # return (avg_r_back - target_distance)**2
    return backbone_distance_loss


if __name__ == "__main__":


    top_path = "data/simple-helix/generated.top"
    # traj_path = "data/simple-helix/output.dat"
    config_path = "data/simple-helix/start.conf"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=config_path, reverse_direction=True)

    body = config_info.states[0]

    displacement_fn, _ = space.periodic(config_info.box_size)

    # pairs = top_info.bonded_nbrs[2:4]
    pairs = top_info.bonded_nbrs
    loss_fn = get_backbone_distance_loss(pairs, displacement_fn)
    curr_loss = loss_fn(body)
    pdb.set_trace()
    print("done")
