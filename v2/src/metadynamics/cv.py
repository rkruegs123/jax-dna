import pdb
from functools import partial
import sys
sys.path.insert(0, 'src/')

import jax.numpy as jnp
from jax_md import space
from jax.tree_util import Partial
from jax import vmap

from utils import Q_to_back_base, Q_to_base_normal
from utils import com_to_hb
import metadynamics.utils as md_utils


# FIXME: sigma is oxDNA length unit. Must fix.
A = 1.30
B = 0.053
DM_COEFF = 1.5
SIGMA = 0.34 # FIXME: oxDNA length unit
# SIGMA = 8
DM = DM_COEFF * SIGMA

def cv_review(d):
    num = A*(1 - (d/DM)**6)
    denom = 1 - (d/DM)**12 # FIXME: the paper has |d| instead of d, but won't this always be true?
    return num/denom - B


cv_alpha = 20.0
# hb_distance = 1.27
hb_distance = 0.25
hb_distance_eps = hb_distance + 0.40 # where it will be 0.5
cv_sigmoid = lambda d: 1 - 1 / (1 + jnp.exp(-cv_alpha * (d - hb_distance_eps)))


# Returns a function that, given a RigidBody, returns the number of base pairs
# Takes a list of pairs of indices, representing paired nucleotides
# FIXME: is d(i, j) defined as the distance between centers of masses? Hydrogen bonding sites?
def get_n_bp_fn_original(bps, displacement_fn):
    d = space.map_bond(partial(displacement_fn))

    bp_i = bps[:, 0]
    bp_j = bps[:, 1]

    def get_n_bp(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        base_sites = body.center + com_to_hb * back_base_vectors

        dr_base = d(base_sites[bp_i], base_sites[bp_j])
        r_base = jnp.linalg.norm(dr_base, axis=1)

        cvs = cv_review(r_base)
        n_bp = jnp.sum(cvs)
        return n_bp
    return get_n_bp

def plot_cv():
    import numpy as onp
    import matplotlib.pyplot as plt

    c_intended = cv_review(1.27*SIGMA)
    print(f"c(d) at d = 1.27*SIGMA: {c_intended}")

    # ds = onp.linspace(-3*SIGMA, 3*SIGMA, 20)
    ds = onp.linspace(-1, 4, 100)
    cvs = cv_review(ds)
    # plt.axvline(1.27*SIGMA)
    plt.axvline(1.27)
    plt.axvline(0.4)
    plt.plot(ds, cvs)
    plt.show()

    return


def get_interstrand_dist_fn(bps, displacement_fn):

    d = space.map_bond(partial(displacement_fn))

    bp_i = bps[:, 0]
    bp_j = bps[:, 1]

    n = bps.shape[0]

    def interstrand_dist_fn(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        base_sites = body.center + com_to_hb * back_base_vectors

        norm_strand_1_sum = jnp.sum(base_sites[bp_i], axis=0) / n
        norm_strand_2_sum = jnp.sum(base_sites[bp_j], axis=0) / n

        cv = jnp.linalg.norm(norm_strand_1_sum - norm_strand_2_sum, axis=0)
        return cv

    return interstrand_dist_fn


def get_theta_fn(a_3p_idx, a_5p_idx, b_3p_idx, b_5p_idx):
    def theta_fn(body):
        a_vec = body.center[a_3p_idx] - body.center[a_5p_idx]
        b_vec = body.center[b_3p_idx] - body.center[b_5p_idx]

        num = jnp.dot(a_vec, b_vec)
        denom = jnp.linalg.norm(a_vec) * jnp.linalg.norm(b_vec)
        return jnp.arccos(num / denom)
    return theta_fn


# Custom dot-product trick
def get_n_bp_fn_custom(bps, displacement_fn, method):
    d = space.map_bond(partial(displacement_fn))

    bp_i = bps[:, 0]
    bp_j = bps[:, 1]

    height = 1.0
    center = -1.0
    width = jnp.sqrt(0.05)
    gauss_fn = lambda x: height * jnp.exp(-(x-center)**2/(2*width**2)) # FIXME: should really just use partial(gaussian) correctly
    gauss_fn = vmap(gauss_fn)

    assert(method in ["sigmoid", "review"])
    cv_fn = cv_sigmoid if method == "sigmoid" else cv_review

    def get_n_bp(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        base_sites = body.center + com_to_hb * back_base_vectors

        dr_base = d(base_sites[bp_i], base_sites[bp_j])
        r_base = jnp.linalg.norm(dr_base, axis=1)

        cvs = cv_fn(r_base)

        base_normals = Q_to_base_normal(Q) # space frame, normalized
        norm_dot_products = jnp.einsum('ij, ij->i', base_normals[bp_i], base_normals[bp_j])
        clamped_norm_dps = gauss_fn(norm_dot_products)

        n_bp = jnp.dot(cvs, clamped_norm_dps)
        return n_bp
    return get_n_bp


# FIXME: can test by simulating a helix with unbound frays and passing various states in here
if __name__ == "__main__":
    # plot_cv()

    # pdb.set_trace()


    # import sys
    # sys.path.append('src/')
    from loader.trajectory import TrajectoryInfo
    from loader.topology import TopologyInfo

    top_path = "data/test-data/simple-helix/generated.top"
    conf_path = "data/test-data/simple-helix/start.conf"
    traj_path = "data/test-data/simple-helix/output.dat"

    # top_path = "../../tmp-oxdna/metad_2022-12-16_21-41-38/generated.top"
    # conf_path = "../../tmp-oxdna/metad_2022-12-16_21-41-38/start.conf"
    # traj_path = "../../tmp-oxdna/metad_2022-12-16_21-41-38/output.dat"

    # top_path = "data/test-data/unbound-strands-overlap/generated.top"
    # conf_path = "data/test-data/unbound-strands-overlap/start.conf"
    # traj_path = "data/test-data/unbound-strands-overlap/output.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

    # body = config_info.states[0]
    # body = traj_info.states[-1]
    body = traj_info.states[2]

    displacement_fn, shift_fn = space.periodic(config_info.box_size)
    bps = jnp.array([
        [0, 15],
        [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        [6, 9],
        [7, 8]
    ])

    n_bp_fn = get_n_bp_fn_custom(bps, displacement_fn)
    n_bp = n_bp_fn(body)
    print(f"# Base Pairs: {n_bp}")

    pdb.set_trace()
    interstrand_dist_fn = get_interstrand_dist_fn(bps, displacement_fn)
    interstrand_dist = interstrand_dist_fn(body)
    print(f"Interstrand Distance: {interstrand_dist}")

    pdb.set_trace()
