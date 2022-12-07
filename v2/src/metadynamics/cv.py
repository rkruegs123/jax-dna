import pdb
from functools import partial
# import sys
# sys.path.insert(0, 'src/')

import jax.numpy as jnp
from jax_md import space

from utils import Q_to_back_base
from utils import com_to_hb


# FIXME: sigma is oxDNA length unit. Must fix.
A = 1.30
B = 0.053
DM_COEFF = 1.5
SIGMA = 0.34 # FIXME: oxDNA length unit
# SIGMA = 8
DM = DM_COEFF * SIGMA

def cv(d):
    num = A*(1 - (d/DM)**6)
    denom = 1 - (d/DM)**12 # FIXME: the paper has |d| instead of d, but won't this always be true?
    return num/denom - B

# Returns a function that, given a RigidBody, returns the number of base pairs
# Takes a list of pairs of indices, representing paired nucleotides
# FIXME: is d(i, j) defined as the distance between centers of masses? Hydrogen bonding sites?
def get_n_bp_fn(bps, displacement_fn):
    d = space.map_bond(partial(displacement_fn))

    bp_i = bps[:, 0]
    bp_j = bps[:, 1]

    def get_n_bp(body):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        base_sites = body.center + com_to_hb * back_base_vectors

        dr_base = d(base_sites[bp_i], base_sites[bp_j])
        r_base = jnp.linalg.norm(dr_base, axis=1)

        cvs = cv(r_base)
        n_bp = jnp.sum(cvs)
        return n_bp
    return get_n_bp

def plot_cv():
    import numpy as onp
    import matplotlib.pyplot as plt

    c_intended = cv(1.27*SIGMA)
    print(f"c(d) at d = 1.27*SIGMA: {c_intended}")

    ds = onp.linspace(-3*SIGMA, 3*SIGMA, 20)
    cvs = cv(ds)
    plt.axvline(1.27*SIGMA)
    plt.plot(ds, cvs)
    plt.show()

    return


def get_interstrand_dist_fn(bps, displacement_fn):

    d = space.map_bond(partial(displacement_fn))

    bp_i = bps[:, 0]
    bp_j = bps[:, 1]

    n = bps.shape[0]

    def interstrand_dist_fn(interstrand_dist_fn):
        Q = body.orientation
        back_base_vectors = Q_to_back_base(Q)
        base_sites = body.center + com_to_hb * back_base_vectors

        norm_strand_1_sum = jnp.sum(base_sites[bp_i], axis=0) / n
        norm_strand_2_sum = jnp.sum(base_sites[bp_j], axis=0) / n

        cv = jnp.linalg.norm(norm_strand_1_sum - norm_strand_2_sum, axis=0)
        pdb.set_trace()
        return cv

    return interstrand_dist_fn



# FIXME: can test by simulating a helix with unbound frays and passing various states in here
if __name__ == "__main__":
    # plot_cv()

    # pdb.set_trace()


    # import sys
    # sys.path.append('src/')
    from loader.trajectory import TrajectoryInfo
    from loader.topology import TopologyInfo

    # top_path = "data/test-data/simple-helix/generated.top"
    # conf_path = "data/test-data/simple-helix/start.conf"
    # traj_path = "data/test-data/simple-helix/output.dat"

    top_path = "data/test-data/unbound-strands-overlap/generated.top"
    conf_path = "data/test-data/unbound-strands-overlap/start.conf"
    traj_path = "data/test-data/unbound-strands-overlap/output.dat"

    top_info = TopologyInfo(top_path, reverse_direction=True)
    config_info = TrajectoryInfo(top_info, traj_path=conf_path, reverse_direction=True)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)

    # body = config_info.states[0]
    body = traj_info.states[-1]

    displacement_fn, shift_fn = space.periodic(config_info.box_size)
    bps = jnp.array([
        # [0, 15],
        [1, 14],
        [2, 13],
        [3, 12],
        [4, 11],
        [5, 10],
        [6, 9],
        # [7, 8]
    ])

    # n_bp_fn = get_n_bp_fn(bps, displacement_fn)
    # n_bp = n_bp_fn(body)
    # print(f"# Base Pairs: {n_bp}")

    interstrand_dist_fn = get_interstrand_dist_fn(bps, displacement_fn)
    interstrand_dist = interstrand_dist_fn(body)
    print(f"Interstrand Distance: {interstrand_dist}")

    pdb.set_trace()
