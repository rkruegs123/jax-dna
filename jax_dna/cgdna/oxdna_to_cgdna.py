import pdb

import jax.numpy as jnp
# from jax.scipy import linalg
from jax import vmap

from utils import Q_to_back_base, Q_to_base_normal
import denman_beavers

sqrtm3x3 = denman_beavers.get_denman_beavers(3, 25)


OX_TO_ANG = 8.518 # 1 oxdna unit = 8.518 angstrongs

# Parameteres to map oxdna coordinates to interaction centers
# A = 0; C = 1, G = 2, T = 3
# Note: currently, only the COM is used to compute translations.
# oxdna 1.0 version:
STACK_X = [0.34*OX_TO_ANG, 0.34*OX_TO_ANG, 0.34*OX_TO_ANG, 0.34*OX_TO_ANG]
HB_X = [0.4*OX_TO_ANG, 0.4*OX_TO_ANG, 0.4*OX_TO_ANG, 0.4*OX_TO_ANG]
BB_X = [-0.4*OX_TO_ANG, -0.4*OX_TO_ANG, -0.4*OX_TO_ANG, -0.4*OX_TO_ANG]
BB_Y = [0.*OX_TO_ANG, 0.*OX_TO_ANG, 0.*OX_TO_ANG, 0.*OX_TO_ANG]



TZU_HB_PU = jnp.array([0,0.795,0.0])
TZU_HB_PI = jnp.array([0,2.5885,0.0])


# parameter to map nucleotide centers to Euler translations
TZU_C_PU = jnp.array([0, 5.11, 0.0])
TZU_C_PI = jnp.array([0, 5.11, 0.0])


# FIXME: is there any place to use a displacement function?

class OxdnaFrame:
    """
    Class for storing oxDNA coordinates.
    """
    def __init__(self, c, bv, n):
        self.center = c * OX_TO_ANG
        self.base_v = bv
        self.normal = n
        self.base_norv = jnp.cross(n,bv)

    def int_centers(self, ty):
        """
        Computes the positions of the interaction centers
        """
        tid = 0

        back_bone = self.center + BB_X[tid] * self.base_v + BB_Y[tid] * self.base_norv
        stack = self.center + STACK_X[tid] * self.base_v
        hbond = self.center + HB_X[tid] * self.base_v
        return (back_bone, hbond, stack)

class EFrame:
    """
    Class for representing euler coordinates
    """
    def __init__(self, p,ori):
        self.pos = p # note: np array
        self.orientation = ori # note: np matrix

class IntCoord:
    """
    Class for representing internal (intra or inter) coordinates
    """
    def __init__(self, tr,rot):
        self.tran = tr # np array
        self.rot = rot # np array

class Base:
    """
    A class for representing a nucleotide base, i.e.
    one nucleotide and its frames
    """
    def __init__(self, oxc, ty):
        self.oxframe = oxc # note: an instance of OxdnaFrame
        self.type = ty

        y = -oxc.base_v
        z = -oxc.normal
        x = -oxc.base_norv

        ori = jnp.column_stack((x,y,z))

        p = jnp.zeros(3,dtype=float)
        # mapping to center of mass. Gives same result

        p = jnp.where((ty == 0) | (ty == 2),
                      oxc.center - jnp.dot(ori, TZU_C_PU),
                      oxc.center - jnp.dot(ori, TZU_C_PI))

        self.frame = EFrame(p, ori) # note: an instance of EFrame


def caym1(A):
    """
    Computes the inverse of the Cayley transormation, which maps
    a vector to an SO(3) rotation. Note that m1 stands for -1. This
    function assumes radians.

    Note that v is vect(A - A.transposed())
    - vect(M), with M skew is defined as v = (M(2,1), M(0,2), M(1,0))
    - see Daiva Petkevičiūtė thesis (Maddocks' student)
    """
    c = 2 * 1 / (1+jnp.trace(A))
    v = jnp.zeros(3, dtype=float)
    M = A - A.transpose()

    v = v.at[0].set(M[2][1].real)
    v = v.at[1].set(M[0][2].real)
    v = v.at[2].set(M[1][0].real)

    t = c*v
    return t


class BasePair:
    """
    Class for representing a base pair.

    Stores two bases, a base pair frame (an instance of EFrame),
    and the intra coordinates
    """
    def __init__(self, b1, b2):
        self.base_W = b1 # +
        self.base_C = b2 # -

        # flip the Crick base
        F = jnp.zeros((3, 3), dtype=float)
        F = F.at[0,0].set(1.)
        F = F.at[1,1].set(-1.)
        F = F.at[2,2].set(-1.)

        # compute average bp frame
        p = (b1.frame.pos + b2.frame.pos)*0.5
        DC = jnp.dot(b2.frame.orientation, F) # flipped Crick frame
        A2 = jnp.dot(DC.transpose(), b1.frame.orientation)
        A = sqrtm3x3(A2)
        ori = jnp.dot(DC, A)
        self.frame = EFrame(p,ori)

        # compute intra coordinates
        rot = caym1(A2)
        tr = jnp.dot(self.frame.orientation.transpose(), b1.frame.pos - b2.frame.pos) # FIXME: use displacement function?
        self.intra_coord = IntCoord(tr.real, rot.real)


def get_reader(num_bases, num_base_pairs, seq):
    seq_mapper = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_ids = jnp.array([seq_mapper[nuc] for nuc in seq])

    def compute_bp_prop_twist(i, coms, bvs, normals):
        b1 = Base(OxdnaFrame(coms[i], bvs[i], normals[i]), seq_ids[i])
        b2 = Base(OxdnaFrame(coms[num_bases-i-1], bvs[num_bases-i-1], normals[num_bases-i-1]), seq_ids[num_bases-i-1])
        bp = BasePair(b1, b2)
        prop_twist = bp.intra_coord.rot[1]
        return prop_twist
    compute_all_bp_prop_twists = vmap(compute_bp_prop_twist, (0, None, None, None))

    def time_step_fn(state): # state is a RigidBody for a single point in time
        coms = state.center
        bvs = Q_to_back_base(state.orientation)
        normals = Q_to_base_normal(state.orientation)
        all_prop_twists = compute_all_bp_prop_twists(jnp.arange(num_base_pairs), coms, bvs, normals)
        return all_prop_twists

    def reader(trajectory): # trajectory is a RigidBody
        traj_prop_twists = vmap(time_step_fn)(trajectory)
        traj_prop_twists_deg = (180/jnp.pi)*traj_prop_twists
        return traj_prop_twists_deg
    return reader
