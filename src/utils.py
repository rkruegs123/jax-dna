import toml
from pathlib import Path
import pdb
import numpy as np
from itertools import combinations
from io import StringIO
import pandas as pd

from jax import vmap
from jax_md.rigid_body import Quaternion, RigidBody
import jax.numpy as jnp

from smoothing import get_f1_smoothing_params, get_f2_smoothing_params, get_f3_smoothing_params, \
    get_f4_smoothing_params, get_f5_smoothing_params

from jax.config import config
config.update("jax_enable_x64", True)




# Probabilistic sequence utilities
DNA_MAPPER = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}
DNA_BASES = set("ACGT")

# seq is a string
def get_one_hot(seq):
    seq = seq.upper()
    if not set(seq).issubset(DNA_BASES):
        raise RuntimeError(f"Sequence contains bases other than ACGT: {seq}")
    seq_one_hot = [DNA_MAPPER[b] for b in seq]
    return np.array(seq_one_hot, dtype=np.float64) # float so they can become probabilistic


# oxDNA unit conversions

# Tom's thesis, page 23, bottom
ang_per_oxdna_length = 8.518
def angstroms_to_oxdna_length(ang):
    return ang / ang_per_oxdna_length
def oxdna_length_to_angstroms(l):
    return l * ang_per_oxdna_length
# Tom's thesis, page 24, top
joules_per_oxdna_energy = 4.142e-20
def joules_to_oxdna_energy(j):
    return j / joules_per_oxdna_energy
def oxdna_energy_to_joules(e):
    return e * joules_per_oxdna_energy
kb = 1.380649e-23 # joules per kelvin
kb_oxdna = kb / joules_per_oxdna_energy # oxdna energy per kelvin

"""
Option 1: Exact, but source of minor error
def get_kt(t): # t is temperature in kelvin
    return kb_oxdna * t
"""
# Option 2: Inexact, but exactly what was intended (i.e. kt=0.1E @ 300 K)
def get_kt(t):
    return 0.1 * t/300.0
# Tom's thesis, page 36, bottom (Section 2.5)
amu_per_oxdna_mass = 100
def amu_to_oxdna_mass(amu):
    return amu / amu_per_oxdna_mass
def oxdna_mass_to_amu(m):
    return m * amu_per_oxdna_mass

"""
nucleotide_mass = 3.1575 # 3.1575 M
moment_of_inertia = 0.43512
"""

# Peter's thesis changes mass and moment of inertia to 1
nucleotide_mass = 1.0
moment_of_inertia = [1.0, 1.0, 1.0]

backbone_to_com = 0.24 # 0.24 l. Tom's thesis, page 36, bottom (Section 2.5)
backbone_to_stacking_angstroms = 6.3 # 6.3 A. Tom's thesis, page 23
backbone_to_stacking = angstroms_to_oxdna_length(backbone_to_stacking_angstroms)
backbone_to_hb_angstroms = 6.8 # 6.8 A. Tom's thesis, page 23
backbone_to_hb = angstroms_to_oxdna_length(backbone_to_hb_angstroms)
"""
Diagram (not to scale):
(backbone)----[com]------(stacking)--(hb)
"""
"""
Note: for whatever reason, using the 0.24 value from Tom's thesis doesn't give the values from the oxDNA code and/or documentation. We use those directly, without understanding why they are the way they are
# FIXME: the computed values here for `com_to_X` conflict with the values in "Geometry of the Model": https://dna.physics.ox.ac.uk/index.php/Documentation
com_to_stacking = backbone_to_stacking - backbone_to_com
com_to_hb = backbone_to_hb - backbone_to_com
com_to_backbone = -backbone_to_com
"""
com_to_stacking = 0.34
com_to_hb = 0.4
com_to_backbone = -0.4



# Box size is float, jax_traj is list of RigidBody's
def jax_traj_to_oxdna_traj(jax_traj, box_size, every_n=1, output_name="test.dat"):
    # jax_traj is a list of state.position (of rigid bodies)
    output_lines = list()
    n = jax_traj[0].center.shape[0]

    for i, st in enumerate(jax_traj):
        if i % every_n != 0:
            continue
        output_lines.append(f"t = {i}\n")
        output_lines.append(f"b = {box_size} {box_size} {box_size}\n") # FIXME: only cubes for now
        output_lines.append(f"E = 0.0 0.0 0.0\n") # FIXME: dummy
        back_base_vectors = Q_to_back_base(st.orientation)
        base_normal_vectors = Q_to_base_normal(st.orientation)
        velocities = np.zeros((n, 3))
        angular_velocities = np.zeros((n, 3))

        for idx in range(n):
            line_vals = np.concatenate((st.center[idx], back_base_vectors[idx],
                                        base_normal_vectors[idx], velocities[idx],
                                        angular_velocities[idx])).astype(str)
            output_lines.append(' '.join(line_vals) + "\n")


    pdb.set_trace()
    with open(output_name, 'w') as of:
        of.writelines(output_lines)

    return



def principal_axes_to_euler_angles(x, y, z):
    """
    There are two options to compute the Tait-Bryan angles. Each can be seen at the respective links:
    (1) From wikipedia (under Tait-Bryan angles): https://en.wikipedia.org/wiki/Euler_angles
    (2) Equation 10A-C: https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

    However, note that the definition from Wikipedia (i.e. the one using arcsin) has numerical stability issues,
    so we use the definition from (2) (i.e. the one using arctan2)

    Note that if we were following (1), we would do:
    psi = np.arcsin(x[1] / np.sqrt(1 - x[2]**2))
    theta = np.arcsin(-x[2])
    phi = np.arcsin(y[2] / np.sqrt(1 - x[2]**2))

    Note that Tait-Bryan (i.e. Cardan) angles are *not* proper euler angles
    """

    psi = np.arctan2(x[1], x[0])
    if np.abs(x[2]) > 1:
        # FIXME: could clamp?
        # pdb.set_trace()
        x[2] = np.round(x[2])
    theta = np.arcsin(-x[2])
    phi = np.arctan2(y[2], z[2])

    return psi, theta, phi



# Takes in a list of lines and returns a RigidBody
# Note: it is the burden of the user of this function to pass the right number of lines
# in other words, this function should be able to infer `n`
def read_state(state_df):
    n = state_df.shape[0]
    R = np.empty((n, 3), dtype=np.float64)
    quat = np.empty((n, 4), dtype=np.float64)

    # for i, nuc_line in state_df.iterrows(): # i won't start at 0 as iterrows() sets `i` to be the absolute index
    for i, (idx, nuc_line) in enumerate(state_df.iterrows()):
        nuc_info = nuc_line.tolist()
        assert(len(nuc_info) == 16)
        nuc_info = nuc_info[1:] # remove time

        com = nuc_info[:3]
        back_base_vector = nuc_info[3:6] # back_base
        base_normal = nuc_info[6:9] # base_norm
        velocity = nuc_info[9:12]
        angular_velocity = nuc_info[12:15]

        # Method 1
        alpha, beta, gamma = principal_axes_to_euler_angles(back_base_vector,
                                                            np.cross(base_normal, back_base_vector),
                                                            base_normal)

        # https://ntrs.nasa.gov/citations/19770024290
        # https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
        # Page A-11 (ZYX)
        def get_q(t1, t2, t3):
            q0 = np.sin(0.5*t1)*np.sin(0.5*t2)*np.sin(0.5*t3) + np.cos(0.5*t1)*np.cos(0.5*t2)*np.cos(0.5*t3)
            q1 = -np.sin(0.5*t1)*np.sin(0.5*t2)*np.cos(0.5*t3) + np.sin(0.5*t3)*np.cos(0.5*t1)*np.cos(0.5*t2)
            q2 = np.sin(0.5*t1)*np.sin(0.5*t3)*np.cos(0.5*t2) + np.sin(0.5*t2)*np.cos(0.5*t1)*np.cos(0.5*t3)
            q3 = np.sin(0.5*t1)*np.cos(0.5*t2)*np.cos(0.5*t3) - np.sin(0.5*t2)*np.sin(0.5*t3)*np.cos(0.5*t1)
            # q = Quaternion(np.array([q0, q1, q2, q3]))
            return q0, q1, q2, q3

        q0, q1, q2, q3 = get_q(alpha, beta, gamma)


        # For testing
        # q = Quaternion(np.array([q0, q1, q2, q3]))
        # recovered_back_base = q_to_back_base(q)  # should equal back_base_vector


        # Method 2 -- BROKEN
        """
        # https://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
        # Table 1, page 6 gives the accuracy of this method -- it's terrible. This is also summarized in the conclusion on page 8


        rot_matrix = np.array([back_base_vector, np.cross(base_normal, back_base_vector), base_normal]).T
        tr = np.trace(rot_matrix)
        q0 = np.sqrt((tr + 1) / 4)
        q1 = np.sqrt(rot_matrix[0, 0] / 2 + (1 - tr) / 4)
        q2 = np.sqrt(rot_matrix[1, 1] / 2 + (1 - tr) / 4)
        q3 = np.sqrt(rot_matrix[2, 2] / 2 + (1 - tr) / 4)
        """



        """
        # Testing
        q = Quaternion(np.array([q0, q1, q2, q3]))
        recovered_back_base = q_to_back_base(q) # should equal back_base_vector
        recovered_cross_prod = q_to_cross_prod(q) # should equal np.cross(base_normal, back_base_vector)
        recovered_base_normal = q_to_base_normal(q) # should equal base_normal
        """

        R[i, :] = com
        quat[i, :] = np.array([q0, q1, q2, q3])

    body = RigidBody(R, Quaternion(quat))
    return body




def _read_traj_info(traj_lines, n):
    assert(len(traj_lines) % (n+3) == 0)
    time_steps = int(len(traj_lines) / (n+3))
    all_state_lines = [traj_lines[(n+3)*t:(n+3)*t+(n+3)]  for t in range(time_steps)]

    # Construct trajectory df
    df_lines = list()
    bs = list()
    Es = list()
    ts = list()
    for state_lines in all_state_lines:
        t = float(state_lines[0].split('=')[1].strip())
        ts.append(t)

        b = state_lines[1].split('=')[1].strip().split(' ')
        b = np.array(b).astype(np.float64)
        bs.append(b)

        E = state_lines[2].split('=')[1].strip().split(' ')
        E = np.array(E).astype(np.float64)
        Es.append(E)

        t_lines = [[t] + state_info.strip().split() for state_info in state_lines[3:]]
        df_lines += t_lines


    ts = np.array(ts, dtype=np.float64)
    bs = np.array(bs, dtype=np.float64)
    Es = np.array(Es, dtype=np.float64)
    traj_df = pd.DataFrame(df_lines,
                           columns=["t",
                                    "com_x", "com_y", "com_z",
                                    "a1_x", "a1_y", "a1_z",
                                    "a3_x", "a3_y", "a3_z",
                                    "v_x", "v_y", "v_z",
                                    "L_x", "L_y", "L_z"],
                           dtype=float)
    return traj_df, ts, bs, Es



# Helper for `read_3to5`
# FIXME: should leave better comments here -- e.g. that read_topology only operates on 5' to 3'. Also,
# FIXME: should also have 5to3 to 3to5
def _read_3to5(top_lines_3to5, traj_lines_3to5):

    traj_info = None

    sys_info = top_lines_3to5[0].strip().split()
    n = int(sys_info[0])
    n_strands = int(sys_info[1])

    top_df = pd.read_csv(StringIO('\n'.join(top_lines_3to5[1:])),
                     names=["strand", "base", "3p_nbr", "5p_nbr"],
                     delim_whitespace=True)

    pdb.set_trace()
    master_idx_mapper = get_rev_orientation_idx_mapper(top_df, n, n_strands)
    top_df = top_df.iloc[top_df.index.map(master_idx_mapper).argsort()].reset_index(drop=True)
    top_df.replace({"5p_nbr": master_idx_mapper, "3p_nbr": master_idx_mapper}, inplace=True)
    cols_reordered = ["strand", "base", "5p_nbr", "3p_nbr"]
    top_df = top_df.reindex(columns=cols_reordered)
    top_info = (top_df, n, n_strands)


    if traj_lines_3to5 is not None:

        traj_df, ts, bs, Es = _read_traj_info(traj_lines_3to5, n)

        # Reorder each timestep using a master_idx_mapper that we populate during above loop and pd.argsort
        # Note: https://stackoverflow.com/questions/61355655/pandas-how-to-sort-rows-of-a-column-using-a-dictionary-with-indexes
        for t in traj_df['t'].unique():
            t_df = traj_df[traj_df.t == t]
            t_df_resorted = t_df.iloc[t_df.index.map(master_idx_mapper).argsort()].reset_index(drop=True)
            traj_df.loc[traj_df.t == t] = t_df_resorted.values


        # Then, flip all base normals by 180 -- just take the negative?
        pdb.set_trace()
        for a3_col in ['a3_x', 'a3_y', 'a3_z']:
            traj_df[a3_col] = -traj_df[a3_col]

        traj_info = (traj_df, ts, bs, Es)

    return top_info, traj_info


# Reads a topology file, and optionally a trajectory file, in 3'->5' format and returns unprocessed information in 5'->3' format
def read_3to5(top_path, traj_path=None):
    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")
    if traj_path is not None and not Path(traj_path).exists():
        raise RuntimeError(f"Trajectory file does not exist at location: {traj_path}")
    with open(top_path) as f:
        top_lines_3to5 = f.readlines()

    traj_lines_3to5 = None
    if traj_path is not None:
        with open(traj_path) as f:
            traj_lines_3to5 = f.readlines()

    top_info, traj_info = _read_3to5(top_lines_3to5, traj_lines_3to5)
    return top_info, traj_info

# Reads a topology file, and optionally a trajectory file, in 5'->3' format and returns unprocessed information in 5'->3' format
def read_5to3(top_path, traj_path=None):
    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")
    if traj_path is not None and not Path(traj_path).exists():
        raise RuntimeError(f"Trajectory file does not exist at location: {traj_path}")
    with open(top_path) as f:
        top_lines = f.readlines()

    sys_info = top_lines[0].strip().split()
    n = int(sys_info[0])
    n_strands = int(sys_info[1])
    top_df = pd.read_csv(StringIO('\n'.join(top_lines[1:])),
                         names=["strand", "base", "5p_nbr", "3p_nbr"],
                         delim_whitespace=True)

    top_info = (top_df, n, n_strands)

    traj_info = None
    if traj_path is not None:
        with open(traj_path) as f:
            traj_lines = f.readlines()

        traj_df, ts, bs, Es = _read_traj_info(traj_lines, n)
        traj_info = (traj_df, ts, bs, Es)

    return top_info, traj_info

# Takes unprocessed topology information in the 5'->3' format and processes it for simulations
# Requires that we are in 5'->3'
def _process_topology_5to3(top_df, n, n_strands):
    bonded_nbrs = list()

    for i, nuc_row in top_df.iterrows():
        nbr_5p = int(nuc_row['5p_nbr'])
        nbr_3p = int(nuc_row['3p_nbr'])

        if nuc_row.base not in DNA_BASES:
            raise RuntimeError(f"Invalid base: {nuc_row.base}")

        if nbr_3p != -1:
            if not i < nbr_3p:
                # Note: need this for OrderedSparse
                raise RuntimeError(f"Nucleotides must be ordered such that i < j where j is 3' of i and i and j are on the same strand") # Note: circular strands wouldn't obey this
            bonded_nbrs.append((i, nbr_3p)) # 5'->3'

    seq = ''.join(top_df.base.tolist())
    unbonded_nbrs = get_unbonded_neighbors(n, bonded_nbrs)

    return bonded_nbrs, unbonded_nbrs, seq



# Can operate at either 3'->5' or 5'->3'
# set `reverse=True` if the topology file is 3'->5' instead of 5'->'3
# FIXME: need more error checking
def read_topology(top_path, reverse):
    with open(top_path) as f:
        top_lines = f.readlines()

    if reverse:
        (top_df, n, n_strands), _ = read_3to5(top_path, traj_path=None)
    else:
        (top_df, n, n_strands), _ = read_5to3(top_path, traj_path=None)

    # Now, all information is in 5'->3'
    bonded_nbrs, unbonded_nbrs, seq = _process_topology_5to3(top_df, n, n_strands)
    return n, n_strands, bonded_nbrs, unbonded_nbrs, seq


# set `reverse=True` if the topology file and trajectory files are 3'->5' instead of 5'->'3
# Note: Could maybe subsume read_config... but then everything would be a list
# FIXME: note that we shouldn't have separate logic for read_config. Just calls read_traj and returns its first elements. Deleted read_config and have to implement this
def read_trajectory(traj_path, top_path, reverse):
    if not Path(traj_path).exists():
        raise RuntimeError(f"Trajectory file does not exist at location: {traj_path}")
    if not Path(top_path).exists():
        raise RuntimeError(f"Topology file does not exist at location: {top_path}")

    if reverse:
        top_info, traj_info = read_3to5(top_path, traj_path)
    else:
        top_info, traj_info = read_5to3(top_path, traj_path)

    (top_df, n, n_strands) = top_info
    (traj_df, ts, bs, Es) = traj_info

    bonded_nbrs, unbonded_nbrs, seq = _process_topology_5to3(top_df, n, n_strands)

    states = list()
    for t in ts:
        state_df = traj_df[traj_df['t'] == t]
        state = read_state(state_df) # FIXME: need to implement this
        states.append(state)

    return states, bs, ts, Es, n_strands, bonded_nbrs, unbonded_nbrs, seq


# Transform quaternions to nucleotide orientations

## backbone-bsae orientation
def q_to_back_base(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        q0**2 + q1**2 - q2**2 - q3**2,
        2*(q1*q2 + q0*q3),
        2*(q1*q3 - q0*q2)
    ])
Q_to_back_base = vmap(q_to_back_base) # Q is system of quaternions, q is an individual quaternion

## normal orientation
def q_to_base_normal(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q3 + q0*q2),
        2*(q2*q3 - q0*q1),
        q0**2 - q1**2 - q2**2 + q3**2
    ])
Q_to_base_normal = vmap(q_to_base_normal)

## third axis (n x b)
def q_to_cross_prod(q):
    q0, q1, q2, q3 = q.vec
    return jnp.array([
        2*(q1*q2 - q0*q3),
        q0**2 - q1**2 + q2**2 - q3**2,
        2*(q2*q3 + q0*q1)
    ])
Q_to_cross_prod = vmap(q_to_cross_prod)


# FIXME: should really take temperature as input (or kT)
# FIXME: so, this means that thing rely on kT should really only be intermediate values in the *.toml file and they should be updaed to the full (whose name currently exists in the toml file) herex
# Temperature (t) in Kelvin
def get_params(t, config_path="tom.toml"):
    kt = get_kt(t)

    if not Path(config_path).exists():
        raise RuntimeError(f"No file at location: {config_path}")
    params = toml.load(config_path)

    # Excluded Volume
    exc_vol = params['excluded_volume']

    ## f3(dr_backbone)
    b_backbone, dr_c_backbone = get_f3_smoothing_params(exc_vol['dr_star_backbone'],
                                                        exc_vol['eps_exc'],
                                                        exc_vol['sigma_backbone'])
    params['excluded_volume']['b_backbone'] = b_backbone
    params['excluded_volume']['dr_c_backbone'] = dr_c_backbone

    ## f3(dr_base)
    b_base, dr_c_base = get_f3_smoothing_params(exc_vol['dr_star_base'],
                                                        exc_vol['eps_exc'],
                                                        exc_vol['sigma_base'])
    params['excluded_volume']['b_base'] = b_base
    params['excluded_volume']['dr_c_base'] = dr_c_base

    ## f3(dr_back_base)
    b_back_base, dr_c_back_base = get_f3_smoothing_params(exc_vol['dr_star_back_base'],
                                                          exc_vol['eps_exc'],
                                                          exc_vol['sigma_back_base'])
    params['excluded_volume']['b_back_base'] = b_back_base
    params['excluded_volume']['dr_c_back_base'] = dr_c_back_base

    ## f3(dr_base_back)
    b_base_back, dr_c_base_back = get_f3_smoothing_params(exc_vol['dr_star_base_back'],
                                                          exc_vol['eps_exc'],
                                                          exc_vol['sigma_base_back'])
    params['excluded_volume']['b_base_back'] = b_base_back
    params['excluded_volume']['dr_c_base_back'] = dr_c_base_back


    # Stacking
    params['stacking']['eps_stack'] = params['stacking']['eps_stack_base'] + params['stacking']['eps_stack_kt_coeff'] * kt # Do this quickly so that it's included in `stacking`
    stacking = params['stacking']

    ## f1(dr_stack)
    b_backbone, dr_c_backbone = get_f3_smoothing_params(exc_vol['dr_star_backbone'],
                                                        exc_vol['eps_exc'],
                                                        exc_vol['sigma_backbone'])
    b_low_stack, dr_c_low_stack, b_high_stack, dr_c_high_stack = get_f1_smoothing_params(
        stacking['eps_stack'], stacking['dr0_stack'], stacking['a_stack'], stacking['dr_c_stack'],
        stacking['dr_low_stack'], stacking['dr_high_stack'])
    params['stacking']['b_low_stack'] = b_low_stack
    params['stacking']['dr_c_low_stack'] = dr_c_low_stack
    params['stacking']['b_high_stack'] = b_high_stack
    params['stacking']['dr_c_high_stack'] = dr_c_high_stack

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(stacking['a_stack_4'],
                                                         stacking['theta0_stack_4'],
                                                         stacking['delta_theta_star_stack_4'])
    params['stacking']['b_theta_4'] = b_theta_4
    params['stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_5p)
    b_theta_5, delta_theta_5_c = get_f4_smoothing_params(stacking['a_stack_5'],
                                                         stacking['theta0_stack_5'],
                                                         stacking['delta_theta_star_stack_5'])
    params['stacking']['b_theta_5'] = b_theta_5
    params['stacking']['delta_theta_5_c'] = delta_theta_5_c

    ## f4(theta_6p)
    b_theta_6, delta_theta_6_c = get_f4_smoothing_params(stacking['a_stack_6'],
                                                         stacking['theta0_stack_6'],
                                                         stacking['delta_theta_star_stack_6'])
    params['stacking']['b_theta_6'] = b_theta_6
    params['stacking']['delta_theta_6_c'] = delta_theta_6_c

    ## f5(-cos(phi1))
    b_neg_cos_phi1, neg_cos_phi1_c = get_f5_smoothing_params(stacking['a_stack_1'],
                                                             stacking['neg_cos_phi1_star_stack'])
    params['stacking']['b_neg_cos_phi1'] = b_neg_cos_phi1
    params['stacking']['neg_cos_phi1_c'] = neg_cos_phi1_c

    ## f5(-cos(phi2))
    b_neg_cos_phi2, neg_cos_phi2_c = get_f5_smoothing_params(stacking['a_stack_2'],
                                                             stacking['neg_cos_phi2_star_stack'])
    params['stacking']['b_neg_cos_phi2'] = b_neg_cos_phi2
    params['stacking']['neg_cos_phi2_c'] = neg_cos_phi2_c


    # Hydrogen Bonding
    hb = params['hydrogen_bonding']

    ## f1(dr_hb)
    b_low_hb, dr_c_low_hb, b_high_hb, dr_c_high_hb = get_f1_smoothing_params(
        hb['eps_hb'], hb['dr0_hb'], hb['a_hb'], hb['dr_c_hb'],
        hb['dr_low_hb'], hb['dr_high_hb'])
    params['hydrogen_bonding']['b_low_hb'] = b_low_hb
    params['hydrogen_bonding']['dr_c_low_hb'] = dr_c_low_hb
    params['hydrogen_bonding']['b_high_hb'] = b_high_hb
    params['hydrogen_bonding']['dr_c_high_hb'] = dr_c_high_hb

    ## f4(theta_1)
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(hb['a_hb_1'],
                                                         hb['theta0_hb_1'],
                                                         hb['delta_theta_star_hb_1'])
    params['hydrogen_bonding']['b_theta_1'] = b_theta_1
    params['hydrogen_bonding']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_2)
    b_theta_2, delta_theta_2_c = get_f4_smoothing_params(hb['a_hb_2'],
                                                         hb['theta0_hb_2'],
                                                         hb['delta_theta_star_hb_2'])
    params['hydrogen_bonding']['b_theta_2'] = b_theta_2
    params['hydrogen_bonding']['delta_theta_2_c'] = delta_theta_2_c

    ## f4(theta_3)
    b_theta_3, delta_theta_3_c = get_f4_smoothing_params(hb['a_hb_3'],
                                                         hb['theta0_hb_3'],
                                                         hb['delta_theta_star_hb_3'])
    params['hydrogen_bonding']['b_theta_3'] = b_theta_3
    params['hydrogen_bonding']['delta_theta_3_c'] = delta_theta_3_c

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(hb['a_hb_4'],
                                                         hb['theta0_hb_4'],
                                                         hb['delta_theta_star_hb_4'])
    params['hydrogen_bonding']['b_theta_4'] = b_theta_4
    params['hydrogen_bonding']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_7)
    b_theta_7, delta_theta_7_c = get_f4_smoothing_params(hb['a_hb_7'],
                                                         hb['theta0_hb_7'],
                                                         hb['delta_theta_star_hb_7'])
    params['hydrogen_bonding']['b_theta_7'] = b_theta_7
    params['hydrogen_bonding']['delta_theta_7_c'] = delta_theta_7_c

    ## f4(theta_8)
    b_theta_8, delta_theta_8_c = get_f4_smoothing_params(hb['a_hb_8'],
                                                         hb['theta0_hb_8'],
                                                         hb['delta_theta_star_hb_8'])
    params['hydrogen_bonding']['b_theta_8'] = b_theta_8
    params['hydrogen_bonding']['delta_theta_8_c'] = delta_theta_8_c


    # Cross Stacking
    cross = params['cross_stacking']

    ## f2(dr_hb) (FIXME: is the _hb a typo? Should be _cross_stack, no?)
    b_low_cross, dr_c_low_cross, b_high_cross, dr_c_high_cross = get_f2_smoothing_params(
        cross['k'],
        cross['r0_cross'],
        cross['dr_c_cross'],
        cross['dr_low_cross'],
        cross['dr_high_cross'])
    params['cross_stacking']['b_low_cross'] = b_low_cross
    params['cross_stacking']['dr_c_low_cross'] = dr_c_low_cross
    params['cross_stacking']['b_high_cross'] = b_high_cross
    params['cross_stacking']['dr_c_high_cross'] = dr_c_high_cross

    ## f4(theta_1)
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(cross['a_cross_1'],
                                                         cross['theta0_cross_1'],
                                                         cross['delta_theta_star_cross_1'])
    params['cross_stacking']['b_theta_1'] = b_theta_1
    params['cross_stacking']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_2)
    b_theta_2, delta_theta_2_c = get_f4_smoothing_params(cross['a_cross_2'],
                                                         cross['theta0_cross_2'],
                                                         cross['delta_theta_star_cross_2'])
    params['cross_stacking']['b_theta_2'] = b_theta_2
    params['cross_stacking']['delta_theta_2_c'] = delta_theta_2_c

    ## f4(theta_3)
    b_theta_3, delta_theta_3_c = get_f4_smoothing_params(cross['a_cross_3'],
                                                         cross['theta0_cross_3'],
                                                         cross['delta_theta_star_cross_3'])
    params['cross_stacking']['b_theta_3'] = b_theta_3
    params['cross_stacking']['delta_theta_3_c'] = delta_theta_3_c

    ## f4(theta_4) + f4(pi - theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(cross['a_cross_4'],
                                                         cross['theta0_cross_4'],
                                                         cross['delta_theta_star_cross_4'])
    params['cross_stacking']['b_theta_4'] = b_theta_4
    params['cross_stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_7) + f4(pi - theta_7)
    b_theta_7, delta_theta_7_c = get_f4_smoothing_params(cross['a_cross_7'],
                                                         cross['theta0_cross_7'],
                                                         cross['delta_theta_star_cross_7'])
    params['cross_stacking']['b_theta_7'] = b_theta_7
    params['cross_stacking']['delta_theta_7_c'] = delta_theta_7_c

    ## f4(theta_8) + f4(pi - theta_8)
    b_theta_8, delta_theta_8_c = get_f4_smoothing_params(cross['a_cross_8'],
                                                         cross['theta0_cross_8'],
                                                         cross['delta_theta_star_cross_8'])
    params['cross_stacking']['b_theta_8'] = b_theta_8
    params['cross_stacking']['delta_theta_8_c'] = delta_theta_8_c


    # Coaxial Stacking
    coax = params['coaxial_stacking']

    ## f2(dr_coax)
    b_low_coax, dr_c_low_coax, b_high_coax, dr_c_high_coax = get_f2_smoothing_params(
        coax['k_coax'],
        coax['dr0_coax'],
        coax['dr_c_coax'],
        coax['dr_low_coax'],
        coax['dr_high_coax'])
    params['coaxial_stacking']['b_low_coax'] = b_low_coax
    params['coaxial_stacking']['dr_c_low_coax'] = dr_c_low_coax
    params['coaxial_stacking']['b_high_coax'] = b_high_coax
    params['coaxial_stacking']['dr_c_high_coax'] = dr_c_high_coax

    ## f4(theta_1) + f4(2*pi - theta_1)
    b_theta_1, delta_theta_1_c = get_f4_smoothing_params(coax['a_coax_1'],
                                                         coax['theta0_coax_1'],
                                                         coax['delta_theta_star_coax_1'])
    params['coaxial_stacking']['b_theta_1'] = b_theta_1
    params['coaxial_stacking']['delta_theta_1_c'] = delta_theta_1_c

    ## f4(theta_4)
    b_theta_4, delta_theta_4_c = get_f4_smoothing_params(coax['a_coax_4'],
                                                         coax['theta0_coax_4'],
                                                         coax['delta_theta_star_coax_4'])
    params['coaxial_stacking']['b_theta_4'] = b_theta_4
    params['coaxial_stacking']['delta_theta_4_c'] = delta_theta_4_c

    ## f4(theta_5) + f4(pi - theta_5)
    b_theta_5, delta_theta_5_c = get_f4_smoothing_params(coax['a_coax_5'],
                                                         coax['theta0_coax_5'],
                                                         coax['delta_theta_star_coax_5'])
    params['coaxial_stacking']['b_theta_5'] = b_theta_5
    params['coaxial_stacking']['delta_theta_5_c'] = delta_theta_5_c

    ## f4(theta_6) + f4(pi - theta_6)
    b_theta_6, delta_theta_6_c = get_f4_smoothing_params(coax['a_coax_6'],
                                                         coax['theta0_coax_6'],
                                                         coax['delta_theta_star_coax_6'])
    params['coaxial_stacking']['b_theta_6'] = b_theta_6
    params['coaxial_stacking']['delta_theta_6_c'] = delta_theta_6_c

    ## f5(cos(phi3))
    b_cos_phi3, cos_phi3_c = get_f5_smoothing_params(coax['a_coax_3p'],
                                                     coax['cos_phi3_star_coax'])
    params['coaxial_stacking']['b_cos_phi3'] = b_cos_phi3
    params['coaxial_stacking']['cos_phi3_c'] = cos_phi3_c

    ## f5(cos(phi4))
    b_cos_phi4, cos_phi4_c = get_f5_smoothing_params(coax['a_coax_4p'],
                                                     coax['cos_phi4_star_coax'])
    params['coaxial_stacking']['b_cos_phi4'] = b_cos_phi4
    params['coaxial_stacking']['cos_phi4_c'] = cos_phi4_c

    return params


if __name__ == "__main__":
    # final_params = get_params()

    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"
    conf_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/start.conf"
    traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/output.dat"

    read_trajectory(traj_path, top_path, reverse=True)
    # convert_3to5_to_5to3(top_path, traj_path)


    # body, box_size, n_strands, bonded_nbrs, unbonded_nbrs, seq = read_config(conf_path, top_path)

    pdb.set_trace()

    # jax_traj_to_oxdna_traj([body], box_size[0], output_name="recovered.dat")

    pdb.set_trace()
    print("done")
