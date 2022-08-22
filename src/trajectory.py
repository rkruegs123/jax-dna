from pathlib import Path
import pdb
import pandas as pd
import numpy as np

from jax_md.rigid_body import Quaternion, RigidBody
from utils import Q_to_back_base, Q_to_base_normal



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


# Reverses the orientation of a traj_df (either from 5'->3' to 3'->5' *or* from 3'->5' to 5'->3')
# Note: https://stackoverflow.com/questions/61355655/pandas-how-to-sort-rows-of-a-column-using-a-dictionary-with-indexes
def get_rev_traj_df(traj_df, rev_orientation_mapper):
    rev_traj_df = traj_df.copy(deep=True)
    for t in rev_traj_df['t'].unique():
        t_df = rev_traj_df[rev_traj_df.t == t]
        t_df_resorted = t_df.iloc[t_df.index.map(rev_orientation_mapper).argsort()].reset_index(drop=True)
        rev_traj_df.loc[rev_traj_df.t == t] = t_df_resorted.values

    # Then, flip all base normals by 180 by taking their negative
    for a3_col in ['a3_x', 'a3_y', 'a3_z']:
        rev_traj_df[a3_col] = -rev_traj_df[a3_col]

    return rev_traj_df


# Takes in a list of box_sizes (i.e. one array of shape (3,) for each simulation step) and returns a single scalar value
def bs_to_box_size(bs):
    if not np.all(bs == bs[0]):
            raise RuntimeError(f"Currently only supports trajectories with a fixed box size")
    box_dims = bs[0]
    if not np.all(box_dims == box_dims[0]):
        raise RuntimeError(f"Currently only support cubic simulation boxes (i.e. all dimensions are the same)")

    return box_dims[0]

class TrajectoryInfo:
    """
    Requirements:

    A TrajectoryInfo instance can be constructed in two ways:
      1. Directly from a list of RigidBody objects. In this way, nucleotides are assumed to be given 5'->3'
      2. From a trajectory file and a flag indicating whether the trajectory file is given in the 3'->5' or 5'->3' orientation
    To differentiate between these two options, we follow the style from the following:
      - https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-implement-multiple-constructors
    Note that for now, we do not store the times, box sizes, or energies when we read from file. If we did, we would likely want to require these to also be given when `states` is provided explicitly

    The following is more information about each parameter:
    - states: a list of RigidBody. Must be 5'->3'
    - top_info: an instance of TopologyInfo -- note that it *always* stores 5'->3' info
    - traj_path: path to the trajectory file
    - reverse_direction: a boolean flag indicating whether or not the trajectory stored in traj_path is 3'->5' or 5'->3'
    """
    def __init__(self, top_info,
                 states=None, box_size=None, # Constructor 1
                 traj_path=None, reverse_direction=None # Constructor 2
    ):
        if states is None and traj_path is None:
            raise RuntimeError("One of states and traj_path must be set")
        if states is not None and traj_path is not None:
            raise RuntimeError("Only one of states and traj_path can be set")

        if traj_path is not None and reverse_direction is None:
            raise RuntimeError("If traj_path is set, reverse_direction must also be set")
        if states is not None and box_size is None:
            raise RuntimeError("If states is set, box_size must also be set")

        self.top_info = top_info

        if states is not None:
            # Store the states directly
            self.read_from_states(states, box_size)
        else:
            # Read from file
            self.read_from_file(traj_path, reverse_direction)


    def build_5to3_df(self, traj_lines_5to3):
        # Read in the raw dataframe and trajectory info from the file contents
        traj_df, ts, bs, Es = _read_traj_info(traj_lines_5to3, self.top_info.n)

        # Store all information
        self.traj_df = traj_df
        self.box_size = bs_to_box_size(bs)
        # self.ts = ts
        # self.bs = bs
        # self.Es = Es

    def build_3to5_df(self, traj_lines_3to5):
        # Read in the raw dataframe and trajectory info from the file contents
        traj_df_3to5, ts, bs, Es = _read_traj_info(traj_lines_3to5, self.top_info.n)

        self.box_size = bs_to_box_size(bs)

        # Store the times, box sizes, and energies, as these do not depend on orientation
        # self.ts = ts
        # self.bs = bs
        # self.Es = Es

        # Reorder each timestep using the index mapper from `self.top_info` and flip base normals
        traj_df_5to3 = get_rev_traj_df(traj_df_3to5, self.top_info.rev_orientation_mapper)

        # Store our updated trajectory dataframe
        self.traj_df = traj_df_5to3


    # generates `states` from `traj_df` (assumed to be 5'->3')
    def process_traj_df(self):
        states = list()
        for t in self.traj_df.t.unique(): # better than iterating over self.ts in case of numerical instability
            state_df = self.traj_df[self.traj_df['t'] == t]
            state = read_state(state_df) # FIXME: need to implement this
            states.append(state)

        self.states = states


    # generates `traj_df` from `states` (assumed to be 5'->3')
    def process_states(self):

        traj_df_lines = list()

        # FIXME: for now, we just have `t` increment by 1
        for t, state in enumerate(self.states):
            back_base_vs = Q_to_back_base(state.orientation)
            base_normals = Q_to_base_normal(state.orientation)
            coms = state.center

            for i in range(self.n):
                nuc_at_t_line = [t] + coms[i] + back_base_vs[i] + base_normals[i] \
                                + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] # dummy velocity and momentum for now
                traj_df_lines.append(nuc_at_t_line)

        traj_df = pd.DataFrame(df_lines,
                               columns=["t",
                                        "com_x", "com_y", "com_z",
                                        "a1_x", "a1_y", "a1_z",
                                        "a3_x", "a3_y", "a3_z",
                                        "v_x", "v_y", "v_z",
                                        "L_x", "L_y", "L_z"],
                               dtype=float)
        self.traj_df = traj_df


    def read_from_states(self, states, box_size):
        self.states = states
        self.box_size = box_size
        self.process_states()

    def read_from_file(self, traj_path, reverse_direction):
        # Check that our trajectory file exists
        if not Path(traj_path).exists():
            raise RuntimeError(f"Trajectory file does not exist: {traj_path}")

        self.traj_path = traj_path
        self.reverse_direction = reverse_direction

        # Read the lines from our trajectory file
        with open(self.traj_path) as f:
            traj_lines = f.readlines()

        if self.reverse_direction:
            self.build_3to5_df(traj_lines)
        else:
            self.build_5to3_df(traj_lines)

        # Like in topology.py, once we've constructed our 5'->3' `self.traj_df`, we post-process -- i.e. convert each timestep in the dataframe to a RigidBody
        self.process_traj_df()

    # Write self.traj_df (always 5'->3') to an oxDNA-style trajectory file
    # if reverse=True, write in 3'->5' format
    # if write_topology=True, we also write the topology to file in the orientation determined by `reverse`
    def write(self, traj_opath, reverse, write_topology, top_opath=None):
        # Write the topology file if flag is set
        if write_topology:
            if top_opath is None:
                raise RuntimeError(f"Topology file output path must be set if you want to write a topology file")
            self.top_info.write(top_opath, reverse)

        # Write the trajectory file
        traj_df_to_write = self.traj_df
        if reverse:
            # get a 3'->5' version of self.traj_df
            traj_df_to_write = get_rev_traj_df(self.traj_df, self.top_info.rev_orientation_mapper)

        ts = self.traj_df.t.unique() # Note: dummy values for `ts` if read from states
        # FIXME: Using dummy box_values and energy values for now
        # FIXME: check that size of self.box_size is 1
        bs = np.full((len(ts), 3), fill_value=self.box_size)
        Es = np.full((len(ts), 3), fill_value=0.0)

        out_lines_traj = list()

        for t, b, E in zip(ts, bs, Es):
            t_df = traj_df_to_write[traj_df_to_write.t == t]

            out_lines_traj.append(f"t = {t}")
            out_lines_traj.append(f"b = {' '.join([str(b_dim) for b_dim in b])}")
            out_lines_traj.append(f"E = {' '.join([str(e_val) for e_val in E])}")
            t_lines = t_df.drop('t', axis=1).to_csv(
                header=None, index=False, sep=" ").strip('\n').split('\n')
            out_lines_traj += t_lines

        with open(traj_opath, 'w+') as of_traj:
            of_traj.write('\n'.join(out_lines_traj))



if __name__ == "__main__":
    from topology import TopologyInfo


    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"
    traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/output.dat"


    """
    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.top"
    traj_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/polyA_10bp/generated.dat"
    """

    top_info = TopologyInfo(top_path, reverse_direction=True)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=True)
    pdb.set_trace()
    traj_info.write("test_out.dat", True, True, "top_out.dat")
    print("done")
