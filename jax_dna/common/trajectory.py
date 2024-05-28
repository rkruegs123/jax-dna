# ruff: noqa
# fmt: off
import pdb
import unittest
from pathlib import Path
import pandas as pd
import numpy as onp
from tqdm import tqdm

import jax.numpy as jnp
from jax import jit
from jax_md.rigid_body import RigidBody, Quaternion

from jax_dna.common.utils import Q_to_back_base, Q_to_base_normal, tree_stack
from jax_dna.common.topology import TopologyInfo

from jax.config import config
config.update("jax_enable_x64", True)



class TrajectoryInfo:
    """
    A class for storing trajectory data.

    There are two options for creating an instance of this class:
    1) From a set of states (i.e. an iterator of RigidBody's)
    2) From an input file (in oxDNA format)

    In either case, the primary data structure of a TrajectoryInfo
    instance is `traj_df`, a DataFrame storing the trajectory
    information in 5'->3' orientation. Importantly, after
    construction, `traj_df` is assumed to be in 5'->3' orientation
    This means that when constructing from a set of states, the
    states must follow this convention. When reading from a file,
    set `reverse_direction=True` to specify that the input file
    is represented in 3'->5' orientation.

    Once the trajectory DataFrame is loaded, there are several
    functionalities for working with it. For example:
    - `get_states` convertes `self.traj_df` to a list of RigidBody's
    - `write` writes `self.traj_df` to a trajectory file in
      standard oxDNA format
    """

    def __init__(
            self, top_info,

            # Constructor 1: Read from iterator of RigidBody's (assumed 5'->3')
            read_from_states=False, states=None, box_size=None,

            # Constructor 2: Read from file (3'->5' or 5'->3')
            read_from_file=False, traj_path=None, reverse_direction=None, reindex=False
    ):

        self.top_info = top_info
        self.n = self.top_info.n

        # Error checking
        both_constructors_used = (not read_from_states and not read_from_file)
        neither_constructor_used = (read_from_states and read_from_file)
        if both_constructors_used or neither_constructor_used:
            msg = f"Exactly one of read_from_states or read_from_file must be True"
            raise RuntimeError(msg)

        if read_from_states:
            if states is None or box_size is None:
                msg = f"If reading from states, must set both states and box size"
                raise RuntimeError(msg)

            if (traj_path is not None or reverse_direction is not None):
                msg = f"If reading from state, do not set any flags for reading from file"
                raise RuntimeError(msg)

        if read_from_file:
            if traj_path is None or reverse_direction is None:
                msg = f"If reading from file, must set both traj_path and reverse_direction"
                raise RuntimeError(msg)

            if (states is not None or box_size is not None):
                msg = f"If reading from file, do not set any flags for reading from states"
                raise RuntimeError(msg)

        # Construct 5'->3' traj_df from either states or file
        if read_from_file:
            traj_df, box_size = self.load_from_file(traj_path, reverse_direction, reindex)
        else:
            traj_df, box_size = self.load_from_states(states, box_size)
        self.traj_df = traj_df
        self.box_size = box_size



    def get_rev_traj_df(self, traj_df, rev_orientation_mapper):
        """
        Reverses the orientation of a traj_df (either from 5'->3'
        to 3'->5' *or* from 3'->5' to 5'->3').

        Note: https://stackoverflow.com/questions/61355655/pandas-how-to-sort-rows-of-a-column-using-a-dictionary-with-indexes
        """

        rev_traj_df = traj_df.copy(deep=True)
        n = len(rev_orientation_mapper)
        for t in rev_traj_df['t'].unique():
            t_df = rev_traj_df[rev_traj_df.t == t]
            t_df_resorted = t_df.iloc[(t_df.index % n).map(rev_orientation_mapper).argsort()].reset_index(drop=True)
            rev_traj_df.loc[rev_traj_df.t == t] = t_df_resorted.values

        # Then, flip all base normals by 180 by taking their negative
        for a3_col in ['a3_x', 'a3_y', 'a3_z']:
            rev_traj_df[a3_col] = -rev_traj_df[a3_col]

        return rev_traj_df


    def load_from_file(self, traj_path, reverse_direction, reindex):
        """
        One of two constructors. This constructor accepts a filepath,
        as well as metadata specifying whether or not the file is
        3'->5' instead of 5'->3', and constructs a trajectory DataFrame
        and computes a box size.
        """

        # Check that our trajectory file exists
        if not Path(traj_path).exists():
            raise RuntimeError(f"Trajectory file does not exist: {traj_path}")

        # Construct trajectory dataframe
        with open(traj_path) as f:
            traj_lines = f.readlines()

        assert(len(traj_lines) % (self.n+3) == 0)
        time_steps = int(len(traj_lines) / (self.n+3))
        all_state_lines = [traj_lines[(self.n+3)*t:(self.n+3)*t+(self.n+3)] for t in range(time_steps)]

        df_lines = list()
        bs = list()
        Es = list()
        ts = list()
        idx = 0
        for state_lines in tqdm(all_state_lines, desc="Loading trajectory from file"):
            if reindex:
                t = idx
                idx += 1
            else:
                t = float(state_lines[0].split('=')[1].strip())
            ts.append(t)

            b = state_lines[1].split('=')[1].strip().split(' ')
            b = onp.array(b).astype(onp.float64)
            bs.append(b)

            E = state_lines[2].split('=')[1].strip().split(' ')
            E = onp.array(E).astype(onp.float64)
            Es.append(E)

            t_lines = [[t] + state_info.strip().split() for state_info in state_lines[3:]]
            df_lines += t_lines

        ts = onp.array(ts, dtype=onp.float64)
        bs = onp.array(bs, dtype=onp.float64)
        Es = onp.array(Es, dtype=onp.float64)
        col_names = [
            "t", "com_x", "com_y", "com_z", "a1_x", "a1_y", "a1_z", "a3_x",
            "a3_y", "a3_z", "v_x", "v_y", "v_z", "L_x", "L_y", "L_z"
        ]
        traj_df = pd.DataFrame(df_lines, columns=col_names, dtype=float)

        # Reverse the orientation if necessary
        if reverse_direction:
            traj_df = self.get_rev_traj_df(traj_df, self.top_info.rev_orientation_mapper)

        # Retrieve the box size
        if not onp.all(bs == bs[0]):
            raise RuntimeError(f"Currently only supports trajectories with a fixed box size")

        box_dims = bs[0]
        if not onp.all(box_dims == box_dims[0]):
            raise RuntimeError(f"Currently only support cubic simulation boxes (i.e. all dimensions are the same)")
        box_size = box_dims[0]

        return traj_df, box_size


    def load_from_states(self, states, box_size):
        """
        One of two constructors. This constructor accepts a set
        of states (i.e. an iterator of RigidBody's) and a box size
        and constructs a trajectory DataFrame. It also returns the
        box_size to keep a standard API for constructors.
        """

        traj_df_lines = list()

        if isinstance(states, list):
            states = tree_stack(states)
        assert(isinstance(states, RigidBody))
        assert(len(states.center.shape) == 3)
        n_states = states.center.shape[0]

        assert(len(states[0].center.shape) == 2)
        assert(states[0].center.shape[0] == self.n and states[0].center.shape[1] == 3)

        @jit
        def get_state_vecs(state):
            back_base_vs = Q_to_back_base(state.orientation)
            base_normals = Q_to_base_normal(state.orientation)
            return back_base_vs, base_normals

        # note: for now, we just have `t` increment by 1
        for t in tqdm(range(n_states), desc="Loading trajectory from states"):
            state = states[t]
            back_base_vs, base_normals = get_state_vecs(state)
            coms = state.center

            for i in range(self.n):
                nuc_at_t_line = [t] + coms[i].tolist() + back_base_vs[i].tolist() + base_normals[i].tolist() \
                                + [0.0, 0.0, 0.0] + [0.0, 0.0, 0.0] # dummy velocity and momentum
                traj_df_lines.append(nuc_at_t_line)

        col_names = [
            "t", "com_x", "com_y", "com_z", "a1_x", "a1_y", "a1_z",
            "a3_x", "a3_y", "a3_z", "v_x", "v_y", "v_z", "L_x", "L_y", "L_z"
        ]
        traj_df = pd.DataFrame(traj_df_lines,
                               columns=col_names,
                               dtype=float)

        return traj_df, box_size


    def principal_axes_to_euler_angles(self, x, y, z):
        """
        A utility function for converting a set of principal axes
        (that define a rotation matrix) to a commonly used set of
        Tait-Bryan Euler angles.

        There are two options to compute the Tait-Bryan angles. Each can be seen at the respective links:
        (1) From wikipedia (under Tait-Bryan angles): https://en.wikipedia.org/wiki/Euler_angles
        (2) Equation 10A-C: https://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html

        However, note that the definition from Wikipedia (i.e. the one using arcsin) has numerical stability issues,
        so we use the definition from (2) (i.e. the one using arctan2)

        Note that if we were following (1), we would do:
        psi = onp.arcsin(x[1] / onp.sqrt(1 - x[2]**2))
        theta = onp.arcsin(-x[2])
        phi = onp.arcsin(y[2] / onp.sqrt(1 - x[2]**2))

        Note that Tait-Bryan (i.e. Cardan) angles are *not* proper euler angles
        """

        psi = onp.arctan2(x[1], x[0])
        if onp.abs(x[2]) > 1:
            # FIXME: could clamp?
            # pdb.set_trace()
            x[2] = onp.round(x[2])
        theta = onp.arcsin(-x[2])
        phi = onp.arctan2(y[2], z[2])

        return psi, theta, phi

    def euler_angles_to_quaternion(self, t1, t2, t3):
        """
        A utility function for converting euler angles to quaternions.
        Used when converting a trajectory DataFrame to a set of states.

        We follow the ZYX convention. For details, see page A-11 in
        https://ntrs.nasa.gov/api/citations/19770024290/downloads/19770024290.pdf
        from the following set of documentation:
        https://ntrs.nasa.gov/citations/19770024290
        """
        q0 = onp.sin(0.5*t1)*onp.sin(0.5*t2)*onp.sin(0.5*t3) + onp.cos(0.5*t1)*onp.cos(0.5*t2)*onp.cos(0.5*t3)
        q1 = -onp.sin(0.5*t1)*onp.sin(0.5*t2)*onp.cos(0.5*t3) + onp.sin(0.5*t3)*onp.cos(0.5*t1)*onp.cos(0.5*t2)
        q2 = onp.sin(0.5*t1)*onp.sin(0.5*t3)*onp.cos(0.5*t2) + onp.sin(0.5*t2)*onp.cos(0.5*t1)*onp.cos(0.5*t3)
        q3 = onp.sin(0.5*t1)*onp.cos(0.5*t2)*onp.cos(0.5*t3) - onp.sin(0.5*t2)*onp.sin(0.5*t3)*onp.cos(0.5*t1)
        return q0, q1, q2, q3

    def state_df_to_state(self, state_df):
        """
        Takes in a DataFrame defining a single state and returns
        a RigidBody. This is a utility function for converting
        a trajectory DataFrame to a list of RigidBody's

        Note: it is the burden of the user of this function to pass
        the right number of lines. In other words, this function
        should be able to infer `n`
        """

        n = state_df.shape[0]
        assert(n == self.n)
        R = onp.empty((n, 3), dtype=onp.float64)
        quat = onp.empty((n, 4), dtype=onp.float64)

        for i, (idx, nuc_line) in enumerate(state_df.iterrows()):
            nuc_info = nuc_line.tolist()
            assert(len(nuc_info) == 16)
            nuc_info = nuc_info[1:] # remove time

            com = nuc_info[:3]
            back_base_vector = nuc_info[3:6] # back_base
            base_normal = nuc_info[6:9] # base_norm
            velocity = nuc_info[9:12]
            angular_velocity = nuc_info[12:15]

            alpha, beta, gamma = self.principal_axes_to_euler_angles(
                back_base_vector,
                onp.cross(base_normal, back_base_vector),
                base_normal)

            q0, q1, q2, q3 = self.euler_angles_to_quaternion(alpha, beta, gamma)

            R[i, :] = com
            quat[i, :] = onp.array([q0, q1, q2, q3])

        R = jnp.array(R, dtype=jnp.float64)
        quat = jnp.array(quat, dtype=jnp.float64)
        body = RigidBody(R, Quaternion(quat))
        return body

    def get_states(self):
        """
        A utility function for converting the trajectory
        DataFrame to an iterator of RigidBody's

        This is necessary for retrieving, e.g., an initial state
        from a file
        """
        states = list()
        ts = list(self.traj_df.t.unique())
        for t in tqdm(ts, desc="Retrieving states"):
            state_df = self.traj_df[self.traj_df['t'] == t]
            state = self.state_df_to_state(state_df)
            states.append(state)
        return states

    def write(self, traj_opath, reverse, write_topology=False, top_opath=None):
        """
        Write the trajectory to file.

        Since self.traj_df is always 5'->3', by default it
        is written in this orientation. Set reverse=True to
        write 3'->5'.

        Optionally, the topology file can be written by setting
        write_topology=True and specifying an output file
        with top_opath. Since TopologyInfo also stores information
        in 5'->3' orientation, the orientation of topology
        writing is the same as is set by the `reverse` flag
        """

        # Write the topology file if flag is set
        if write_topology:
            if top_opath is None:
                raise RuntimeError(f"Topology file output path must be set if you want to write a topology file")
            self.top_info.write(top_opath, reverse)

        # Write the trajectory file
        traj_df_to_write = self.traj_df
        if reverse:
            # get a 3'->5' version of self.traj_df
            traj_df_to_write = self.get_rev_traj_df(self.traj_df, self.top_info.rev_orientation_mapper)

        ts = self.traj_df.t.unique() # Note: dummy values for `ts` if read from states
        bs = onp.full((len(ts), 3), fill_value=self.box_size) # dummy box sizes
        Es = onp.full((len(ts), 3), fill_value=0.0) # dummy energy values

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


class TestTrajectory(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_init(self):
        top_path = self.test_data_basedir / "simple-helix" / "generated.top"
        top_info = TopologyInfo(top_path, reverse_direction=True)

        conf_path = self.test_data_basedir / "simple-helix" / "start.conf"

        conf_info = TrajectoryInfo(
            top_info,
            read_from_file=True, traj_path=conf_path, reverse_direction=True
        )


if __name__ == "__main__":
    unittest.main()
