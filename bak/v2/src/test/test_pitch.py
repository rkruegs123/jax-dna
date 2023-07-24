import pandas as pd
import pdb
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from loss.pitch import get_pitches
from loader.trajectory import TrajectoryInfo
from loader.topology import TopologyInfo
from loader.get_params import get_default_params
from utils import bcolors


def run():
    basedir = Path("v2/data/test-data/simple-helix/")
    top_path = basedir / "generated.top"
    traj_path = basedir / "output.dat"
    pitch_path = basedir / "pitch.dat"

    print(f"----Checking pitch agreement for trajectory at location: {traj_path}----")

    quartets = jnp.array([[1, 14, 2, 13], [2, 13, 3, 12], [3, 12, 4, 11],
                          [4, 11, 5, 10], [5, 10, 6, 9], [6, 9, 7, 8]])

    oxdna_pitches = pd.read_csv(pitch_path,
                                names=["q1", "q2", "q3",
                                       "q4", "q5", "q6"],
                                delim_whitespace=True)
    oxdna_pitches = oxdna_pitches.iloc[1: , :] # drop first line from oxdna_pitches

    # Note for simplicity we do not read in the reverse direction so that we don't have to reverse the quartet indices
    top_info = TopologyInfo(top_path, reverse_direction=False)
    traj_info = TrajectoryInfo(top_info, traj_path=traj_path, reverse_direction=False)

    computed_pitches = [get_pitches(body, quartets) for body in traj_info.states]

    for i, (idx, row) in enumerate(oxdna_pitches.iterrows()):
        print(f"\tState {i}:")
        ith_oxdna_pitches = row.to_numpy()
        ith_computed_pitches = computed_pitches[i]
        print(f"\t\tComputed pitches: {ith_computed_pitches}")
        print(f"\t\toxDNA pitches: {ith_oxdna_pitches}")
        if not np.allclose(ith_oxdna_pitches, ith_computed_pitches, atol=1e-4, rtol=1e-8):
            print(bcolors.FAIL + "\t\tFail!\n" + bcolors.ENDC)
            pdb.set_trace()
        else:
            print(bcolors.OKGREEN + "\t\tSuccess!\n" + bcolors.ENDC)
