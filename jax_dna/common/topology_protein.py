from io import StringIO
import numpy as onp
import pandas as pd
from pathlib import Path
import pdb
import unittest
from itertools import combinations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md.partition import NeighborListFormat, neighbor_list

from jax_dna.common.utils import bcolors, DNA_ALPHA, RNA_ALPHA


RES_ALPHA = "MGKTRADEYVLQWFSHNPCI"


class ProteinTopology:
    def __init__(self, top_path, par_path):
        self.top_path = Path(top_path)
        self.par_path = Path(par_path)

        self.load()

    def load(self):

        # Check that our topology file exists
        if not Path(self.top_path).exists():
            raise RuntimeError(f"Topology file does not exist: {self.top_path}")

        # Check that our parameter file exists
        if not Path(self.par_path).exists():
            raise RuntimeError(f"Parameter file does not exist: {self.par_path}")

        # Read the lines from our topology file
        with open(self.top_path) as f:
            top_lines = f.readlines()

        # Read the information from the first line -- # of particles (nucleotides + amino acids) and # of strands
        sys_info = top_lines[0].strip().split()
        assert(len(sys_info) == 2)
        self.n = int(sys_info[0])
        self.n_strands = int(sys_info[1])

        # Read the input file into a dataframe, regardless of orientation
        res_types = list()
        network = list()
        for curr_idx, line in enumerate(top_lines[1:]):
            tokens = line.strip().split()
            strand_idx = int(tokens[0])
            assert(strand_idx < 0)
            res_type = tokens[1]
            assert(res_type in set(RES_ALPHA))
            res_types.append(res_type)
            n_term_nbr = int(tokens[2]) # do nothing with this
            c_term_nbr = int(tokens[3])
            extra_nbrs = [int(idx) for idx in tokens[4:]]
            # res_nbrs = [int(idx) for idx in tokens[2:] if idx != "-1"]
            # for nbr_idx in res_nbrs:
            for nbr_idx in [c_term_nbr] + extra_nbrs:
                if nbr_idx != -1:
                    network.append((curr_idx, nbr_idx))
        network_set = set(network)
        self.network = onp.array(network)
        self.seq_idx = onp.array([RES_ALPHA.index(res) for res in res_types], dtype=onp.int32)

        # Read the parameter file
        with open(self.par_path) as f:
            par_lines = f.readlines()

        sys_info_par = par_lines[0].strip().split()
        assert(len(sys_info_par) == 1)
        assert(int(sys_info[0]) == self.n)

        spring_constants = onp.zeros((self.n, self.n)).astype(onp.float64)
        eq_distances = onp.zeros((self.n, self.n)).astype(onp.float64)
        for par_line in par_lines[1:]:
            tokens = par_line.strip().split()
            assert(len(tokens) == 5)
            idx1, idx2 = tokens[:2]
            idx1 = int(idx1)
            idx2 = int(idx2)
            assert((idx1, idx2) in network_set)
            bond_type = tokens[3]
            assert(bond_type == "s")

            eq_distance = float(tokens[2])
            spring_constant = float(tokens[4])

            eq_distances[idx1, idx2] = eq_distance
            spring_constants[idx1, idx2] = spring_constant

        self.eq_distances = eq_distances
        self.spring_constants = spring_constants







if __name__ == "__main__":
    test_data_basedir = Path("data/test-data")

    top_path = test_data_basedir / "protein-top" / "KDPG" / "kdpg.top"
    par_path = test_data_basedir / "protein-top" / "KDPG" / "kdpg.par"
    top_info = ProteinTopology(top_path, par_path)
