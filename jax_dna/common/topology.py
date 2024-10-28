from io import StringIO
import numpy as onp
import pandas as pd
from pathlib import Path
import pdb
import unittest
from itertools import combinations

import jax
import jax.numpy as jnp
from jax_md.partition import NeighborListFormat, neighbor_list

from jax_dna.common.utils import bcolors, DNA_ALPHA, RNA_ALPHA

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)



def get_unbonded_neighbors(n, bonded_neighbors):
    """
    Takes a set of bonded neighbors and returns the set
    of unbonded neighbors for a given `n` by enumerating
    all possibilities of pairings for `n` and removing
    the bonded neighbors
    """

    # First, set to all neighbors
    unbonded_neighbors = set(combinations(range(n), 2))

    # Then, remove all bonded neighbors
    unbonded_neighbors -= set(bonded_neighbors)

    # Finally, remove identities (which shouldn't be in there in the first place)
    unbonded_neighbors -= set([(i, i) for i in range(n)])

    # Return as a list
    return list(unbonded_neighbors) # return as list

def get_rev_orientation_idx_mapper(top_df, n, n_strands):
    """
    Constructs a dictionary that maps base indices to new base
    indices in a way that conserves the set of base identities
    occuring on each strand but reverses their intra-strand order

    Note that, explicitly, the values in the dictionary correspond
    to rows in a topology DataFrame. This mapping is used to
    reorder the rows of such a DataFrame. So, we require that
    base identities are numbered in the same way that DataFrame
    rows are -- 0-indexed, and increasing by 1.

    Also note that this function will return identical dictionaries
    for a top_df in the 5'->3' orientation as well as for its reverse
    (i.e. the corresponding top_df in the 3'->5' orientation, in
    which each strand is reversed).
    """

    master_idx_mapper = dict()

    for strand in range(1, n_strands + 1): # assumes strands are 1-indexed
        # Get the indices
        index_orig = top_df[top_df.strand == strand].index
        index_rev = index_orig[::-1]

        strand_idx_mapper = dict(zip(index_orig, index_rev))
        master_idx_mapper.update(strand_idx_mapper)
    return master_idx_mapper


def get_rev_top_df(top_df, rev_orientation_mapper):
    """
    Reverses the orientation of a top_df (either from 5'->3'
    to 3'->5' *or* from 3'->5' to 5'->3')
    """

    rev_top_df = top_df.copy(deep=True)

    rev_top_df = rev_top_df.iloc[rev_top_df.index.map(rev_orientation_mapper).argsort()].reset_index(drop=True)

    # Update the values of neighbors appropriately
    rev_top_df.replace({"5p_nbr": rev_orientation_mapper, "3p_nbr": rev_orientation_mapper}, inplace=True)

    # Swap the order of the 3p and 5p neighbor columns
    cols = list(top_df.columns.values)
    cols_reordered = ["strand", "base", cols[3], cols[2]]

    rev_top_df = rev_top_df.reindex(columns=cols_reordered)
    return rev_top_df


def check_valid_top_df(top_df, n_strands, n, alphabet=DNA_ALPHA, verbose=False):
    """
    Checks that the given topology DataFrame is valid, irrespective
    of the direction (i.e. 3'->5' or 5'->3')
    """

    # Check for valid bases
    for i, nuc_row in top_df.iterrows():
        if nuc_row.base not in set(alphabet):
            raise RuntimeError(f"Invalid base at position {i}: {nuc_row.base}")

    # Check that top_df strands are 1-indexed and increase by 1
    if not set(top_df.strand.unique()) == set(onp.arange(1, n_strands+1)):
        raise RuntimeError(f"Strand numbers must be 1-indexed and increase by 1")

    # Check that the base identities are 0-indexed and increase by 1
    base_identities = set(top_df["3p_nbr"].unique()) | set(top_df["5p_nbr"].unique())
    if -1 in base_identities:
        base_identities.remove(-1)
    # note: this is an approximation, and wouldn't capture unbonded bases
    if not base_identities == set(onp.arange(n)):
        raise RuntimeError(f"Base identities must be 0-indexed and increase by 1")

    # Print a warning for properties we can't check
    if verbose:
        print(f"{bcolors.WARNING}Unchecked requirement: row indices must correspond to base identity in topology file. E.g. base 0 must be on the first line (line 0), and so on{bcolors.ENDC}")


class TopologyInfo:
    """
    A class to represent an oxDNA topology.

    Importantly, the topology is *always* stored in the 5'->3'
    direction. However, files can be read from either the
    3'->5' orientation or the 5'->3' orientation. This is to
    support legacy oxDNA files, which were traditionally
    represented 3'->5'.

    Specify the direction with the `reverse_direction` flag:
    True if input file is 3'->5', False otherwise (5'->3').
    """
    def __init__(self, top_path, reverse_direction, is_rna=False, allow_circle=False):
        self.top_path = Path(top_path)
        self.reverse_direction = reverse_direction
        self.is_rna = is_rna
        self.allow_circle = allow_circle
        if self.is_rna:
            self.alphabet = RNA_ALPHA
        else:
            self.alphabet = DNA_ALPHA

        self.load()

    def load(self):
        """
        Read in topology information from initialized values.
        Regardless of the input orientation (i.e. 3'->5' or
        5'->3'), data is always stored in 5'->3' format. This
        function is responsible for ensuring this.
        """

        # Check that our topology file exists
        if not Path(self.top_path).exists():
            raise RuntimeError(f"Topology file does not exist: {self.top_path}")

        # Read the lines from our topology file
        with open(self.top_path) as f:
            top_lines = f.readlines()

        # Read the information from the first line -- # of nucleotides and # of strands
        sys_info = top_lines[0].strip().split()
        assert(len(sys_info) == 2)
        self.n = int(sys_info[0])
        self.n_strands = int(sys_info[1])

        # Read the input file into a dataframe, regardless of orientation
        if self.reverse_direction:
            input_col_names = ["strand", "base", "3p_nbr", "5p_nbr"]
        else:
            input_col_names = ["strand", "base", "5p_nbr", "3p_nbr"]
        top_df = pd.read_csv(
            StringIO('\n'.join(top_lines[1:])),
            names=input_col_names,
            delim_whitespace=True)

        check_valid_top_df(top_df, self.n_strands, self.n, self.alphabet)

        # Construct a dictionary to reverse the orientation
        rev_orientation_mapper = get_rev_orientation_idx_mapper(top_df, self.n, self.n_strands)
        self.rev_orientation_mapper = rev_orientation_mapper

        # Fix the orientation if necessary, and save df
        if self.reverse_direction:
            self.top_df = get_rev_top_df(top_df, self.rev_orientation_mapper)
        else:
            self.top_df = top_df

        # Once we've constructed our 5'->3' `self.top_df`, infer the neighbors and sequence
        bonded_nbrs = list()
        for i, nuc_row in self.top_df.iterrows():
            nbr_5p = int(nuc_row['5p_nbr'])
            nbr_3p = int(nuc_row['3p_nbr'])

            if nbr_3p != -1:

                if self.allow_circle:
                    if i < nbr_3p:
                        bonded_pair = (i, nbr_3p)
                    else:
                        bonded_pair = (nbr_3p, i)
                    bonded_nbrs.append(bonded_pair)
                else:
                    if not i < nbr_3p:
                        # Note: need this for OrderedSparse
                        raise RuntimeError(f"Nucleotides must be ordered such that i < j where j is 3' of i and i and j are on the same strand") # Note: circular strands wouldn't obey this
                    bonded_nbrs.append((i, nbr_3p)) # 5'->3'

        self.bonded_nbrs = onp.array(bonded_nbrs)
        self.seq = ''.join(self.top_df.base.tolist())
        self.unbonded_nbrs = get_unbonded_neighbors(self.n, bonded_nbrs)
        self.unbonded_nbrs = onp.array(self.unbonded_nbrs)

        # Store which nucleotides are on the ends
        is_end = list()
        for i, nuc_row in self.top_df.iterrows():
            nbr_5p = int(nuc_row['5p_nbr'])
            nbr_3p = int(nuc_row['3p_nbr'])
            if nbr_5p == -1 or nbr_3p == -1:
                is_end.append(True)
            else:
                is_end.append(False)
        self.is_end = jnp.array(onp.array(is_end).astype(onp.int32))


    def write(self, opath, reverse):
        """
        Write self.top_df (always 5'->3') to an oxDNA-style topology
        file. If reverse=True, write in 3'->5' format.
        """

        top_df_to_write = self.top_df
        if reverse:
            top_df_to_write = get_rev_top_df(self.top_df, self.rev_orientation_mapper)

        out_lines_top = [f"{self.n} {self.n_strands}"]
        out_lines_top += top_df_to_write.to_csv(
            header=None, index=False, sep=" ").strip('\n').split('\n')

        with open(opath, 'w+') as of:
            of.write('\n'.join(out_lines_top))

    def get_neighbor_list_fn(self, displacement_fn, box_size, r_cutoff, dr_threshold):

        # Construct nx2 mask
        dense_mask = onp.full((self.n, 2), self.n, dtype=onp.int32)
        counter = onp.zeros(self.n, dtype=onp.int32)
        for bp1, bp2 in self.bonded_nbrs:
            dense_mask[bp1, counter[bp1]] = bp2
            counter[bp1] += 1

            dense_mask[bp2, counter[bp2]] = bp1
            counter[bp2] += 1
        dense_mask = jnp.array(dense_mask, dtype=jnp.int32)

        def bonded_nbrs_mask_fn(dense_idx):
            nbr_mask1 = (dense_idx == dense_mask[:, 0].reshape(self.n, 1))
            dense_idx = jnp.where(nbr_mask1, self.n, dense_idx)

            nbr_mask2 = (dense_idx == dense_mask[:, 1].reshape(self.n, 1))
            dense_idx = jnp.where(nbr_mask2, self.n, dense_idx)
            return dense_idx

        neighbor_list_fn = neighbor_list(
            displacement_fn,
            box=box_size,
            r_cutoff=r_cutoff,
            dr_threshold=dr_threshold,
            custom_mask_function=bonded_nbrs_mask_fn,
            format=NeighborListFormat.OrderedSparse,
            disable_cell_list=True
        )

        return neighbor_list_fn


class TestTopology(unittest.TestCase):
    test_data_basedir = Path("data/test-data")

    def test_init(self):
        top_path = self.test_data_basedir / "simple-helix" / "generated.top"
        top_info = TopologyInfo(top_path, reverse_direction=True)

    def test_rev_mapping(self):
        top_df_dict_5to3 = {
            "strand": [1, 1, 1, 2, 2, 2],
            "base": ["A", "T", "G", "C", "A", "G"],
            "5p_nbr": [-1, 0, 1, -1, 3, 4],
            "3p_nbr": [1, 2, -1, 4, 5, -1]
        }
        top_df_5to3 = pd.DataFrame.from_dict(top_df_dict_5to3)

        computed_mapper = get_rev_orientation_idx_mapper(top_df_5to3, n=6, n_strands=2)
        true_mapper = {
            0: 2,
            2: 0,
            1: 1,
            3: 5,
            5: 3,
            4: 4
        }
        self.assertEqual(computed_mapper, true_mapper)

        top_df_3to5 = get_rev_top_df(top_df_5to3, computed_mapper)
        self.assertTrue(top_df_5to3.equals(get_rev_top_df(top_df_3to5, computed_mapper)))

    # FIXME: test neighbor list function


if __name__ == "__main__":
    unittest.main()
