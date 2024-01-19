from pathlib import Path
import pdb
import pandas as pd
from io import StringIO
from itertools import combinations
import numpy as np

import jax.numpy as jnp

from utils import DNA_BASES

from jax_md.partition import NeighborList, NeighborListFormat, neighbor_list


def get_unbonded_neighbors(n, bonded_neighbors):
    # First, set to all neighbors
    unbonded_neighbors = set(combinations(range(n), 2))

    # Then, remove all bonded neighbors
    unbonded_neighbors -= set(bonded_neighbors)

    # Finally, remove identities (which shouldn't be in there in the first place)
    unbonded_neighbors -= set([(i, i) for i in range(n)])

    # Return as a list
    return list(unbonded_neighbors) # return as list


def get_rev_orientation_idx_mapper(top_df, n, n_strands):

    master_idx_mapper = dict()

    # FIXME: error check that strands are 1-indexed? Or take top_df.strands.unique().min()...
    for strand in range(1, n_strands + 1): # strands are 1-indexed
        # Get the indexes
        index_orig = top_df[top_df.strand == strand].index
        index_rev = index_orig[::-1]

        strand_idx_mapper = dict(zip(index_orig, index_rev))
        master_idx_mapper.update(strand_idx_mapper)
    return master_idx_mapper



# Reverses the orientation of a top_df (either from 5'->3' to 3'->5' *or* from 3'->5' to 5'->3')
def get_rev_top_df(top_df, rev_orientation_mapper):
    rev_top_df = top_df.copy(deep=True)

    rev_top_df = rev_top_df.iloc[rev_top_df.index.map(rev_orientation_mapper).argsort()].reset_index(drop=True)

    # Update the values of neighbors appropriately
    rev_top_df.replace({"5p_nbr": rev_orientation_mapper, "3p_nbr": rev_orientation_mapper}, inplace=True)

    # Swap the order of the 3p and 5p neighbor columns
    cols = list(top_df.columns.values)
    cols_reordered = ["strand", "base", cols[3], cols[2]]
    # cols_reordered = ["strand", "base", "5p_nbr", "3p_nbr"] # e.g. if we are going 3'->5' to 5'->3'
    rev_top_df = rev_top_df.reindex(columns=cols_reordered)

    return rev_top_df

# Always 5'->3'
# reverse_direction=True if top_path points to a 3'->5' topology file, False otherwise (5'->3')
class TopologyInfo:
    def __init__(self, top_path, reverse_direction):
        # Store our initial information
        self.top_path = top_path
        self.reverse_direction = reverse_direction
        self.rev_orientation_mapper = None

        # Read in our topology information in 5'->3' format
        self.read()

        # If we didn't populate it during reading, populate our orientation mapper
        if self.rev_orientation_mapper is None:
            self.rev_orientation_mapper = get_rev_orientation_idx_mapper(
                self.top_df, self.n, self.n_strands)


    def build_5to3_df(self, top_lines_5to3):
        self.top_df = pd.read_csv(StringIO('\n'.join(top_lines_5to3[1:])),
                                  names=["strand", "base", "5p_nbr", "3p_nbr"],
                                  delim_whitespace=True)

    def build_3to5_df(self, top_lines_3to5):
        # Load in a 3'->'5 dataframe
        top_df_3to5 = pd.read_csv(StringIO('\n'.join(top_lines_3to5[1:])),
                                  names=["strand", "base", "3p_nbr", "5p_nbr"],
                                  delim_whitespace=True)

        # Construct the idx mapper to reverse orientations, and save it
        rev_orientation_mapper = get_rev_orientation_idx_mapper(top_df_3to5, self.n, self.n_strands)
        self.rev_orientation_mapper = rev_orientation_mapper

        # Change the order of nucleotides to be 5'->3'
        top_df_5to3 = get_rev_top_df(top_df_3to5, self.rev_orientation_mapper)

        # Set self.top_df to be the new, 5'->3' dataframe
        self.top_df = top_df_5to3


    # Once we have self.top_df populated (note: self.top_df is always 5'->3'), process its information
    def process(self):
        bonded_nbrs = list()

        for i, nuc_row in self.top_df.iterrows():
            nbr_5p = int(nuc_row['5p_nbr'])
            nbr_3p = int(nuc_row['3p_nbr'])

            if nuc_row.base not in DNA_BASES:
                raise RuntimeError(f"Invalid base: {nuc_row.base}")

            if nbr_3p != -1:
                if not i < nbr_3p:
                    # Note: need this for OrderedSparse
                    raise RuntimeError(f"Nucleotides must be ordered such that i < j where j is 3' of i and i and j are on the same strand") # Note: circular strands wouldn't obey this
                bonded_nbrs.append((i, nbr_3p)) # 5'->3'

        self.bonded_nbrs = np.array(bonded_nbrs)
        self.seq = ''.join(self.top_df.base.tolist()) # FIXME: could one-hot
        self.unbonded_nbrs = get_unbonded_neighbors(self.n, bonded_nbrs)
        self.unbonded_nbrs = np.array(self.unbonded_nbrs)

    def read(self):
        # Check that our topology file exists
        if not Path(self.top_path).exists():
            raise RuntimeError(f"Topology file does not exist: {self.top_path}")

        # Read the lines from our topology file
        with open(self.top_path) as f:
            top_lines = f.readlines()

        # Read the information from the first line -- # of nucleotides and # of strands
        sys_info = top_lines[0].strip().split()
        self.n = int(sys_info[0])
        self.n_strands = int(sys_info[1])

        # Populate our 5'->3' `self.top_df` depending on the direction of our initial file
        if self.reverse_direction:
            self.build_3to5_df(top_lines)
        else:
            self.build_5to3_df(top_lines)

        # Once we've constructed our 5'->3' `self.top_df`, infer the neighbors and sequence
        self.process()


    # Write self.top_df (always 5'->3') to an oxDNA-style topology file
    # if reverse=True, write in 3'->5' format
    def write(self, opath, reverse):
        top_df_to_write = self.top_df
        if reverse:
            top_df_to_write = get_rev_top_df(self.top_df, self.rev_orientation_mapper)

        out_lines_top = [f"{self.n} {self.n_strands}"]
        out_lines_top += top_df_to_write.to_csv(
            header=None, index=False, sep=" ").strip('\n').split('\n')

        with open(opath, 'w+') as of:
            of.write('\n'.join(out_lines_top))

    def get_neighbor_list_fn(self, displacement_fn, box_size, r_cutoff, dr_threshold):
        import jax.debug

        # Construct nx2 mask
        ## FIXME: does this have to be symmetric? maybe not
        dense_mask = np.full((self.n, 2), self.n, dtype=np.int32)
        counter = np.zeros(self.n, dtype=np.int32)
        for bp1, bp2 in self.bonded_nbrs:
            dense_mask[bp1, counter[bp1]] = bp2
            counter[bp1] += 1

            dense_mask[bp2, counter[bp2]] = bp1
            counter[bp2] += 1
        dense_mask = jnp.array(dense_mask, dtype=jnp.int32)

        """
        mask_val = self.n
        # all_bonded_nbrs = np.concatenate((self.bonded_nbrs, self.bonded_nbrs[:, [1, 0]])) # includes reverse order
        # mask_pairs = tuple(all_bonded_nbrs.T)
        to_mask = jnp.array([(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)], dtype=jnp.int32)
        mask_pairs = tuple(jnp.array(to_mask).T) # will be a tuple of jnp arrays
        def bonded_nbrs_mask_fn(dense_idx):
            # return dense_idx.at[self.bonded_nbrs[:, 0], self.bonded_nbrs[:, 1]].set(mask_val)
            jax.debug.breakpoint()
            return dense_idx.at[mask_pairs].set(mask_val)
        """

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
            format=NeighborListFormat.OrderedSparse
        )

        return neighbor_list_fn


if __name__ == "__main__":
    top_path = "/home/ryan/Documents/Harvard/research/brenner/jaxmd-oxdna/data/simple-helix/generated.top"
    top_info = TopologyInfo(top_path, True)
    pdb.set_trace()
    print("done")
