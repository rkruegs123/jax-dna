from io import StringIO
import numpy as onp
import pandas as pd
from pathlib import Path
import pdb
import unittest
import itertools
from itertools import combinations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax_md.partition import NeighborListFormat, neighbor_list

from jax_dna.common.utils import bcolors, DNA_ALPHA, RNA_ALPHA


RES_ALPHA = "MGKTRADEYVLQWFSHNPCI"


class ProteinNucAcidTopology:
    def __init__(self, top_path, par_path, reverse=False):
        self.top_path = Path(top_path)
        self.par_path = Path(par_path)
        self.reverse = reverse

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
        assert(len(sys_info) == 5)

        self.n = int(sys_info[0])
        self.n_strands_total = int(sys_info[1])
        self.n_na = int(sys_info[2])
        self.n_protein = int(sys_info[3])
        self.n_na_strands = int(sys_info[4])
        self.n_protein_strands = self.n_strands_total - self.n_na_strands

        assert(self.n == self.n_na + self.n_protein)

        # Read the input file into a dataframe, regardless of orientation

        ## Protein
        res_types = list()
        network = list()

        ## NA
        bonded_nbrs = list()
        nt_types = list()

        started_na = False
        start_nt_idx = None
        curr_strand_idx = None
        curr_strand_count = 0
        strand_counts = list()
        n_protein_strand_count = 0
        n_na_strand_count = 0
        is_end = list()
        is_nt_idx = list()
        is_protein_idx = list()
        for curr_idx, line in enumerate(top_lines[1:]):
            tokens = line.strip().split()
            strand_idx = int(tokens[0])

            if curr_strand_idx is None:
                curr_strand_idx = strand_idx
                curr_strand_count += 1

                if strand_idx < 0:
                    n_protein_strand_count += 1
                else:
                    n_na_strand_count += 1

            elif curr_strand_idx != strand_idx:

                if strand_idx < 0:
                    assert(strand_idx < curr_strand_idx)
                else:
                    assert(strand_idx > curr_strand_idx)
                strand_counts.append(curr_strand_count)
                curr_strand_count = 1
                curr_strand_idx = strand_idx

                if strand_idx < 0:
                    n_protein_strand_count += 1
                else:
                    n_na_strand_count += 1

            else:
                curr_strand_count += 1

            if strand_idx < 0:
                # Protein

                assert(not started_na)

                res_type = tokens[1]
                assert(res_type in set(RES_ALPHA))
                res_types.append(res_type)
                n_term_nbr = int(tokens[2]) # do nothing with this
                c_term_nbr = int(tokens[3])
                extra_nbrs = [int(idx) for idx in tokens[4:]]
                for nbr_idx in [c_term_nbr] + extra_nbrs:
                    if nbr_idx != -1:
                        network.append((curr_idx, nbr_idx))

                if n_term_nbr == -1 or c_term_nbr == -1:
                    is_end.append(1)
                else:
                    is_end.append(0)

                is_nt_idx.append(0)
                is_protein_idx.append(1)

            else:
                # Nucleic Acids
                if not started_na:
                    started_na = True
                    start_nt_idx = curr_idx
                    assert(curr_idx == self.n_protein)

                assert(strand_idx > 0)

                assert(len(tokens) == 4)

                nt = tokens[1]
                nt_types.append(nt)

                # We read in in 3'->5', but will eventually reverse things so these pairs will eventually be 5'->3'
                nbr_3p = int(tokens[2])
                nbr_5p = int(tokens[3])

                if nbr_5p != -1:
                    bonded_nbrs.append((curr_idx, nbr_5p))

                if nbr_5p == -1 or nbr_3p == -1:
                    is_end.append(1)
                else:
                    is_end.append(0)

                is_nt_idx.append(1)
                is_protein_idx.append(0)
        strand_counts.append(curr_strand_count)

        self.network = onp.array(network)
        self.is_end = onp.array(is_end).astype(onp.int32)
        self.is_nt_idx = onp.array(is_nt_idx).astype(onp.int32)
        self.is_protein_idx = onp.array(is_protein_idx).astype(onp.int32)


        assert(self.n_protein_strands == n_protein_strand_count)
        assert(self.n_na_strands == n_na_strand_count)

        self.bonded_nbrs = onp.array(bonded_nbrs)
        strand_bounds = list(itertools.pairwise([0, *itertools.accumulate(strand_counts)]))
        dummy_protein_nt = "A"
        if self.reverse:
            nt_types_rev = list(itertools.chain.from_iterable([nt_types[s:e][::-1] for s, e in strand_bounds[self.n_protein_strands:]]))
            self.nt_seq_idx = onp.array([-1]*self.n_protein + [DNA_ALPHA.index(nt) for nt in nt_types_rev])
            self.nt_seq = dummy_protein_nt*self.n_protein + ''.join(nt_types_rev)
        else:
            self.nt_seq_idx = onp.array([-1]*self.n_protein + [DNA_ALPHA.index(nt) for nt in nt_types])
            self.nt_seq = dummy_protein_nt*self.n_protein + ''.join(nt_types)



        idxs_nt = onp.arange(self.n)
        rev_idxs_nt = list(itertools.chain.from_iterable([idxs_nt[s:e][::-1] for s, e in strand_bounds[self.n_protein_strands:]]))
        idxs = onp.arange(self.n)
        rev_idxs = list(onp.arange(self.n_protein)) + rev_idxs_nt
        self.rev_orientation_mapper = dict(zip(idxs, rev_idxs))

        network_set = set(network)
        self.anm_network = onp.array(network)
        self.aa_seq_idx = onp.array([RES_ALPHA.index(res) for res in res_types] + [-1]*self.n_na, dtype=onp.int32)

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


        # Generate unbonded pairs

        ## Get NA unbonded pairs

        ### First, set to all neighbors
        unbonded_nbrs_nt = set(combinations(range(self.n_protein, self.n), 2))

        ### Then, remove all bonded neighbors
        unbonded_nbrs_nt -= set(bonded_nbrs)

        #### Incase of circles
        rev_bonded_nbrs = set([(j, i) for (i, j) in bonded_nbrs])
        unbonded_nbrs_nt -= rev_bonded_nbrs

        ### Finally, remove identities (which shouldn't be in there in the first place)
        unbonded_nbrs_nt -= set([(i, i) for i in range(self.n_protein, self.n)])

        unbonded_nbrs_nt = list(unbonded_nbrs_nt)

        ## Get NA/Protein hybrids pairs
        protein_idxs = onp.arange(self.n_protein)
        nt_idxs = onp.arange(self.n_protein, self.n)
        na_protein_unbonded_nbrs = list()
        for p_idx in protein_idxs:
            for nt_idx in nt_idxs:
                na_protein_unbonded_nbrs.append((p_idx, nt_idx))
        na_protein_unbonded_nbrs = onp.array(na_protein_unbonded_nbrs)

        self.unbonded_nbrs_nt = unbonded_nbrs_nt
        self.unbonded_nbrs_protein_nt = na_protein_unbonded_nbrs
        self.unbonded_nbrs = onp.concatenate([unbonded_nbrs_nt, na_protein_unbonded_nbrs])

    # FIXME: doesn't mask out pairs in the network. Have to change the mask function i think a bit for that when more than two neighbors, as the ANMnetwork has
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


if __name__ == "__main__":
    test_data_basedir = Path("data/test-data")

    top_path = test_data_basedir / "protein-top" / "HCAGE" / "hcage.top"
    par_path = test_data_basedir / "protein-top" / "HCAGE" / "hcage.par"
    top_info = ProteinNucAcidTopology(top_path, par_path)

    # FIXME
    # 1. neighbor list support for proteins will be tough without being too inefficient. just shouldn't do it for now.
    # 2. Should all inter-protein pairs from unbonded neighbors. Maybe do it by construction rather than by omission
    # 3. well, all unbonded neighbors have excluded volume... so we actually don't have to check types for that. BUT, DNA2 unbonded should be filtered based on type... should maybe do what we do in SSEC repo where we return have a member function that e.g. returns db_dgs and hb_dgs instead of db_dg, and then we can compute them all and mask... if we do this we have to make sure to test
