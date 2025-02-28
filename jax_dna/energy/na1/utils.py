"""Utility functions for oxNA energy calculations."""

import jax.numpy as jnp

import jax_dna.input.topology as jd_top
import jax_dna.utils.types as typ


def is_rna_pair(i: int, j: int, nt_type: typ.Arr_Nucleotide) -> jnp.ndarray:
    """Checks if both nucleotides at `i` and `j` are RNA."""
    return jnp.logical_and(nt_type[i] == jd_top.NucleotideType.RNA, nt_type[j] == jd_top.NucleotideType.RNA)


def is_dna_rna_pair(i: int, j: int, nt_type: typ.Arr_Nucleotide) -> jnp.ndarray:
    """Checks if `i` is DNA and `j` is RNA."""
    return jnp.logical_and(nt_type[i] == jd_top.NucleotideType.DNA, nt_type[j] == jd_top.NucleotideType.RNA)
