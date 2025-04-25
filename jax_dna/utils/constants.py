"""Constants for the oxDNA model."""

import jax.numpy as jnp

DNA_ALPHA = "ACGT"
RNA_ALPHA = "ACGU"
N_NT = len(DNA_ALPHA)

NUCLEOTIDES_IDX: dict[str, int] = {nt: nt_idx for nt_idx, nt in enumerate(DNA_ALPHA)} | {
    nt: nt_idx for nt_idx, nt in enumerate(RNA_ALPHA)
}

BP_TYPES = ["AT", "TA", "GC", "CG"]
N_BP_TYPES = len(BP_TYPES)

N_NT_PER_BP = 2

BP_IDXS = jnp.array([[DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)] for nt1, nt2 in BP_TYPES])

BP_IDX_MAP = {(DNA_ALPHA.index(nt1), DNA_ALPHA.index(nt2)): bp_idx for bp_idx, (nt1, nt2) in enumerate(BP_TYPES)}


DEFAULT_TEMP = 296.15  # Kelvin

TWO_DIMENSIONS = 2
