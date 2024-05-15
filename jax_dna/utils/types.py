"""Common types used in the package."""

from enum import Enum
from pathlib import Path

import jaxtyping as jaxtyp


class OxdnaFormat(Enum):
    """OxDNA file format."""

    CLASSIC = 1
    NEW = 2


PathOrStr = Path | str

# jaxtyping array type documentation: https://docs.kidger.site/jaxtyping/api/array/#array
# NOTE: jaxtyping has it's own scalar type but general to any numeric type
#       we can switch to it here if we think that is ok.
Scalar = jaxtyp.Float[jaxtyp.Array, ""]
Vector3D = jaxtyp.Float[jaxtyp.Array, "3"]
Vector4D = jaxtyp.Float[jaxtyp.Array, "4"]
Arr_Nucleotide = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides"]
Arr_Nucleotide_3 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 3"]
Arr_Nucleotide_4 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 4"]
Arr_Nucleotide_15 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 15"]
Arr_Bonded_Neighbors = jaxtyp.Int[jaxtyp.Array, "n_bonded_pairs 2"]
Arr_Bonded_Neighbors = jaxtyp.Int[jaxtyp.Array, "n_unbonded_pairs 2"]
Arr_States = jaxtyp.Int[jaxtyp.Array, "#n_states"]
Arr_States_3 = jaxtyp.Int[jaxtyp.Array, "#n_states 3"]
