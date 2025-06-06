"""Common types used in the package."""

from enum import Enum
from pathlib import Path
from typing import Any

import jaxtyping as jaxtyp


class oxDNAFormat(Enum):  # noqa: N801 - ocDNA is a special word
    """oxDNA file format."""

    CLASSIC = 1
    NEW = 2


class oxDNASimulatorType(Enum):  # noqa: N801 - ocDNA is a special word
    """oxDNA simulator type."""

    DNA1 = 1
    DNA2 = 2


class oxDNAModelHType(Enum):  # noqa: N801 - ocDNA is a special word
    """oxDNA model.h file type."""

    INTEGER = 1
    FLOAT = 2
    STRING = 3


PathOrStr = Path | str

# jaxtyping array type documentation: https://docs.kidger.site/jaxtyping/api/array/#array
# NOTE: jaxtyping has it's own scalar type but general to any numeric type
#       we can switch to it here if we think that is ok.
Scalar = jaxtyp.Float[jaxtyp.Array, ""]
Vector3D = jaxtyp.Float[jaxtyp.Array, "3"]
Vector4D = jaxtyp.Float[jaxtyp.Array, "4"]
Arr_N = jaxtyp.Array  # jaxtyp.Float[jaxtyp.Array, "#n"]
Arr_Nucleotide = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides"]
Arr_Nucleotide_Int = jaxtyp.Int[jaxtyp.Array, "#n_nucleotides"]
Arr_Nucleotide_2 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 2"]
Arr_Nucleotide_2_Int = jaxtyp.Int[jaxtyp.Array, "#n_nucleotides 2"]
Arr_Nucleotide_3 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 3"]
Arr_Nucleotide_4 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 4"]
Arr_Nucleotide_15 = jaxtyp.Float[jaxtyp.Array, "#n_nucleotides 15"]
Arr_Bonded_Neighbors = jaxtyp.Int[jaxtyp.Array, "#n_bonded_pairs"]
Arr_Unbonded_Neighbors = jaxtyp.Int[jaxtyp.Array, "#n_unbonded_pairs"]
Arr_Bonded_Neighbors_2 = jaxtyp.Int[jaxtyp.Array, "#n_bonded_pairs 2"]
Arr_Unbonded_Neighbors_2 = jaxtyp.Int[jaxtyp.Array, "#n_unbonded_pairs 2"]
Arr_States = jaxtyp.Int[jaxtyp.Array, "#n_states"]
Arr_States_3 = jaxtyp.Int[jaxtyp.Array, "#n_states 3"]

ARR_OR_SCALAR = Arr_N | Scalar

Arr_Unpaired = jaxtyp.Int[jaxtyp.Array, "#n_unpaired"]
Arr_Unpaired_Pseq = jaxtyp.Float[jaxtyp.Array, "#n_unpaired 4"]
Arr_Bp = jaxtyp.Int[jaxtyp.Array, "#n_bp 2"]
Arr_Bp_Pseq = jaxtyp.Float[jaxtyp.Array, "#n_bp 4"]
Discrete_Sequence = Arr_Nucleotide_Int
Probabilistic_Sequence = tuple[Arr_Unpaired_Pseq, Arr_Bp_Pseq]
Sequence = Discrete_Sequence | Probabilistic_Sequence

MetaData = dict[str, Any]
Grads = jaxtyp.PyTree
Params = jaxtyp.PyTree | dict[str, jaxtyp.PyTree]
PyTree = jaxtyp.PyTree

SimulatorActorOutput = str
