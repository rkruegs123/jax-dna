import dataclasses as dc
from typing import Any

import jax_dna.input.trajectory as jdtraj
import jax_dna.utils.types as typ

@dc.dataclass
class OxDnaConfig:
    executable: typ.PathOrStr



def run(
    input_config: dict[str, Any],
    meta:dict[str, Any],
    params: dict[str, Any]
) -> tuple[jdtraj.Trajectory, dict[str, Any]]:
    pass




