import dataclasses as dc

import jax_dna.utils.types as typ

@dc.dataclass(frozen=True)
class OxDNAConfig:
    executable_path: typ.PathOrStr
    input_file: typ.PathOrStr


@dc.dataclass(frozen=True)
class OxDNAInput:
    """Input information for an oxDNA simulation.

    See details of options here:
    https://lorenzo-rovigatti.github.io/oxDNA/input.html
    """
    # Core options
    T:str # temperature in simulation units or kelvin suffix ([K|k]) or celsius suffix ([C|c]
    restart_step_counter:bool
    steps:int
    conf_file:typ.PathOrStr # this is an input file
    topology:typ.PathOrStr # this is an input file
    trajectory_file:typ.PathOrStr # this is the output file
    trajectory_print_momenta:bool = True
    time_scale:str # one of "linear" or "log_lin"
    print_conf_interal:int
    print_energy_every:int
    interaction_type:str = "DNA" # on of "DNA", "DNA2", "RNA", "RNA2"
    max_io:float = 1.0 # units are MB/s
    fix_diffusion:bool = True
    fix_diffusion_every:int = 100_000
