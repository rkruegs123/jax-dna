"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import os
import subprocess
from pathlib import Path

import chex
import jax

import jax_dna.input.oxdna_input as jd_oxdna
import jax_dna.input.topology as jd_top
import jax_dna.input.trajectory as jd_traj

REQUIRED_KEYS = {
    "oxnda_bin",
    "input_directory",
}

ERR_OXDNA_NOT_FOUND = "OXDNA binary not found at: {}"
ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"
ERR_INPUT_FILE_NOT_FOUND = "Input file not found: {}"
ERR_OXDNA_FAILED = "OXDNA simulation failed"
OXDNA_TRAJECTORY_FILE_KEY = "trajectory_file"
OXDNA_TOPOLOGY_FILE_KEY = "topology"

BIN_PATH_ENV_VAR = "OXDNA_BIN_PATH"
ERR_BIN_PATH_NOT_SET = "OXDNA_BIN_PATH environment variable not set"


@chex.dataclass(frozen=True, init=False)
class oxDNASimulator:  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    def __init__(self, input_dir:str|Path, **kwargs):
        """Initialize the oxDNA simulator.

        We override the __init__ method so we can throw away the kwargs that other
        simulators might need.
        """
        pass


    def run(
        self,
        **kwargs,
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""
        init_dir = Path(self.input_dir)

        input_file = init_dir / "input"
        if not input_file.exists():
            raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

        if BIN_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BIN_PATH_NOT_SET)

        oxdna_config = jd_oxdna.read(init_dir / "input")
        output_file = init_dir / oxdna_config["trajectory_file"]

        completed_proc = subprocess.run(
            [  # noqa: S603
                os.environ[BIN_PATH_ENV_VAR],
                str(input_file),
            ],
            stderr=subprocess.STDOUT,
            check=True,
        )

        if completed_proc.returncode != 0:
            with Path(init_dir / "oxdna.log").open("w") as f:
                f.write(completed_proc.stdout)
            raise RuntimeError(ERR_OXDNA_FAILED)

        # read the output trajectory file
        topology = jd_top.from_oxdna_file(init_dir / oxdna_config["topology"])
        # return the trajectory
        return jd_traj.from_file(output_file, topology.strand_counts, is_oxdna=True)
