"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import os
import subprocess
from pathlib import Path

import chex

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


@chex.dataclass
class oxDNASimulator:  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    def __post_init__(self) -> None:
        """Build the run function using the provided parameters."""

    def run(
        self,
        input_directory: str | Path,
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""
        # get the output file location from the input file, called "trajectory_file"
        input_directory = Path(input_directory)

        input_file = input_directory / "input"
        if not input_file.exists():
            raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

        if BIN_PATH_ENV_VAR not in os.environ:
            raise ValueError(ERR_BIN_PATH_NOT_SET)

        oxdna_config = jd_oxdna.read(input_directory / "input")
        output_file = input_directory / oxdna_config["trajectory_file"]

        completed_proc = subprocess.run(
            [  # noqa: S603
                os.environ[BIN_PATH_ENV_VAR],
                str(input_file),
            ],
            stderr=subprocess.STDOUT,
            check=True,
        )

        if completed_proc.returncode != 0:
            with Path(input_directory / "oxdna.log").open("w") as f:
                f.write(completed_proc.stdout)
            raise RuntimeError(ERR_OXDNA_FAILED)

        # read the output trajectory file
        topology = jd_top.from_oxdna_file(input_directory / oxdna_config["topology"])
        # return the trajectory
        return jd_traj.from_file(output_file, topology.strand_counts, is_oxdna=True)
