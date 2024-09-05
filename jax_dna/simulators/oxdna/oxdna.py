"""OXDNA sampler module.

Run an jax_dna simulation using an oxDNA sampler.
"""

import subprocess
from pathlib import Path
from typing import Any

import chex

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


@chex.dataclass
class oxDNASimulator:  # noqa: N801 oxDNA is a special word
    """A sampler base on running an oxDNA simulation."""

    oxdna_bin: Path

    def __post_init__(self) -> None:
        """Build the run function using the provided parameters."""

    def run(
        input_directory:str,
    ) -> jd_traj.Trajectory:
        """Run an oxDNA simulation."""
        # get the output file location from the input file, called "trajectory_file"
        # run oxDNA from the target directory, assume that an env var
        # read the output trajectory file
        # return the trajectory



def run(
    input_config: dict[str, Any], meta: dict[str, Any], params: dict[str, Any]
) -> tuple[jd_traj.Trajectory, dict[str, Any]]:
    """Run an OXDNA simulation."""
    _ = params

    _validate_input_config(input_config)

    input_file = Path(input_config["input_directory"]) / "input"

    _run_oxdna(Path(input_config["oxdna_bin"]), input_file)

    oxdna_config = _parse_input_file(input_file.read_text().split("\n"))
    topology = jd_top.from_oxdna_file(Path(oxdna_config[OXDNA_TOPOLOGY_FILE_KEY]))

    trajectory = jd_traj.from_file(
        Path(oxdna_config[OXDNA_TRAJECTORY_FILE_KEY]),
        topology.strand_counts,
        is_oxdna=True,
    )

    return (trajectory, meta)


# TODO(ryanhausen): This should probably be found as an environment variable rather than a str path
# https://github.com/ssec-jhu/jax-dna/issues/8
def _run_oxdna(oxdna_bin: Path, input_file: Path) -> None:
    if not oxdna_bin.exists():
        raise FileNotFoundError(ERR_OXDNA_NOT_FOUND.format(oxdna_bin))

    if not input_file.exists():
        raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

    completed_proc = subprocess.run(
        [  # noqa: S603
            str(oxdna_bin),
            str(input_file),
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    if completed_proc.returncode != 0:
        with Path("oxdna.log").open("w") as f:
            f.write(completed_proc.stdout)
        raise RuntimeError(ERR_OXDNA_FAILED)


def _is_valid_input_file_line(stripped_line: str) -> bool:
    return len(stripped_line) > 0 and not stripped_line.startswith("#")


def _parse_input_file_line(stripped_line: str) -> list[tuple[str, str]]:
    return [tuple(map(str.strip, kv.split("="))) for kv in stripped_line.split(";")]


def _parse_input_file(file_data: list[str]) -> dict[str, Any]:
    stripped_lines = map(str.strip, file_data)
    valid_lines = filter(_is_valid_input_file_line, stripped_lines)

    return dict(_parse_input_file_line(line) for line in valid_lines)


def _validate_input_config(input_config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_KEYS - input_config.keys()
    if missing_keys:
        raise ValueError(ERR_MISSING_REQUIRED_KEYS.format(missing_keys))
