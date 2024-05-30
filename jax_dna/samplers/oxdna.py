import subprocess
from pathlib import Path
from typing import Any

import jax_dna.input.trajectory as jd_traj
import jax_dna.input.topology as jd_top
import jax_dna.utils.types as typ

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

def run(
    input_config: dict[str, Any],
    meta:dict[str, Any],
    params: dict[str, Any]
) -> tuple[jd_traj.Trajectory, dict[str, Any]]:
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


def _run_oxdna(oxdna_bin: Path, input_file: Path) -> None:

    if not oxdna_bin.exists():
        raise FileNotFoundError(ERR_OXDNA_NOT_FOUND.format(oxdna_bin))

    if not input_file.exists():
        raise FileNotFoundError(ERR_INPUT_FILE_NOT_FOUND.format(input_file))

    completed_proc = subprocess.run(
        [
            str(oxdna_bin),
            str(input_file),
        ],
        stderr=subprocess.STDOUT
    )

    if completed_proc.returncode != 0:
        with open("oxdna.log", "w") as f:
            f.write(completed_proc.stdout)
        raise RuntimeError(ERR_OXDNA_FAILED)



def _is_valid_input_file_line(stripped_line:str) -> bool:
    return len(stripped_line) > 0 and not stripped_line.startswith("#")

def _parse_input_file_line(stripped_line:str) -> list[tuple[str, str]]:
    return [tuple(map(str.strip, kv.split("="))) for kv in stripped_line.split(";")]

def _parse_input_file(file_data:list[str]) -> dict[str, Any]:
    input_config = {}
    stripped_lines = map(str.strip, file_data)
    valid_lines = filter(_is_valid_input_file_line, stripped_lines)
    input_config = dict(
        _parse_input_file_line(line)
        for line in valid_lines
    )

    return input_config


def _validate_input_config(input_config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_KEYS - input_config.keys()
    if missing_keys:
        raise ValueError(ERR_MISSING_REQUIRED_KEYS.format(missing_keys))
