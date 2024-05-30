from typing import Any

import jax_md

import jax_dna.input.trajectory as jd_traj
import jax_dna.input.topology as jd_top

REQUIRED_KEYS = {
    "trajectory_file",
    "topology_file",
}

ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"


def run(
    input_config: dict[str, Any],
    meta:dict[str, Any],
    params: dict[str, Any]
) -> tuple[jd_traj.Trajectory, dict[str, Any]]:

    _validate_input_config(input_config)

    topology = jd_top.from_oxdna_file(input_config["topology_file"])

    trajectory = None

    return (trajectory, meta)


def _validate_input_config(input_config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_KEYS - input_config.keys()
    if missing_keys:
        raise ValueError(ERR_MISSING_REQUIRED_KEYS.format(missing_keys))


def _get_space(input_config: dict[str, Any]) -> jax_md.space.Space:

    if input_config.get("space", "free") == "free":
        return jax_md.space.free()