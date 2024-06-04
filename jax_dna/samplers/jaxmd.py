from typing import Any

import jax
import jax.numpy as jnp
import jax_md

import jax_dna.input.trajectory as jd_traj
import jax_dna.input.topology as jd_top

REQUIRED_KEYS = {
    "trajectory_file",
    "topology_file",
}

ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"
ERR_INVALID_SPACE = "Invalid space type: {}"

def run(
    input_config: dict[str, Any],
    meta:dict[str, Any],
    params: dict[str, Any]
) -> tuple[
    jd_traj.Trajectory|list[jax_md.rigid_body.RigidBody],
    dict[str, Any]
]:

    _validate_input_config(input_config)


    topology = jd_top.from_oxdna_file(input_config["topology_file"])
    init_body = jd_traj.from_file(
        input_config["trajectory_file"],
        topology.strand_counts,
        is_oxdna=True,
    ).states[0].to_rigid_body()

    displacement_fn, shift_fn = _get_space(input_config)

    init_fn, step_fn = input_config["simulate_fn"](
        input_config["energy_fn"],
        shift_fn,
        input_config["dt"],
        input_config["kT"],
        input_config["gamma"],
    )

    init_state = init_fn(
        random_key,
        body,
        mass=mass,
        seq=topology.seq_one_hot,
        bonded_nbrs=topology.bonded_neighbors,
        unbonded_nbrs=topology.unbonded_neighbors.T,
    )

    def simulation_step_fn(state, _):
        state = step_fn(
            state,
            seq=topology.seq_one_hot,
            bonded_nbrs=topology.bonded_neighbors,
            unbonded_nbrs=topology.unbonded_neighbors.T,
        )
        return state, state.position


    _, trajectory = input_config["scan_fn"](
        simulation_step_fn,
        init_state,
        jnp.arange(input_config["n_steps"])
    )

    return (trajectory, meta)


def _validate_input_config(input_config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_KEYS - input_config.keys()
    if missing_keys:
        raise ValueError(ERR_MISSING_REQUIRED_KEYS.format(missing_keys))


def _get_space(input_config: dict[str, Any]) -> jax_md.space.Space:

    if input_config.get("space", "free") == "free":
        return jax_md.space.free()
    else:
        raise ValueError(ERR_INVALID_SPACE.format(input_config["space"]))

