"""A sampler based on running a jax_md simulation routine."""

from collections.abc import Callable
import functools
from typing import Any

import jax
import jax_md

import jax_dna.energy as jd_energy
import jax_dna.input.trajectory as jd_traj
import jax_dna.samplers.jax_md.utils as jaxmd_utils

REQUIRED_KEYS = {
    "track_gradients",
    "input_directory",
    "energy_configs",
    "energy_function",
    "opt_params",
}

ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"

SIM_STATE = tuple[jax_md.simulate.SimulationState, jaxmd_utils.NeighborHelper]


def run(
    params: dict[str, Any],
    input_config: dict[str, Any],
) -> tuple[jd_traj.Trajectory, dict[str, Any]]:
    """Run a jax_md simulation."""
    _ = params

    _validate_input_config(input_config)

    # load input files

    # run simulation

    # return trajectory and metadata when done


def _build_run_fn(
    energy_configs: list[jd_energy.base.BaseEnergyConfiguration],
    energy_fns: list[jd_energy.base.BaseEnergyFunction],
    simulator_params: jaxmd_utils.StaticSimulatorParams,
    space: jax_md.space.Space,
    transform_fn: Callable,
    simulator_init: Callable[[Callable, Callable], jax_md.simulate.Simulator],
    key: jax.random.PRNGKey,
) -> Callable[[dict[str, float]], jd_traj.Trajectory]:
    displacement_fn, shift_fn = space

    # if there isn't a dynamic neighbor list, we can just use the static one
    # which ends up being something like top_info.unbonded_nbrs.T
    neighbors = jaxmd_utils.NeighborHelper()

    def run_fn(opt_params: dict[str, float]) -> jd_traj.Trajectory:
        # The  energy function configuration init calls need to happen inside the function
        # so that if the gradient is calculated for this run it will be tracked
        transformed_fns = [
            energy_fn(
                displacement_fn=displacement_fn,
                params=(config | param).init_params(),
            )
            for param, config, energy_fn in zip(opt_params, energy_configs, energy_fns, strict=True)
        ]

        energy_fn = jd_energy.ComposedEnergyFunction(
            transformed_fns,
            rigid_body_transform_fn=transform_fn,
        )

        init_fn, step_fn = simulator_init(energy_fn, shift_fn)

        init_state = init_fn(  # noqa: F841, temp
            key=key,
            unbonded_neighbors=neighbors.idx,
            **simulator_params.init_fn,
        )

        def apply_fn(in_state: SIM_STATE, _: int) -> tuple[SIM_STATE, jax_md.rigid_body.RigidBody]:
            state, neighbors = in_state

            state = step_fn(
                state,
                unbonded_neighbors=neighbors.idx,
                **simulator_params.step_fn,
            )

            neighbors = neighbors.update(state.position.center)

            return (state, neighbors), state.position

        scan_fn = jax.lax.scan if simulator_params.checkpoint_every <= 0 else functools.partial(jaxmd_utils.checkpoint_scan, checkpoint_every=simulator_params.checkpoint_every)






def _validate_input_config(input_config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_KEYS - input_config.keys()
    if missing_keys:
        raise ValueError(ERR_MISSING_REQUIRED_KEYS.format(missing_keys))
