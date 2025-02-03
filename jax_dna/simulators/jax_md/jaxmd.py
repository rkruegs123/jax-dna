"""A sampler based on running a jax_md simulation routine."""

import functools
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy as jdna_energy
import jax_dna.energy.base as jd_energy_fn
import jax_dna.energy.configuration as jd_energy_cnfg
import jax_dna.input.topology as jd_topology
import jax_dna.input.trajectory as jd_traj
import jax_dna.simulators.base as jd_sim_base
import jax_dna.simulators.io as jd_sio
import jax_dna.simulators.jax_md.utils as jaxmd_utils
import jax_dna.utils.types as jd_types

REQUIRED_KEYS = {
    "track_gradients",
    "input_directory",
    "energy_configs",
    "energy_function",
    "opt_params",
}

ERR_MISSING_REQUIRED_KEYS = "Missing required keys: {}"

SIM_STATE = tuple[jaxmd_utils.SimulationState, jaxmd_utils.NeighborHelper]


@chex.dataclass
class JaxMDSimulator(jd_sim_base.BaseSimulation):
    """A sampler based on running a jax_md simulation routine."""

    energy_configs: list[jd_energy_cnfg.BaseConfiguration]
    energy_fns: list[jd_energy_fn.BaseEnergyFunction]
    simulator_params: jaxmd_utils.StaticSimulatorParams
    space: jax_md.space.Space
    transform_fn: Callable
    simulator_init: Callable[[Callable, Callable], jax_md.simulate.Simulator]
    neighbors: jaxmd_utils.NeighborHelper
    topology: jd_topology.Topology

    def __post_init__(self) -> None:
        """Builds the run function using the provided parameters."""
        self.run = build_run_fn(
            self.energy_configs,
            self.energy_fns,
            self.simulator_params,
            self.space,
            self.transform_fn,
            self.simulator_init,
            self.neighbors,
        )


def build_run_fn(
    energy_configs: list[jd_energy_cnfg.BaseConfiguration],
    energy_fns: list[jd_energy_fn.BaseEnergyFunction],
    simulator_params: jaxmd_utils.StaticSimulatorParams,
    space: jax_md.space.Space,
    transform_fn: Callable,
    simulator_init: Callable[[Callable, Callable], jax_md.simulate.Simulator],
    neighbors: jaxmd_utils.NeighborHelper,
) -> Callable[[dict[str, float], jax_md.rigid_body.RigidBody, int, jax.random.PRNGKey], jd_traj.Trajectory]:
    """Builds the run function for the jax_md simulation."""
    displacement_fn, shift_fn = space
    scan_fn = (
        jax.lax.scan
        if simulator_params.checkpoint_every <= 0
        else functools.partial(jaxmd_utils.checkpoint_scan, checkpoint_every=simulator_params.checkpoint_every)
    )

    def run_fn(
        opt_params: jd_types.Params,
        init_state: jax_md.rigid_body.RigidBody,
        n_steps: int,
        key: jax.random.PRNGKey,
    ) -> jd_sio.SimulatorTrajectory:
        # The  energy function configuration init calls need to happen inside the function
        # so that if the gradient is calculated for this run it will be tracked

        energy_fn = jdna_energy.energy_fn_builder(
            energy_fns=energy_fns,
            energy_configs=energy_configs,
            transform_fn=transform_fn,
            displacement_fn=displacement_fn,
        )(opt_params)

        init_fn, step_fn = simulator_init(energy_fn, shift_fn, **simulator_params.sim_init_fn)

        init_state = init_fn(
            key=key,
            R=init_state,
            unbonded_neighbors=neighbors.idx.T,
            **simulator_params.init_fn,
        )

        def apply_fn(in_state: SIM_STATE, _: int) -> tuple[SIM_STATE, jax_md.rigid_body.RigidBody]:
            state, neighbors = in_state
            state = step_fn(
                state,
                unbonded_neighbors=neighbors.idx.T,
                **simulator_params.step_fn,
            )

            neighbors = neighbors.update(state.position.center)

            return (state, neighbors), state.position

        _, trajectory = scan_fn(jax.jit(apply_fn), (init_state, neighbors), jnp.arange(n_steps))

        return jd_sio.SimulatorTrajectory(rigid_body=trajectory)

    return run_fn
