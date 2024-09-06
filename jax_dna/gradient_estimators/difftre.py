"""Optimization using the DiffTRe method.

DiffTRe: https://www.nature.com/articles/s41467-021-27241-4
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as jd_energy_fn
import jax_dna.energy.configuration as jd_energy_cnfg
import jax_dna.input.topology as jd_topology
import jax_dna.input.trajectory as jd_traj


# TODO: sim_fn, should probably have an aliased type
def get_gradients(
    opt_params: list[dict[str, float]],
    energy_fns: list[jd_energy_fn.BaseEnergyFunction],
    energy_configs: list[jd_energy_cnfg.BaseConfiguration],
    rigid_body_transform_fn: Callable,
    loss_fns: Callable,
    topology: jd_topology.Topology,
    space: jax_md.space.Space,
    sim_init: Callable,
    sim_init_kwargs: dict[str, Any],
    n_eq_steps: int,
    sample_every: int,
    beta: float,
    n_effs: int,
) -> dict[str, float]:
    """Get the gradients for the current configuration."""
    displacement_fn = space[0]

    def compute_loss(opt_params: list[dict[str, float]]) -> float:
        simulator = sim_init(
            **sim_init_kwargs,
        )

        energy_fn = _build_energy_fn(
            opt_params,
            energy_fns,
            energy_configs,
            rigid_body_transform_fn,
            displacement_fn,
        )

        # how are unbonded neighors maintained for neighbor lists in the jaxmd simulator?
        # perhaps it doesn't matter because we calculate we only do this infrequently?
        energy_fn = lambda R: energy_fn(
            R,
            seq=topology.seq_one_hot,
            bonded_neighbors=topology.bonded_neighbors,
            unbonded_neighbors=topology.unbonded_neighbors.T,
        )
        energy_fn = jax.vmap(jax.jit(energy_fn))

        # TODO as ask the simulator to only return the states we care about rather than slicing
        trajectory = simulator.run(opt_params)
        energies = energy_fn(
            jnp.stack([state.to_rigid_body() for state in trajectory.states[n_eq_steps::sample_every]])
        )

        # initial trajectory and energies
        trajectory, energies

        loss_fns = [
            wrap_loss_fn(
                loss_fn=loss_fn, energy_fn=energy_fn, beta=beta, n_eq_steps=n_eq_steps, sample_every=sample_every
            )
            for loss_fn in loss_fns
        ]

        losses, n_effs = list(zip(*[loss_fn(trajectory, energies) for loss_fn in loss_fns]))

        # this is simple mean, but could be more sophisticated
        loss = jnp.mean(losses)

        return loss, n_effs


def _build_energy_fn(
    opt_params: list[dict[str, float]],
    energy_fns: list[jd_energy_fn.BaseEnergyFunction],
    energy_configs: list[jd_energy_cnfg.BaseConfiguration],
    rigid_body_transform_fn: Callable,
    displacement_fn: Callable,
) -> jd_energy_fn.ComposedEnergyFunction:
    initialized_energy_fns = [
        e_fn(
            displacement_fn=displacement_fn,
            params=e_c.init_params() | op,
        )
        for op, e_c, e_fn in zip(opt_params, energy_configs, energy_fns)
    ]

    return jd_energy_fn.ComposedEnergyFunction(
        energy_fns=initialized_energy_fns,
        rigid_body_transform_fn=rigid_body_transform_fn,
    )


def wrap_loss_fn(
    loss_fn: Callable,
    energy_fn: jd_energy_fn.ComposedEnergyFunction,
    beta: float,
    n_eq_steps: int,
    sample_every: int,
) -> Callable[[jd_traj.Trajectory, jnp.ndarray], tuple[float, float]]:
    """Wrap the loss function to include DiffTRe weights."""

    def wrapped_fn(trajectory: jd_traj.Trajectory, energies: jnp.ndarray) -> tuple[float, float]:
        new_energies = energy_fn(
            jnp.stack(jax.tree.map(lambda state: state.to_rigid_body(), trajectory.states[n_eq_steps::sample_every]))
        )
        diffs = new_energies - energies
        boltz = jnp.exp(-beta * diffs)
        weights = boltz / jnp.sum(boltz)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        return loss_fn(trajectory, weights), n_eff
