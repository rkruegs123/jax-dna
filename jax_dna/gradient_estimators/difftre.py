"""Optimization using the DiffTRe method.

DiffTRe: https://www.nature.com/articles/s41467-021-27241-4
"""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as jd_energy_fn
import jax_dna.energy.configuration as jd_energy_cnfg
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types


def build_energy_function(
    opt_params: list[dict[str, float]],
    displacement_fn: Callable,
    energy_fns: tuple[jd_energy_fn.BaseEnergyFunction],
    energy_configs: tuple[jd_energy_cnfg.BaseConfiguration],
    rigid_body_transform_fn: Callable,
    seq: jd_types.Sequence,  # make sure this is jax
    bonded_neighbors: jnp.ndarray,
    unbonded_neighbors: jnp.ndarray,
) -> jd_energy_fn.ComposedEnergyFunction:
    """Builds the energy function for the given parameters."""
    initialized_energy_fns = [
        e_fn(
            displacement_fn=displacement_fn,
            params=(e_c | op).init_params(),
        )
        for op, e_fn, e_c in zip(opt_params, energy_fns, energy_configs, strict=True)
    ]

    energy_fn = jd_energy_fn.ComposedEnergyFunction(
        energy_fns=initialized_energy_fns,
        rigid_body_transform_fn=rigid_body_transform_fn,
    )

    # The unvmapped version of this function operates on a single rigid body
    # with rigid_body.center \in R^nx3 and rigid_body.orientation \in R^nx4
    # The vmap version of this function operates on a batch of rigid bodies
    # with rigid_body.center \in R^bxnx3 and rigid_body.orientation \in R^bxnx4
    return jax.vmap(
        lambda rigid_body: energy_fn(
            rigid_body,
            seq=seq,
            bonded_neighbors=bonded_neighbors,
            unbonded_neighbors=unbonded_neighbors,
        )
    )


def _compute_states_energies(
    params: list[dict[str, float]],
    key: jax.random.PRNGKey,
    sim_init_fn: Callable,
    energy_configs: tuple[jd_energy_cnfg.BaseConfiguration],
    energy_fns: tuple[jd_energy_fn.BaseEnergyFunction],
    init_state: jax_md.rigid_body.RigidBody,
    n_steps: jd_types.Scalar,
    n_eq_steps: jd_types.Scalar,
    sample_every: jd_types.Scalar,
    energy_fn_builder: Callable[[list[dict[str, float]]], Callable],
) -> tuple[jd_sio.SimulatorTrajectory, jax_md.rigid_body.RigidBody, jnp.ndarray]:
    """Calls the simulation function to get the states and energies.

    Don't try to JIT this function, not all sim functions are JIT-able. Instead
    JIT sim_init and run.

    """
    trajectory = sim_init_fn(
        energy_configs=energy_configs,
        energy_fns=energy_fns,
    )
    trajectory = trajectory.run(
        opt_params=params,
        init_state=init_state,
        n_steps=n_steps,
        key=key,
    )
    trajectory = trajectory.slice(slice(n_eq_steps, None, sample_every))

    ref_states = trajectory.rigid_body
    ref_energies = energy_fn_builder(params)(ref_states)

    return trajectory, ref_states, ref_energies


def _compute_loss(
    opt_params: list[dict[str, float]],
    energy_fn_builder: Callable[[dict[str, float]], Callable],
    beta: float,
    loss_fns: tuple[Callable],
    trajectory: jd_sio.SimulatorTrajectory,
    ref_states: jax_md.rigid_body.RigidBody,
    ref_energies: jnp.ndarray,
    losses_reduce_fn: Callable = jnp.mean,
) -> tuple[float, int]:
    new_energies = energy_fn_builder(opt_params)(ref_states)
    diffs = new_energies - ref_energies
    boltz = jnp.exp(-beta * diffs)
    weights = boltz / jnp.sum(boltz)
    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
    losses = jax.tree_util.tree_map(
        lambda loss_fn: loss_fn(trajectory=trajectory, weights=weights),
        loss_fns,
    )
    losses = jnp.atleast_2d(jnp.array(losses).T).T
    return losses_reduce_fn(losses[:, 0]), (n_eff, losses)


_loss_w_grads = jax.value_and_grad(_compute_loss, has_aux=True)


@chex.dataclass(frozen=True)
class DiffTRe:
    """DiffTRe optimizer."""

    energy_fn_builder: Callable[[list[dict[str, jd_types.ARR_OR_SCALAR]]], Callable]
    beta: jd_types.Scalar
    min_n_eff: jd_types.Scalar
    loss_fns: tuple[Callable]
    losses_reduce_fn: Callable
    sim_init_fn: Callable
    energy_configs: tuple[jd_energy_cnfg.BaseConfiguration]
    energy_fns: tuple[jd_energy_fn.BaseEnergyFunction]
    init_state: jax_md.rigid_body.RigidBody
    n_steps: jd_types.Scalar
    n_eq_steps: jd_types.Scalar
    sample_every: jd_types.Scalar
    _trajectory: jd_sio.SimulatorTrajectory | None = None
    _ref_states: jax_md.rigid_body.RigidBody | None = None
    _ref_energies: jnp.ndarray | None = None

    def initialize(self, opt_params: list[dict[str, float]], key: jax.random.PRNGKey) -> "DiffTRe":
        """Initialize the reference states and energies."""
        new_obj = self
        if self._ref_states is None:
            _, split = jax.random.split(key)
            trajectory, ref_states, ref_energies = _compute_states_energies(
                opt_params,
                split,
                self.sim_init_fn,
                self.energy_configs,
                self.energy_fns,
                self.init_state,
                self.n_steps,
                self.n_eq_steps,
                self.sample_every,
                self.energy_fn_builder,
            )

            new_obj = self.replace(
                _ref_states=ref_states,
                _ref_energies=ref_energies,
                _trajectory=trajectory,
            )
        return new_obj

    def __call__(
        self,
        opt_params: list[dict[str, float]],
        key: jax.random.PRNGKey,
    ) -> tuple["DiffTRe", list[dict[str, float]], float, tuple[float, ...]]:
        """Compute the loss and gradients for the given parameters."""
        (loss, (n_eff, losses)), grads = _loss_w_grads(
            opt_params,
            self.energy_fn_builder,
            self.beta,
            self.loss_fns,
            self._trajectory,
            self._ref_states,
            self._ref_energies,
            self.losses_reduce_fn,
        )

        new_obj = self
        regenerate_trajectory = n_eff < self.min_n_eff
        # if n_eff is greater than the threshold we don't need to recompute the
        # reference states and energies.
        if regenerate_trajectory:
            key, split = jax.random.split(key)
            trajectory, ref_states, ref_energies = _compute_states_energies(
                opt_params,
                split,
                self.sim_init_fn,
                self.energy_configs,
                self.energy_fns,
                self.init_state,
                self.n_steps,
                self.n_eq_steps,
                self.sample_every,
                self.energy_fn_builder,
            )

            (loss, (n_eff, losses)), grads = _loss_w_grads(
                opt_params,
                self.energy_fn_builder,
                self.beta,
                self.loss_fns,
                self._trajectory,
                self._ref_states,
                self._ref_energies,
                self.losses_reduce_fn,
            )

            new_obj = self.replace(
                _ref_states=ref_states,
                _ref_energies=ref_energies,
                _trajectory=trajectory,
            )

        return new_obj, grads, loss, losses, regenerate_trajectory


if __name__ == "__main__":
    pass
