"""Optimization using the DiffTRe method.

DiffTRe: https://www.nature.com/articles/s41467-021-27241-4
"""

from collections.abc import Callable
import functools
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as jd_energy_fn
import jax_dna.energy.configuration as jd_energy_cnfg
import jax_dna.input.topology as jd_topology
import jax_dna.input.trajectory as jd_traj



chex.dataclass(frozen=True, init=True)
class DiffTRe:
    beta: float
    n_eq_steps: int
    min_n_eff: int
    sample_every: int
    space: jax_md.space.Space
    topology: jd_topology.Topology
    rigid_body_transform_fn: Callable
    energy_configs: tuple[jd_energy_cnfg.BaseConfiguration]
    energy_fns: tuple[jd_energy_fn.BaseEnergyFunction]
    loss_fns: tuple[Callable]
    sim_init_fn: Callable
    n_sim_steps: int
    key:jax.random.PRNGKey
    ref_states: tuple[jd_traj.NucleotideState, ...] | None = None
    ref_eneriges: jnp.ndarray | None = None


    @property
    def displacement_fn(self) -> Callable:
        return self.space[0]


    def intialize(self, opt_params:list[dict[str,float]]) -> "DiffTRe":
        if self.ref_states is None:
            key, split = jax.random.split(self.key)
            ref_states, ref_energies = self.compute_states_energies(
                opt_params,
                split
            )
        else:
            ref_states = self.ref_states
            ref_energies = self.ref_energies

        return self.replace(
            ref_states=ref_states,
            ref_energies=ref_energies,
            key=key,
        )


    def build_energy_fn(
        self,
        opt_params: list[dict[str, float]],
    ) -> jd_energy_fn.ComposedEnergyFunction:
        initialized_energy_fns = [
            e_fn(
                displacement_fn=self.displacement_fn,
                params=e_c.init_params() | op,
            )
            for op, e_fn, e_c in zip(opt_params, self.energy_fns, self.energy_configs)
        ]

        energy_fn = jd_energy_fn.ComposedEnergyFunction(
            energy_fns=initialized_energy_fns,
            rigid_body_transform_fn=self.rigid_body_transform_fn,
        )

        return jax.vmap(lambda R: energy_fn(
            R,
            seq=self.topology.seq_one_hot,
            bonded_neighbors=self.topology.bonded_neighbors,
            unbonded_neighbors=self.topology.unbonded_neighbors.T,
        ))


    def compute_states_energies(
        self,
        params:list[dict[str, float]],
        trajectory: jd_traj.Trajectory,
        key: jax.random.PRNGKey,
    ) -> jnp.ndarray:

        # run sim get states
        ref_states = self.sim_init_fn(
            energy_configs=self.energy_configs,
            energy_fns=self.energy_fns,
        ).run(
            params,
            # TODO(ryanhausen): why -1 here and not zero?
            # source: https://github.com/ssec-jhu/jax-dna/blob/addc621dc61f212029a0ab9aa4761dedc3f5513e/experiments/optimize_structural_difftre.py#L237
            trajectory.states[-1].to_rigid_body(),
            self.n_steps,
            key,
        ).state_rigid_bodies[self.n_eq_steps::self.sample_every]
        # get the reference energies

        #  calculate
        ref_energies = self.build_energy_fn(
            params,
        )(ref_states)

        return ref_states, ref_energies


    def compute_loss(
        self,
        opt_params: list[dict[str, float]],
        ref_states: tuple[jd_traj.Trajectory, ...],
        ref_energies: jnp.ndarray,
        loss_fns: tuple[Callable, ...]
    ) -> tuple[float, int]:

        new_energies = self.build_energy_fn(opt_params)(ref_states)
        diffs = new_energies - ref_energies
        boltz = jnp.exp(-self.beta * diffs)
        weights = boltz / jnp.sum(boltz)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        losses = jax.tree_util.tree_map(
            lambda l: l(trajectory=ref_states, weights=weights),
            loss_fns,
        )

        # TODO(ryanhausen): this is simple mean, but could be more sophisticated
        # so we should consider how to let the user inject this logic
        return jnp.mean(losses), (n_eff, losses)


    def __call__(
        self,
        opt_params: list[dict[str, float]],
        loss_fns: Callable,
        key: jax.random.PRNGKey,
    ) -> tuple["DiffTRe", list[dict[str, float]], tuple[float,...], tuple[float,...]]:

        # this should be jitted too maybe a decorator?
        (loss, (n_eff, losses)), grads = jax.value_and_grad(self.compute_loss, has_aux=True)(
            opt_params,
            ref_states,
            ref_energies,
            loss_fns
        )

        # if n_eff id greater than the threshold we don't need to recompute the
        # reference states and energies.
        if n_eff >= self.min_n_eff:
            new_obj = self
        else:
            key, split = jax.random.split(key)
            ref_states, ref_energies = self.compute_states_energies(opt_params, ref_states[-1], split)
            (loss, (n_eff, losses)), grads = jax.value_and_grad(self.compute_loss, has_aux=True)(
                opt_params,
                ref_states,
                ref_energies,
                loss_fns
            )
            new_obj = self.replace(
                ref_states=ref_states,
                ref_energies=ref_energies,
                key=key,
            ), grads, loss, losses

        return new_obj, grads, loss, losses
