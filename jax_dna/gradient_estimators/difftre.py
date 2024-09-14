"""Optimization using the DiffTRe method.

DiffTRe: https://www.nature.com/articles/s41467-021-27241-4
"""

import functools
import time
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax_md

import jax_dna.energy.base as jd_energy_fn
import jax_dna.energy.configuration as jd_energy_cnfg
import jax_dna.input.topology as jd_topology
import jax_dna.input.trajectory as jd_traj
import jax_dna.simulators.io as jd_sio
import jax_dna.utils.types as jd_types


@chex.dataclass(frozen=True)
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
    key: jax.random.PRNGKey
    init_state: jax_md.rigid_body.RigidBody
    observables: tuple[Callable] | None = None
    slicer: slice | None = None
    trajectory: jd_traj.Trajectory | None = None
    ref_states: tuple[jd_traj.NucleotideState, ...] | None = None
    ref_energies: jnp.ndarray | None = None

    @property
    def displacement_fn(self) -> Callable:
        return self.space[0]

    def intialize(self, opt_params: list[dict[str, float]]) -> "DiffTRe":
        if self.ref_states is None:
            key, split = jax.random.split(self.key)
            trajectory, ref_states, ref_energies = self.compute_states_energies(
                opt_params,
                split,
            )
        else:
            ref_states = self.ref_states
            ref_energies = self.ref_energies
            trajectory = self.trajectory

        return self.replace(
            ref_states=ref_states,
            ref_energies=ref_energies,
            trajectory=trajectory,
            key=key,
            slicer=slice(self.n_eq_steps, None, self.sample_every),
        )

    def build_energy_fn(
        self,
        opt_params: list[dict[str, float]],
    ) -> jd_energy_fn.ComposedEnergyFunction:
        initialized_energy_fns = [
            e_fn(
                displacement_fn=self.displacement_fn,
                params=(e_c | op).init_params(),
            )
            for op, e_fn, e_c in zip(opt_params, self.energy_fns, self.energy_configs)
        ]

        energy_fn = jd_energy_fn.ComposedEnergyFunction(
            energy_fns=initialized_energy_fns,
            rigid_body_transform_fn=self.rigid_body_transform_fn,
        )

        # The unvmapped version of this function operates on a single rigid body
        # with rigid_body.center \in R^nx3 and rigid_body.orientation \in R^nx4
        # The vmap version of this function operates on a batch of rigid bodies
        # with rigid_body.center \in R^bxnx3 and rigid_body.orientation \in R^bxnx4
        return jax.vmap(
            lambda rigid_body: energy_fn(
                rigid_body,
                seq=jnp.array(self.topology.seq_one_hot),
                bonded_neighbors=self.topology.bonded_neighbors,
                unbonded_neighbors=self.topology.unbonded_neighbors.T,
            )
        )

    def compute_states_energies(
        self,
        params: list[dict[str, float]],
        key: jax.random.PRNGKey,
    ) -> tuple[jd_sio.SimulatorTrajectory, jnp.ndarray, jnp.ndarray]:
        # run sim get states

        print("Setting up simulation")
        trajectory = self.sim_init_fn(
            energy_configs=self.energy_configs,
            energy_fns=self.energy_fns,
        )
        print("Running for n steps", self.n_sim_steps, "with init state", self.init_state.center.shape)
        start = time.time()
        # TODO(ryanhausen): add batch support with pmap/vmap here
        trajectory = trajectory.run(
            opt_params=params,
            # TODO(ryanhausen): why -1 here and not zero?
            # source: https://github.com/ssec-jhu/jax-dna/blob/addc621dc61f212029a0ab9aa4761dedc3f5513e/experiments/optimize_structural_difftre.py#L237
            init_state=self.init_state,
            n_steps=self.n_sim_steps,
            key=key,
        )
        print("run took", time.time() - start)

        print("slicing")
        trajectory.slice(self.slicer or slice(self.n_eq_steps, None, self.sample_every))

        #  calculate
        ref_states = trajectory.rigid_body[self.slicer or slice(self.n_eq_steps, None, self.sample_every)]
        ref_energies = self.build_energy_fn(params)(ref_states)

        return trajectory, ref_states, ref_energies

    def compute_loss(
        self,
        opt_params: list[dict[str, float]],
        loss_fns: tuple[Callable, ...],
        trajectory: jd_traj.Trajectory,
        ref_states: tuple[jd_traj.NucleotideState, ...],
        ref_energies: jnp.ndarray,
    ) -> tuple[float, int]:
        new_energies = self.build_energy_fn(opt_params)(ref_states)
        diffs = new_energies - ref_energies
        boltz = jnp.exp(-self.beta * diffs)
        weights = boltz / jnp.sum(boltz)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))

        losses = jax.tree_util.tree_map(
            lambda l: l(trajectory=trajectory.slice(self.slicer), weights=weights),
            loss_fns,
        )
        # print("Computed losses", losses)
        # TODO(ryanhausen): this is simple mean, but could be more sophisticated
        # so we should consider how to let the user inject this logic
        return jnp.mean(jnp.array(losses)), (n_eff, losses)

    def __call__(
        self,
        opt_params: list[dict[str, float]],
        loss_fns: Callable,
        key: jax.random.PRNGKey,
    ) -> tuple["DiffTRe", list[dict[str, float]], tuple[float, ...], tuple[float, ...]]:
        # this should be jitted too maybe a decorator?

        (loss, (n_eff, losses)), grads = jax.value_and_grad(self.compute_loss, has_aux=True)(
            opt_params,
            loss_fns,
            self.trajectory,
            self.ref_states,
            self.ref_energies,
        )

        # if n_eff is greater than the threshold we don't need to recompute the
        # reference states and energies.
        if n_eff >= self.min_n_eff:
            new_obj = self
            observable = None
        else:
            key, split = jax.random.split(key)
            trajectory, ref_states, ref_energies = self.compute_states_energies(opt_params, split)

            (loss, (n_eff, losses)), grads = jax.value_and_grad(self.compute_loss, has_aux=True)(
                opt_params, loss_fns, trajectory, ref_states, ref_energies
            )

            new_obj = self.replace(ref_states=ref_states, ref_energies=ref_energies, trajectory=trajectory, key=key)

        return new_obj, grads, loss, losses, observable
