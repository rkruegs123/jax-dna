"""Utilities for JAX-MD samplers."""

from collections.abc import Callable
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from typing_extensions import override

import jax_dna.input.topology as jdna_top

ERR_CHKPNT_SCN = "`checkpoint_every` must evenly divide the length of `xs`. Got {} and {}."


class SimulationState(Protocol):
    """This is a protocol to help with typing.

    Every state implements at least position and mass. More info about the specific states can be found here:

    https://github.com/jax-md/jax-md/blob/main/jax_md/simulate.py
    """

    position: jax_md.rigid_body.RigidBody
    mass: jax_md.rigid_body.RigidBody


class NeighborHelper(Protocol):
    """Helper class for managing neighbor lists."""

    @property
    def idx(self) -> jnp.ndarray:
        """Return the indices of the unbonded neighbors."""
        ...

    def allocate(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper":
        """Allocate memory for the neighbor list."""
        ...

    def update(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborHelper":
        """Update the neighbor list."""
        ...


@chex.dataclass
class NoNeighborList(NeighborHelper):
    """A dummy neighbor list that does nothing."""

    unbonded_nbrs: jnp.ndarray

    @property
    def idx(self) -> jnp.ndarray:
        """Return the indices of the unbonded neighbors."""
        return self.unbonded_nbrs

    @override
    def allocate(self, locs: jax_md.rigid_body.RigidBody) -> "NoNeighborList":
        """Allocate memory for the neighbor list."""
        return self

    @override
    def update(self, locs: jax_md.rigid_body.RigidBody) -> "NoNeighborList":
        """Update the neighbor list."""
        return self


@chex.dataclass
class NeighborList(NeighborHelper):
    """Neighbor list for managing unbonded neighbors."""

    displacement_fn: Callable
    topology: jdna_top.Topology
    r_cutoff: float
    dr_threshold: float
    box_size: jnp.ndarray
    init_positions: jax_md.rigid_body.RigidBody

    def __post_init__(self) -> None:
        """Initialize the neighbor list."""
        dense_mask = np.full((self.topology.n_nucleotides, 2), self.topology.n_nucleotides, dtype=np.int32)
        counter = np.zeros(self.topology.n_nucleotides, dtype=np.int32)
        for i, j in self.topology.bonded_nbrs:
            dense_mask[i, counter[i]] = j
            counter[i] += 1
            dense_mask[j, counter[j]] = i
            counter[j] += 1
        self.dense_mask = jnp.array(dense_mask, dtype=jnp.int32)

        def bonded_nbrs_mask_fn(dense_idx: int) -> jnp.ndarray:
            nbr_mask1 = dense_idx == dense_mask[:, 0].reshape(self.topology.n_nucleotides, 1)
            dense_idx = jnp.where(nbr_mask1, self.topology.n_nucleotides, dense_idx)

            nbr_mask2 = dense_idx == dense_mask[:, 1].reshape(self.topology.n_nucleotides, 1)
            return jnp.where(nbr_mask2, self.topology.n_nucleotides, dense_idx)

        self._neighborlist_fn = jax_md.partition.neighbor_list(
            self.displacement_fn,
            box_size=self.box_size,
            r_cutoff=self.r_cutoff,
            dr_threshold=self.dr_threshold,
            custom_mask_function=bonded_nbrs_mask_fn,
            disable_cell_list=True,
            format=jax_md.partition.NeighborListFormat.OrderedSparse,
        )

        self._neighbors = self._neighborlist_fn.allocate(self.init_positions.center)

    @property
    def idx(self) -> jnp.ndarray:
        """Return the indices of the unbonded neighbors."""
        return self._neighbors.idx

    @override
    def allocate(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborList":
        """Allocate memory for the neighbor list."""
        self._neighbors = self._neighborlist_fn.allocate(locs)
        return self

    @override
    def update(self, locs: jax_md.rigid_body.RigidBody) -> "NeighborList":
        """Update the neighbor list."""
        self._neighbors = self._neighbors.update(locs.position.center)
        return self._neighbors


@chex.dataclass
class StaticSimulatorParams:
    """Static parameters for the simulator."""

    # this is commonly referred to as `init_body` in the code, but the argument is named `R` in jax_md
    seq: jnp.ndarray
    mass: jax_md.rigid_body.RigidBody
    gamma: jax_md.rigid_body.RigidBody
    bonded_neighbors: jnp.ndarray
    n_steps: int
    checkpoint_every: int
    dt: float
    kT: float  # noqa: N815, the variable is commonly referred to using this casing

    @property
    def sim_init_fn(self) -> Callable:
        """Return the simulator init function."""
        return {
            "dt": self.dt,
            "kT": self.kT,
            "gamma": self.gamma,
        }

    @property
    def init_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        """Return the kwargs for initial state of the simulator."""
        return {
            "seq": self.seq,
            "mass": self.mass,
            "bonded_neighbors": self.bonded_neighbors,
        }

    @property
    def step_fn(self) -> dict[str, jax_md.rigid_body.RigidBody | jnp.ndarray]:
        """Return the kwargs for the step_fn of the simulator."""
        return {
            "seq": self.seq,
            "bonded_neighbors": self.bonded_neighbors,
        }


def split_and_stack(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Split `xs` into `n` pieces and stack them."""
    return jax.tree_map(lambda y: jnp.stack(jnp.split(y, n)), x)


def flatten_n(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """Flatten `x` by `n` levels."""
    # setting n <= 1 does not achieve the desired effect
    chex.assert_scalar_positive(n - 1)
    return jax.tree_map(lambda y: jnp.reshape(y, (-1,) + y.shape[n:]), x)


def checkpoint_scan(
    f: Callable, init: jax_md.rigid_body.RigidBody, xs: jnp.ndarray, checkpoint_every: int
) -> tuple[jax_md.rigid_body.RigidBody, jnp.ndarray]:
    """Replicates the behavior of `jax.lax.scan` but checkpoints gradients every `checkpoint_every` steps."""
    flat_xs, _ = jax.tree_util.tree_flatten(xs)
    length = flat_xs[0].shape[0]
    outer_iterations, residual = divmod(length, checkpoint_every)
    if residual:
        raise ValueError(ERR_CHKPNT_SCN.format(checkpoint_every, length))
    reshaped_xs = split_and_stack(xs, outer_iterations)

    @jax.checkpoint
    def inner_loop(
        _init: jax_md.rigid_body.RigidBody, _xs: jnp.ndarray
    ) -> tuple[jax_md.rigid_body.RigidBody, jnp.ndarray]:
        return jax.lax.scan(f, _init, _xs)

    final, result = jax.lax.scan(inner_loop, init, reshaped_xs)

    return final, flatten_n(result, 2)
