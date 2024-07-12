import dataclasses as dc
import functools
from typing import Any, Callable, Union

import chex
import jax.numpy as jnp
import jax_md

import jax_dna.input.configuration as config
import jax_dna.utils.types as typ

ERR_PARAM_NOT_FOUND = "Parameter '{key}' not found in {class_name}"
ERR_CALL_NOT_IMPLEMENTED = "Subclasses must implement this method"
ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH = "Weights must have the same length as energy functions"
ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS = "energy_fns must be a list of energy functions"
ERR_UNSUPPORTED_OPERATION = "unsupported operand type(s) for {op}: '{left}' and '{right}'"


@chex.dataclass(frozen=True)
class BaseEnergyFunction:
    displacement_fn: Callable

    @property
    def displacement_mapped(self):
        return jax_md.space.map_bond(self.displacement_fn)

    def __add__(self, other: "BaseEnergyFunction") -> "ComposedEnergyFunction":
        if isinstance(other, BaseEnergyFunction):
            return ComposedEnergyFunction([self, other])
        else:
            raise TypeError(ERR_UNSUPPORTED_OPERATION.format(op="+", left=type(self), right=type(other)))

    def __mul__(self, other: float) -> "ComposedEnergyFunction":
        if isinstance(other, float) or isinstance(other, int):
            return ComposedEnergyFunction([self], jnp.array([other], dtype=float))
        else:
            raise TypeError(ERR_UNSUPPORTED_OPERATION.format(op="*", left=type(self), right=type(other)))

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbounded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> float:
        raise NotImplementedError(ERR_CALL_NOT_IMPLEMENTED)


@dc.dataclass(frozen=True)
class ComposedEnergyFunction:
    energy_fns: list[BaseEnergyFunction]
    weights: jnp.ndarray | None = None
    rigid_body_transform_fn: Callable[[jax_md.rigid_body.RigidBody], Any] | None = None

    def __post_init__(self):
        if type(self.energy_fns) != list or not all(isinstance(fn, BaseEnergyFunction) for fn in self.energy_fns):
            raise TypeError(ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS)

        if self.weights is not None and len(self.weights) != len(self.energy_fns):
            raise ValueError(ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH)

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> float:
        if self.rigid_body_transform_fn:
            body = self.rigid_body_transform_fn(body)

        energy_vals = jnp.array([fn(body, seq, bonded_neighbors, unbonded_neighbors) for fn in self.energy_fns])
        if self.weights is None:
            return jnp.sum(energy_vals)
        else:
            return jnp.dot(self.weights, energy_vals)

    def add_energy_fn(self, energy_fn: BaseEnergyFunction, weight: float = 1.0) -> "ComposedEnergyFunction":
        if self.weights is None:
            if weight == 1.0:
                weights = None
            else:
                weights = jnp.array([1.0] * len(self.energy_fns) + [weight])
        else:
            weights = jnp.concatenate([self.weights, jnp.array([weight])])

        return ComposedEnergyFunction(
            self.energy_fns + [energy_fn],
            weights,
        )

    def add_composable_energy_fn(self, energy_fn: "ComposedEnergyFunction") -> "ComposedEnergyFunction":
        other_weights = energy_fn.weights
        weights = None
        if self.weights is None:
            if other_weights is None:
                weights = None
            else:
                weights = jnp.concatenate([jnp.ones(len(self.energy_fns)), other_weights])
        else:
            if other_weights is None:
                weights = jnp.concatenate([self.weights, jnp.ones(len(energy_fn.energy_fns))])
            else:
                weights = jnp.concatenate([self.weights, other_weights])

        return ComposedEnergyFunction(
            self.energy_fns + energy_fn.energy_fns,
            weights,
        )

    def __add__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        if isinstance(other, BaseEnergyFunction):
            return self.add_energy_fn(other)
        elif isinstance(other, ComposedEnergyFunction):
            return self.add_composable_energy_fn(other)

    def __radd__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        return self.__add__(other)
