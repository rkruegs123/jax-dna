"""Base classes for energy functions."""

import dataclasses as dc
from typing import Any, Union

import chex
import jax.numpy as jnp
import jax_md
from _collections_abc import Callable

import jax_dna.utils.types as typ

ERR_PARAM_NOT_FOUND = "Parameter '{key}' not found in {class_name}"
ERR_CALL_NOT_IMPLEMENTED = "Subclasses must implement this method"
ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH = "Weights must have the same length as energy functions"
ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS = "energy_fns must be a list of energy functions"
ERR_UNSUPPORTED_OPERATION = "unsupported operand type(s) for {op}: '{left}' and '{right}'"


@chex.dataclass(frozen=True)
class BaseEnergyFunction:
    """Base class for energy functions.

    This class should not be used directly. Subclasses should implement the __call__ method.

    Attributes:
        displacement_fn (Callable): an instannce of a displacement function from jax_md.space
    """

    displacement_fn: Callable

    @property
    def displacement_mapped(self) -> Callable:
        """Returns the displacement function mapped to the space."""
        return jax_md.space.map_bond(self.displacement_fn)

    def __add__(self, other: "BaseEnergyFunction") -> "ComposedEnergyFunction":
        """Add two energy functions together to create a ComposedEnergyFunction."""
        if not isinstance(other, BaseEnergyFunction):
            raise TypeError(ERR_UNSUPPORTED_OPERATION.format(op="+", left=type(self), right=type(other)))

        return ComposedEnergyFunction([self, other])

    def __mul__(self, other: float) -> "ComposedEnergyFunction":
        """Multiply an energy function by a scalar to create a ComposedEnergyFunction."""
        if not isinstance(other, float | int):
            raise TypeError(ERR_UNSUPPORTED_OPERATION.format(op="*", left=type(self), right=type(other)))

        return ComposedEnergyFunction([self], jnp.array([other], dtype=float))

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbounded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> float:
        """Calculate the energy of the system."""
        raise NotImplementedError(ERR_CALL_NOT_IMPLEMENTED)


@dc.dataclass(frozen=True)
class ComposedEnergyFunction:
    """Represents a linear combination of energy functions.

    Attributes:
        energy_fns (list[BaseEnergyFunction]): a list of energy functions
        weights (jnp.ndarray): optional, the weights of the energy functions
        rigid_body_transform_fn (Callable): a function to transform the rigid body
        to into something that can be used by the energy functions like a DNA1 nucleotide
    """

    energy_fns: list[BaseEnergyFunction]
    weights: jnp.ndarray | None = None
    rigid_body_transform_fn: Callable[[jax_md.rigid_body.RigidBody], Any] | None = None

    def __post_init__(self) -> None:
        """Check that the input is valid."""
        if not isinstance(self.energy_fns, list) or not all(
            isinstance(fn, BaseEnergyFunction) for fn in self.energy_fns
        ):
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
        """Calculates the energy of the system using the all of the function in `energy_fns`.

        Args:
            body (jax_md.rigid_body.RigidBody): The rigid body(ies) of the system
            seq (jnp.ndarray): the sequence of the system
            bonded_neighbors (typ.Arr_Bonded_Neighbors): the bonded neighbors
            unbonded_neighbors (typ.Arr_Unbonded_Neighbors): the unbonded neighbors

        Returns:
            float: the energy of the system
        """
        if self.rigid_body_transform_fn:
            body = self.rigid_body_transform_fn(body)

        energy_vals = jnp.array([fn(body, seq, bonded_neighbors, unbonded_neighbors) for fn in self.energy_fns])
        return jnp.sum(energy_vals) if self.weights is None else jnp.dot(self.weights, energy_vals)

    def add_energy_fn(self, energy_fn: BaseEnergyFunction, weight: float = 1.0) -> "ComposedEnergyFunction":
        """Add an energy function to the list of energy functions.

        Args:
            energy_fn (BaseEnergyFunction): the energy function to add
            weight (float): the weight of the energy function

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction with the added energy function
        """
        if self.weights is None:
            weights = None if weight == 1.0 else jnp.array([1.0] * len(self.energy_fns) + [weight])
        else:
            weights = jnp.concatenate([self.weights, jnp.array([weight])])

        return ComposedEnergyFunction(
            [*self.energy_fns, energy_fn],
            weights,
        )

    def add_composable_energy_fn(self, energy_fn: "ComposedEnergyFunction") -> "ComposedEnergyFunction":
        """Add a ComposedEnergyFunction to the list of energy functions.

        Args:
            energy_fn (ComposedEnergyFunction): the ComposedEnergyFunction to add

        Returns:
            ComposedEnergyFunction: a new ComposedEnergyFunction with the added energy function
        """
        other_weights = energy_fn.weights
        w_none = self.weights is None
        ow_none = other_weights is None
        if w_none and ow_none:
            weights = None
        elif not w_none and not ow_none:
            weights = jnp.concatenate([self.weights, jnp.ones(len(energy_fn.energy_fns))])
        else:
            this_weights = self.weights if not w_none else jnp.ones(len(energy_fn.energy_fns))
            other_weights = other_weights if not ow_none else jnp.ones(len(self.energy_fns))
            weights = jnp.concatenate([this_weights, other_weights])

        return ComposedEnergyFunction(
            self.energy_fns + energy_fn.energy_fns,
            weights,
        )

    def __add__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction by adding another energy function.

        This is a convenience method for the add_energy_fn and add_composable_energy_fn methods.
        """
        if isinstance(other, BaseEnergyFunction):
            energy_fn = self.add_energy_fn
        elif isinstance(other, ComposedEnergyFunction):
            energy_fn = self.add_composable_energy_fn
        else:
            raise TypeError(ERR_UNSUPPORTED_OPERATION.format(op="+", left=type(self), right=type(other)))

        return energy_fn(other)

    def __radd__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction by adding another energy function.

        This is a convenience method for the add_energy_fn and add_composable_energy_fn methods.
        """
        return self.__add__(other)
