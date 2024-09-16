"""Base classes for energy functions."""

from collections.abc import Callable
from typing import Any, Union

import chex
import jax.numpy as jnp
import jax_md

import jax_dna.utils.types as typ

ERR_PARAM_NOT_FOUND = "Parameter '{key}' not found in {class_name}"
ERR_CALL_NOT_IMPLEMENTED = "Subclasses must implement this method"
ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH = "Weights must have the same length as energy functions"
ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS = "energy_fns must be a list of energy functions"


@chex.dataclass(frozen=True)
class BaseEnergyFunction:
    """Base class for energy functions.

    This class should not be used directly. Subclasses should implement the __call__ method.

    Attributes:
        displacement_fn (Callable): an instance of a displacement function from jax_md.space
    """

    displacement_fn: Callable

    @property
    def displacement_mapped(self) -> Callable:
        """Returns the displacement function mapped to the space."""
        return jax_md.space.map_bond(self.displacement_fn)

    def __add__(self, other: "BaseEnergyFunction") -> "ComposedEnergyFunction":
        """Add two energy functions together to create a ComposedEnergyFunction."""
        if not isinstance(other, BaseEnergyFunction):
            return NotImplemented

        return ComposedEnergyFunction(energy_fns=[self, other])

    def __mul__(self, other: float) -> "ComposedEnergyFunction":
        """Multiply an energy function by a scalar to create a ComposedEnergyFunction."""
        if not isinstance(other, float | int):
            return NotImplemented

        return ComposedEnergyFunction(
            energy_fns=[self],
            weights=jnp.array([other], dtype=float),
        )

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbounded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> float:
        """Calculate the energy of the system."""
        raise NotImplementedError(ERR_CALL_NOT_IMPLEMENTED)


@chex.dataclass(frozen=True)
class ComposedEnergyFunction:
    """Represents a linear combination of energy functions.

    Attributes:
        energy_fns (list[BaseEnergyFunction]): a list of energy functions
        weights (jnp.ndarray): optional, the weights of the energy functions
        rigid_body_transform_fn (Callable): a function to transform the rigid body
        to into something that can be used by the energy functions like a DNA1 nucleotide
    """

    energy_fns: tuple[BaseEnergyFunction]
    weights: jnp.ndarray | None = None
    rigid_body_transform_fn: Callable[[jax_md.rigid_body.RigidBody], Any] | None = None

    def __post_init__(self) -> None:
        """Check that the input is valid."""
        if not isinstance(self.energy_fns, list) or not all(
            isinstance(fn, BaseEnergyFunction) for fn in self.energy_fns
        ):
            print([isinstance(fn, BaseEnergyFunction) for fn in self.energy_fns])
            raise TypeError(ERR_COMPOSED_ENERGY_FN_TYPE_ENERGY_FNS)

        if self.weights is not None and len(self.weights) != len(self.energy_fns):
            raise ValueError(ERR_COMPOSED_ENERGY_FN_LEN_MISMATCH)


    def compute_terms(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neighbors: typ.Arr_Bonded_Neighbors,
        unbonded_neighbors: typ.Arr_Unbonded_Neighbors,
    ) -> tuple[jnp.ndarray, ...]:
        if self.rigid_body_transform_fn:
            body = self.rigid_body_transform_fn(body)

        return jnp.array([fn(body, seq, bonded_neighbors, unbonded_neighbors) for fn in self.energy_fns])

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
        energy_vals = self.compute_terms(body, seq, bonded_neighbors, unbonded_neighbors)
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
            energy_fns=[*self.energy_fns, energy_fn],
            weights=weights,
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
            weights = jnp.concatenate([self.weights, other_weights])
        else:
            this_weights = self.weights if not w_none else jnp.ones(len(energy_fn.energy_fns))
            other_weights = other_weights if not ow_none else jnp.ones(len(self.energy_fns))
            weights = jnp.concatenate([this_weights, other_weights])

        return ComposedEnergyFunction(
            energy_fns=self.energy_fns + energy_fn.energy_fns,
            weights=weights,
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
            return NotImplemented

        return energy_fn(other)

    def __radd__(self, other: Union[BaseEnergyFunction, "ComposedEnergyFunction"]) -> "ComposedEnergyFunction":
        """Create a new ComposedEnergyFunction by adding another energy function.

        This is a convenience method for the add_energy_fn and add_composable_energy_fn methods.
        """
        return self.__add__(other)
