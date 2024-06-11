import dataclasses as dc
import functools
from typing import Any, Callable

import jax.numpy as jnp
import jax_md

import jax_dna.utils.types as typ

ERR_CALL_NOT_IMPLEMENTED = "Subclasses must implement this method"


dc.dataclass(frozen=True)


class BaseEnergyFunction:
    displacement_fn: Callable
    params: dict[str, Any]
    opt_params: dict[str, Any]

    def __post_init__(self):
        self.displacement_mapped = jax_md.space.map_bond(functools.partial(self.displacement_fn))

    def __init__(
        self,
        displacement_fn: Callable,
        params: dict[str, Any],
        opt_params: dict[str, Any],
    ):
        self.displacement_fn = displacement_fn
        self.params = params
        self.opt_params = opt_params

        self.displacement_mapped = jax_md.space.map_bond(functools.partial(displacement_fn))

    def get_param(self, key: str) -> typ.Scalar:
        return self.opt_params.get(key, self.params[key])

    def get_params(self, keys: list[str]) -> dict[str, typ.Scalar]:
        return {key: self.get_param(key) for key in keys}

    def __call__(
        self,
        body: jax_md.rigid_body.RigidBody,
        seq: jnp.ndarray,
        bonded_neghbors: typ.Arr_Bonded_Neighbors,
        unbounded_neghbors: typ.Arr_Unbonded_Neighbors,
    ) -> float:
        raise NotImplementedError(ERR_CALL_NOT_IMPLEMENTED)
