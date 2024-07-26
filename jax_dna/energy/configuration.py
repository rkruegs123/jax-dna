import dataclasses as dc
import warnings
from typing import Any, Union

import chex

import jax_dna.utils.types as jdt

ERR_INVALID_MERGE_TYPE = "Cannot merge {this_type} with {that_type}"
ERR_MISSING_REQUIRED_PARAMS = "Required properties {props} are not initialized."
ERR_OPT_DEPENDENT_PARAMS = "Optimized parameters cannot be dependent on other parameters. Only {req_params} permitted, but found {given_params}"
WARN_INIT_PARAMS_NOT_IMPLEMENTED = "init_params not implemented"


@chex.dataclass(frozen=True)
class BaseConfiguration:
    params_to_optimize: tuple[str] = ()
    required_params: tuple[str] = ()
    non_optimizable_required_params: tuple[str] = ()
    OPT_ALL: tuple[str] = ("*",)

    @property
    def opt_params(self) -> dict[str, jdt.Scalar]:
        if self.params_to_optimize == self.OPT_ALL:
            return {
                k: v
                for k, v in dc.asdict(self).items()
                if (k in self.required_params) and (k not in self.non_optimizable_required_params)
            }
        else:
            return {k: v for k, v in dc.asdict(self).items() if k in self.params_to_optimize}

    def __post_init__(self):
        non_initialized_props = [param for param in self.required_params if getattr(self, param) is None]
        if non_initialized_props:
            raise ValueError(ERR_MISSING_REQUIRED_PARAMS.format(props=",".join(non_initialized_props)))

        unoptimizable_params = set(self.params_to_optimize) - set(self.required_params)
        if unoptimizable_params and unoptimizable_params != set(self.OPT_ALL):
            raise ValueError(
                ERR_OPT_DEPENDENT_PARAMS.format(
                    req_params=",".join(self.required_params),
                    given_params=",".join(unoptimizable_params),
                )
            )

    def init_params(self) -> "BaseConfiguration":
        warnings.warn(WARN_INIT_PARAMS_NOT_IMPLEMENTED)
        return self

    @classmethod
    def from_dict(cls, params: dict[str, float], params_to_optimize: tuple[str] = ()) -> "BaseConfiguration":
        return cls(**(params | {"params_to_optimize": params_to_optimize}))

    def __merge__baseconfig(self, other: "BaseConfiguration") -> "BaseConfiguration":
        filtered = {k: v for k, v in dc.asdict(other).items() if v is not None}
        return self.__merge__dict(filtered)

    def __merge__dict(self, other: dict[str, Any]) -> "BaseConfiguration":
        return dc.replace(self, **other)

    # python doesn't like using the bar for type hints when inside the class, use Union for now
    def __or__(self, other: Union["BaseConfiguration", dict[str, jdt.ARR_OR_SCALAR]]) -> "BaseConfiguration":
        if isinstance(other, BaseConfiguration):
            return self.__merge__baseconfig(other)
        elif isinstance(other, dict):
            return self.__merge__dict(other)
        else:
            raise TypeError(ERR_INVALID_MERGE_TYPE.format(this_type=type(self), that_type=type(other)))
