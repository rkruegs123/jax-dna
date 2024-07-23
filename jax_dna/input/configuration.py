import dataclasses as dc
import warnings
from typing import Any

import chex
import jax
import numpy as np
import sympy

import jax_dna.utils.types as jdt

# if we're using python >=3.11 then use native toml module, otherwise use toml from pypi
try:
    import tomllib as toml
except ImportError:
    # tox uses this parser, so we'll use it too
    import tomli as toml

SYMPY_EVAL_N: int = 64

ERR_MISSING_REQUIRED_PARAMS = "Required properties {props} are not initialized."
ERR_OPT_DEPENDENT_PARAMS = "Optimized parameters cannot be dependent on other parameters. Only {req_params} permitted"
ERR_INVALID_MERGE_TYPE = "Cannot merge {this_type} with {that_type}"
WARN_INIT_PARAMS_NOT_IMPLEMENTED = "init_params not implemented"


@chex.dataclass(frozen=True)
class BaseConfiguration:
    params_to_optimize: tuple[str] = ()
    required_params: tuple[str] = ()
    OPT_ALL: tuple[str] = ("*",)

    @property
    def opt_params(self) -> dict[str, jdt.Scalar]:
        if self.params_to_optimize == self.OPT_ALL:
            return dc.asdict(self)
        else:
            return {k: v for k, v in dc.asdict(self).items() if k in self.params_to_optimize}

    def __post_init__(self):
        non_initialized_props = [param for param in self.required_params if getattr(self, param) is None]
        if non_initialized_props:
            raise ValueError(ERR_MISSING_REQUIRED_PARAMS.format(props=",".join(non_initialized_props)))

        if not set(self.required_params).union(set(BaseConfiguration.OPT_ALL)).issubset(set(self.params_to_optimize)):
            raise ValueError(ERR_OPT_DEPENDENT_PARAMS.format(req_params=",".join(req_params=self.required_params)))

    def init_params(self) -> "BaseConfiguration":
        warnings.warn(WARN_INIT_PARAMS_NOT_IMPLEMENTED)
        return self

    @staticmethod
    def parse_toml(file_path: str, key: str | None = None) -> dict[str, Any]:
        with open(file_path, "rb") as f:
            config_dict = toml.load(f)

        if key in config_dict:
            config_dict = config_dict[key]

        return jax.tree.map(BaseConfiguration.parse_value, config_dict)

    @staticmethod
    def parse_value(value: str | float):
        if isinstance(value, str):
            return BaseConfiguration.parse_str(value)
        else:
            return value

    @staticmethod
    def parse_str(value: str) -> str | float:
        try:
            return float(value)
        except ValueError:
            try:
                return float(sympy.parse_expr(value).evalf(n=SYMPY_EVAL_N))
            except (TypeError, ValueError, SyntaxError):
                return value

    def __merge__baseconfig(self, other: "BaseConfiguration") -> "BaseConfiguration":
        filtered = {k: v for k, v in dc.asdict(other).items() if v is not None}
        return self.__merge__dict(filtered)

    def __merge__dict(self, other: dict[str, Any]) -> "BaseConfiguration":
        return dc.replace(self, **other)

    def __or__(self, other: "BaseConfiguration" | dict) -> "BaseConfiguration":
        if isinstance(other, BaseConfiguration):
            return self.__merge__baseconfig(other)
        elif isinstance(other, dict):
            return self.__merge__dict(other)
        else:
            raise TypeError(ERR_INVALID_MERGE_TYPE.format(this_type=type(self), that_type=type(other)))
