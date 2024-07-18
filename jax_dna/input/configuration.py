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
WARN_INIT_PARAMS_NOT_IMPLEMENTED = "init_params not implemented"


@chex.dataclass(frozen=True)
class BaseConfiguration:
    params_to_optimize: list[str] = dc.field(default_factory=list)
    required_params: list[str] = dc.field(default_factory=list)

    @property
    def opt_params(self) -> dict[str, jdt.Scalar]:
        return {k: v for k, v in dc.asdict(self).items() if k in self.params}

    def __post_init__(self):
        non_initialized_props = [param for param in self.required_params if getattr(self, param) is None]
        if non_initialized_props:
            raise ValueError(ERR_MISSING_REQUIRED_PARAMS.format(props=",".join(non_initialized_props)))

    def init_params(self) -> "BaseConfiguration":
        warnings.warn(WARN_INIT_PARAMS_NOT_IMPLEMENTED)
        return self

    @staticmethod
    def parse_toml(file_path: str, key: str) -> dict[str, Any]:
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

    def __or__(self, other: "BaseConfiguration") -> "BaseConfiguration":
        filtered = {k: v for k, v in dc.asdict(other).items() if v is not None}
        return dc.replace(self, **filtered)
