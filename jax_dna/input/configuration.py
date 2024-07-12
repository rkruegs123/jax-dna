import dataclasses as dc
from typing import Any
import warnings

import chex
import jax
import numpy as np
import sympy

# if we're using python >=3.11 then use native toml module, otherwise use toml from pypi
try:
    import tomllib as toml
except ImportError:
    # tox uses this parser, so we'll use it too
    import tomli as toml


@chex.dataclass(frozen=True)
class BaseConfiguration:
    sympy_eval_n: int = 64.0    # this is a float to help with jax tree parsing
                                # but I am not sure if we should keep this.

    def init_params(self) -> "BaseConfiguration":
        warnings.warn("init_params not implemented")
        return self

    @staticmethod
    def parse_toml(file_path: str, key:str) -> dict[str, Any]:

        with open(file_path, "rb") as f:
            config_dict = toml.load(f)

        if key in config_dict:
            config_dict = config_dict[key]

        return jax.tree.map(BaseConfiguration.parse_value, config_dict)


    @staticmethod
    def parse_value(value:str|float):
        if isinstance(value, str):
            return BaseConfiguration.parse_str(value)
        else:
            return value


    @staticmethod
    def parse_str(value:str) -> str|float:
        try:
            return float(value)
        except ValueError:
            try:
                return float(sympy.parse_expr(value).evalf(n=int(BaseConfiguration.sympy_eval_n)))
            except (TypeError, ValueError, SyntaxError):
                return value


    def __or__(self, other: "BaseConfiguration") -> "BaseConfiguration":
        filtered = {k: v for k, v in dc.asdict(other).items() if v is not None}
        return dc.replace(self, **filtered)

