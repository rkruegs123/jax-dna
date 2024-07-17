import dataclasses as dc
import warnings
from typing import Any

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


SYMPY_EVAL_N: int = 64


@chex.dataclass(frozen=True)
class BaseConfiguration:
    params_to_optimize:list[str] = dc.field(default_factory=list)

    def __post_init__(self):
        non_initialized_props = [prop.name for prop in dc.fields(self) if getattr(self, prop.name) is None]
        if non_initialized_props:
            raise ValueError(f"Properties {','.join(non_initialized_props)} are not initialized.")

    def init_params(self) -> "BaseConfiguration":
        warnings.warn("init_params not implemented")
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
