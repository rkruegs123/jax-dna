
from typing import Any

import jax
import numpy as np
import sympy

try:
    import tomllib as toml
except ImportError:
    # tox uses this parser, so we'll use it too
    import tomli as toml


ERR_MISSING_TOML_ENTRY = "Missing entry {entry} in TOML file"
SYMPY_EVAL_N:int = 32

def parse_str(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        try:
            return float(sympy.parse_expr(value).evalf(n=SYMPY_EVAL_N))
        except (TypeError, ValueError, SyntaxError):
            return value


def parse_value(value: str | float | list[str] | list[float]) -> str | float | np.ndarray:
    if isinstance(value, str):
        return parse_str(value)
    elif isinstance(value, list):
        leaves = jax.tree_util.tree_leaves(value)
        if all(isinstance(leaf, str) for leaf in leaves):
            return jax.tree_util.tree_map(parse_str, value)
        elif all(isinstance(leaf, float) for leaf in leaves):
            return np.array(value)

    return value

def parse_toml(file_path: str, key: str | None = None) -> dict[str, Any]:
    with open(file_path, "rb") as f:
        config_dict = toml.load(f)

    if key is not None:
        if key in config_dict:
            config_dict = config_dict[key]
        else:
            raise ValueError(ERR_MISSING_TOML_ENTRY.format(entry=key))

    return jax.tree_util.tree_map(
        parse_value, config_dict, is_leaf=lambda x: isinstance(x, (str, float, list))
    )