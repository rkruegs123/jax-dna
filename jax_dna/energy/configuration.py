"""Configuration class for energy models."""

import warnings
from typing import Any, Union

import chex

import jax_dna.utils.types as jdt

ERR_MISSING_REQUIRED_PARAMS = "Required properties {props} are not initialized."
ERR_OPT_DEPENDENT_PARAMS = "Only {req_params} permitted for optimization, but found {given_params}"
WARN_INIT_PARAMS_NOT_IMPLEMENTED = "init_params not implemented"
WARN_DEPENDENT_PARAMS_NOT_INITIALIZED = "Dependent parameters not initialized"


@chex.dataclass(frozen=True)
class BaseConfiguration:
    """Base class for configuration classes.

    This class should not be used directly.

    Parameters:
        params_to_optimize (tuple[str]): parameters to optimize
        required_params (tuple[str]): required parameters
        non_optimizable_required_params (tuple[str]): required parameters that are not optimizable
        dependent_params (tuple[str]): dependent parameters, these are calculated from the independent parameters
        OPT_ALL (tuple[str]): CONSTANT, is a wild card for all parameters
    """

    params_to_optimize: tuple[str] = ()
    required_params: tuple[str] = ()
    non_optimizable_required_params: tuple[str] = ()
    dependent_params: tuple[str] = ()
    OPT_ALL: tuple[str] = ("*",)

    @property
    def opt_params(self) -> dict[str, jdt.Scalar]:
        """Returns the parameters to optimize."""
        if self.params_to_optimize == self.OPT_ALL:
            params = {
                k: v
                for k, v in self.items()
                if (k in self.required_params) and (k not in self.non_optimizable_required_params)
            }
        else:
            params = {k: v for k, v in self.items() if k in self.params_to_optimize}

        return params

    def __post_init__(self) -> None:
        """Checks validity of the configuration."""
        non_initialized_props = [param for param in self.required_params if getattr(self, param) is None]
        if non_initialized_props:
            raise ValueError(ERR_MISSING_REQUIRED_PARAMS.format(props=",".join(non_initialized_props)))

        optimizable_params = set(self.required_params) - set(self.non_optimizable_required_params)
        unoptimizable_params = set(self.params_to_optimize) - optimizable_params
        if unoptimizable_params and unoptimizable_params != set(self.OPT_ALL):
            raise ValueError(
                ERR_OPT_DEPENDENT_PARAMS.format(
                    req_params=",".join(sorted(optimizable_params)),
                    given_params=",".join(sorted(unoptimizable_params)),
                )
            )

    def init_params(self) -> "BaseConfiguration":
        """Initializes the dependent parameters in configuration.

        Should be implemented in the subclass if dependent parameters are present.
        """
        warnings.warn(WARN_INIT_PARAMS_NOT_IMPLEMENTED, stacklevel=1)
        return self

    @classmethod
    def from_dict(cls, params: dict[str, float], params_to_optimize: tuple[str] = ()) -> "BaseConfiguration":
        """Creates a configuration from a dictionary."""
        return cls(**(params | {"params_to_optimize": params_to_optimize}))

    def to_dictionary(
        self,
        *,
        include_dependent: bool,
        exclude_non_optimizable: bool,
    ) -> dict[str, jdt.ARR_OR_SCALAR]:
        """Converts the configuration to a dictionary."""
        params = {k: getattr(self, k) for k in self.required_params}

        if include_dependent:
            for k in self.dependent_params:
                if val := getattr(self, k):
                    params[k] = val
                else:
                    warnings.warn(WARN_DEPENDENT_PARAMS_NOT_INITIALIZED, stacklevel=1)

        if exclude_non_optimizable:
            for k in self.non_optimizable_required_params:
                params.pop(k, None)

        return params

    def __merge__baseconfig(self, other: "BaseConfiguration") -> "BaseConfiguration":
        """Merges two BaseConfiguration objects."""
        filtered = {k: v for k, v in other.items() if v is not None}
        return self.__merge__dict(filtered)

    def __merge__dict(self, other: dict[str, Any]) -> "BaseConfiguration":
        """Merges a dictionary with the configuration."""
        return self.replace(**other)

    # python doesn't like using the bar for type hints when inside the class, use Union for now
    def __or__(self, other: Union["BaseConfiguration", dict[str, jdt.ARR_OR_SCALAR]]) -> "BaseConfiguration":
        """Convenience method to merge a configuration or a dictionary with the current configuration.

        Returns a new configuration object.
        """
        if isinstance(other, BaseConfiguration):
            merge_fn = self.__merge__baseconfig
        elif isinstance(other, dict):
            merge_fn = self.__merge__dict
        else:
            return NotImplemented

        return merge_fn(other)
