"""Objectives implemented as ray actors."""

import logging
import types
import typing
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax_md
import ray
import typing_extensions

import jax_dna.energy as jdna_energy
import jax_dna.input.tree as jdna_tree
import jax_dna.utils.types as jdna_types

ERR_DIFFTRE_MISSING_KWARGS = "Missing required kwargs: {missing_kwargs}."
ERR_MISSING_ARG = "Missing required argument: {missing_arg}."
ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

EnergyFn = jdna_energy.base.BaseEnergyFunction | jdna_energy.base.ComposedEnergyFunction
empty_dict = types.MappingProxyType({})


class Objective:
    """Base class for objectives that calculate gradients."""

    def __init__(
        self,
        name: str,
        required_observables: list[str],
        needed_observables: list[str],
        logging_observables: list[str],
        grad_or_loss_fn: typing.Callable[[tuple[str, ...]], jdna_types.Grads],
        logger_config: dict[str, typing.Any] = empty_dict,
    ) -> "Objective":
        """Initialize the objective.

        Args:
            name (str): The name of the objective.
            required_observables (list[str]): The observables that are required
                to calculate the gradients.
            needed_observables (list[str]): The observables that are needed to
                calculate the gradients.
            logging_observables (list[str]): The observables that are used for
                logging.
            grad_or_loss_fn (typing.Callable[[tuple[str, ...]], jdna_types.Grads]):
                The function that calculates the loss of the objective
            logger_config (dict[str, typing.Any]): The configuration for the logger.
        """
        if name is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="name"))
        if required_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="required_observables"))
        if needed_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="needed_observables"))
        if logging_observables is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="logging_observables"))
        if grad_or_loss_fn is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="grad_or_loss_fn"))

        self._name = name
        self._required_observables = required_observables
        self._needed_observables = needed_observables
        self._grad_or_loss_fn = grad_or_loss_fn
        self._obtained_observables = []
        self._logging_observables = logging_observables
        logging.basicConfig(**logger_config)
        self._logger = logging.getLogger(__name__)

    def name(self) -> str:
        """Return the name of the objective."""
        return self._name

    def required_observables(self) -> list[str]:
        """Return the observables that are required to calculate the gradients."""
        return self._required_observables

    def needed_observables(self) -> list[str]:
        """Return the observables that are still needed."""
        return self._needed_observables

    def obtained_observables(self) -> list[tuple[str, jdna_types.SimulatorActorOutput]]:
        """Return the latest observed values for all observables."""
        return self._obtained_observables

    def logging_observables(self) -> list[tuple[str, typing.Any]]:
        """Return the latest observed values for the logging observables."""
        lastest_observed = self._obtained_observables
        return_values = []
        for log_obs in self._logging_observables:
            for obs in lastest_observed:
                if obs[0] == log_obs:
                    return_values.append(obs)
                    break
        return return_values

    def is_ready(self) -> bool:
        """Check if the objective is ready to calculate its gradients."""
        obtained_keys = [obs[0] for obs in self._obtained_observables]
        return all(obs in obtained_keys for obs in self._required_observables)

    def update(
        self,
        sim_results: list[tuple[list[str], list[str]]],
    ) -> None:
        """Update the observables with the latest simulation results."""
        for sim_exposes, sim_output in sim_results:
            for exposed, output in filter(
                lambda e: e[0] in self._needed_observables, zip(sim_exposes, sim_output, strict=True)
            ):
                self._obtained_observables.append((exposed, jdna_tree.load_pytree(output)))
                self._needed_observables.remove(exposed)

    def calculate(self) -> list[jdna_types.Grads]:
        """Calculate the gradients of the objective."""
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obtained_observables = sorted(
            self._obtained_observables,
            key=lambda x: self._required_observables.index(x[0]),
        )

        sorted_obs = [x[1] for x in sorted_obtained_observables]

        grads, loss = self._grad_or_loss_fn(*sorted_obs)

        self._obtained_observables = [
            ("loss", loss),
            *list(zip(self._required_observables, sorted_obs, strict=True)),
        ]

        return grads

    def post_step(self, opt_params: dict) -> None:  # noqa: ARG002 - not all objectives need params
        """Reset the needed observables for the next step."""
        self._needed_observables = self._required_observables
        self._obtained_observables = []


@ray.remote
class SimGradObjectiveActor(Objective):
    """Objective that calculates the gradients of a simulation."""


def compute_weights_and_neff(
    beta: float,
    new_energies: jdna_types.Arr_N,
    ref_energies: jdna_types.Arr_N,
) -> tuple[jnp.ndarray, float]:
    """Compute the weights and normalized effective sample size of a trajectory.

    Calculation derived from the DiffTRe algorithm.

    https://www.nature.com/articles/s41467-021-27241-4
    See equations 4 and 5.

    Args:
        beta: The inverse temperature.
        new_energies: The new energies of the trajectory.
        ref_energies: The reference energies of the trajectory.

    Returns:
        The weights and the normalized effective sample size
    """
    diffs = new_energies - ref_energies
    boltz = jnp.exp(-beta * diffs)
    weights = boltz / jnp.sum(boltz)
    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
    return weights, n_eff / len(weights)


def compute_loss(
    opt_params: jdna_types.Params,
    energy_fn_builder: callable,
    beta: float,
    loss_fn: Callable[
        [jax_md.rigid_body.RigidBody, jdna_types.Arr_N, EnergyFn], tuple[jnp.ndarray, tuple[str, typing.Any]]
    ],
    ref_states: jax_md.rigid_body.RigidBody,
    ref_energies: jdna_types.Arr_N,
) -> tuple[float, tuple[float, jnp.ndarray]]:
    """Compute the grads, loss, and auxiliary values.

    Args:
        opt_params: The optimization parameters.
        energy_fn_builder: A function that builds the energy function.
        beta: The inverse temperature.
        loss_fn: The loss function.
        ref_states: The reference states of the trajectory.
        ref_energies: The reference energies of the trajectory.

    Returns:
        The grads, the loss, a tuple containing the normalized effective sample
        size and the measured value of the trajectory, and the new energies.
    """
    energy_fn = energy_fn_builder(opt_params)
    new_energies = energy_fn_builder(opt_params)(ref_states)
    weights, neff = compute_weights_and_neff(
        beta,
        new_energies,
        ref_energies,
    )
    loss, (measured_value, meta) = loss_fn(ref_states, weights, energy_fn)
    return loss, (neff, measured_value, new_energies)


compute_loss_and_grad = jax.value_and_grad(compute_loss, has_aux=True)


class DiffTReObjective(Objective):
    """Objective that calculates the gradients of an objective using DiffTRe."""

    def __init__(
        self,
        name: str,
        required_observables: list[str],
        needed_observables: list[str],
        logging_observables: list[str],
        grad_or_loss_fn: typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        energy_fn_builder: Callable[[jdna_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params: jdna_types.Params,
        trajectory_key: str,
        beta: float,
        n_equilibration_steps: int,
        min_n_eff_factor: float = 0.95,
        max_valid_opt_steps: int | None = None,
        logging_config: dict[str, typing.Any] = empty_dict,
    ) -> "DiffTReObjective":
        """Initialize the DiffTRe objective.

        Args:
            name: The name of the objective.
            required_observables: The observables that are required to calculate the gradients.
            needed_observables: The observables that are needed to calculate the gradients.
            logging_observables: The observables that are used for logging.
            grad_or_loss_fn: The function that calculates the loss of the objective.
            energy_fn_builder: A function that builds the energy function.
            opt_params: The optimization parameters.
            trajectory_key: The key of the trajectory in the observables.
            beta: The inverse temperature.
            n_equilibration_steps: The number of equilibration steps.
            min_n_eff_factor: The minimum normalized effective sample size.
            max_valid_opt_steps: The maximum number of steps a trajectory is valid for.
            logging_config: The configuration for the logger.
        """
        super().__init__(
            name,
            required_observables,
            needed_observables,
            logging_observables,
            grad_or_loss_fn,
            logger_config=logging_config,
        )
        if energy_fn_builder is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="energy_fn_builder"))
        if opt_params is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="opt_params"))
        if trajectory_key is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="trajectory_key"))
        if beta is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="beta"))
        if n_equilibration_steps is None:
            raise ValueError(ERR_MISSING_ARG.format(missing_arg="n_equilibration_steps"))

        self._energy_fn_builder = energy_fn_builder
        self._opt_params = opt_params
        self._trajectory_key = trajectory_key
        self._beta = beta
        self._n_eq_steps = n_equilibration_steps
        self._n_eff_factor = min_n_eff_factor
        self._max_valid_opt_steps = max_valid_opt_steps
        self._opt_steps = 0

        self._reference_states = None
        self._reference_energies = None

    @typing_extensions.override
    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        # want the required observables in the order they are requested
        sorted_obtained_observables = sorted(
            filter(lambda x: x[0] in self._required_observables, self._obtained_observables),
            key=lambda x: self._required_observables.index(x[0]),
        )
        sorted_obs = [x[1] for x in sorted_obtained_observables]

        (loss, (_, measured_value, new_energies)), grads = compute_loss_and_grad(
            self._opt_params,
            self._energy_fn_builder,
            self._beta,
            self._grad_or_loss_fn,
            self._reference_states,
            self._reference_energies,
        )

        latest_neff = next(obs for obs in self._obtained_observables if obs[0] == "neff")
        self._obtained_observables = [
            ("loss", loss),
            latest_neff,
            measured_value,
            *list(zip(self._required_observables, sorted_obs, strict=True)),
        ]

        return grads

    @typing_extensions.override
    def is_ready(self) -> bool:
        have_trajectory = super().is_ready()
        if have_trajectory:
            new_trajectory = next(filter(lambda x: x[0] == self._trajectory_key, self._obtained_observables))[1]

            if self._reference_states is None:
                self._reference_states = new_trajectory.slice(
                    slice(self._n_eq_steps, len(new_trajectory.rigid_body.center), None)
                )
                self._reference_energies = self._energy_fn_builder(self._opt_params)(self._reference_states)

            _, neff = compute_weights_and_neff(
                beta=self._beta,
                new_energies=self._energy_fn_builder(self._opt_params)(self._reference_states),
                ref_energies=self._reference_energies,
            )

            if any(obs[0] == "neff" for obs in self._obtained_observables):
                self._obtained_observables = [
                    (obs[0], neff) if obs[0] == "neff" else obs for obs in self._obtained_observables
                ]
            else:
                self._obtained_observables.append(("neff", neff))

            # if the trajectory is no longer valid remove it form obtained
            # and add it to needed so that a new trajectory is run.
            if neff < self._n_eff_factor or self._opt_steps == self._max_valid_opt_steps:
                self._obtained_observables = list(
                    filter(lambda x: x[0] != self._trajectory_key, self._obtained_observables)
                )
                self._needed_observables = [self._trajectory_key]
                self._reference_states = None
                have_trajectory = False

        return have_trajectory

    @typing_extensions.override
    def post_step(
        self,
        opt_params: jdna_types.Params,
    ) -> None:
        # DiffTre objectives may not need to update the trajectory depending on neff
        # the need for a new trajectory is checked in `is_ready`
        self._obtained_observables = list(filter(lambda x: x[0] == self._trajectory_key, self._obtained_observables))
        self._opt_params = opt_params
        self._opt_steps += 1


@ray.remote
class DiffTReObjectiveActor(DiffTReObjective):
    """Objective that calculates the gradients of an objective using DiffTRe and ray."""
