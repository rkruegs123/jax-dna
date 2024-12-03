
from collections.abc import Callable
import functools
import typing

import jax
import jax_md
import ray

import jax.numpy as jnp

import jax_dna.input.tree as jdna_tree
import jax_dna.utils.types as jdna_types

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

ERR_DIFFTRE_MISSING_KWARGS = "Missing required kwargs: {missing_kwargs}."


class Objective:

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        logging_observables:list[str],
        grad_or_loss_fn:typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        **kwargs:dict[str, typing.Any],
    ):
        self._required_observables = required_observables
        self._needed_observables = needed_observables
        self.grad_or_loss_fn = grad_or_loss_fn
        self._obtained_observables = []
        self._logging_observables = logging_observables


    def needed_observables(self) -> list[str]:
        return self._needed_observables


    def obtained_observables(self) -> list[tuple[str, jdna_types.SimulatorActorOutput]]:
        return self._obtained_observables


    def logging_observables(self) -> list[tuple[str, typing.Any]]:
        lastest_observed = self._obtained_observables
        return_values = []
        for log_obs in self._logging_observables:
            for obs in lastest_observed:
                if obs[0] == log_obs:
                    return_values.append((obs))
                    break
        return return_values


    def is_ready(self) -> bool:
        obtained_keys = [obs[0] for obs in self._obtained_observables]
        return all([obs in obtained_keys for obs in self._required_observables])


    def update(
        self,
        sim_results: list[tuple[list[str], typing.Any]],
    ) -> None:

        for sim_exposes, sim_output in sim_results:
            for exposed, output in filter(
                lambda e: e[0] in self._needed_observables,
                zip(sim_exposes, sim_output)
            ):
                self._obtained_observables.append(
                    (exposed, jdna_tree.load_pytree(output))
                )
                self._needed_observables.remove(exposed)


    def calculate(self) -> list[jdna_types.Grads]:
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)

        sorted_obs = list(map(
            lambda x: x[1],
            sorted(
                self._obtained_observables,
                key=lambda x: self.required_observables.index(x[0]),
            )
        ))

        grads, loss = self.grad_or_loss_fn(*sorted_obs)

        self._obtained_observables = [
            ("loss", loss),
            *[(ro, so) for (ro, so) in zip(self._required_observables, sorted_obs)],
        ]

        return grads


    def post_step(self, opt_params:dict) -> None:
        self._needed_observables = self._required_observables
        self._obtained_observables = []


@ray.remote
class SimGradObjective(Objective):
    pass


def compute_weights_and_neff(
    energy_fn:callable,
    beta:float,
    ref_states:jnp.ndarray,
    new_states:jnp.ndarray,
) -> jnp.ndarray:
    diffs = energy_fn(new_states) - energy_fn(ref_states)
    boltz = jnp.exp(-beta * diffs)
    weights = boltz / jnp.sum(boltz)
    n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))
    return weights, n_eff / len(weights)


@functools.partial(jax.value_and_grad, has_aux=True)
def compute_loss(
    opt_params: jdna_types.Params,
    energy_fn_builder:callable,
    beta:float,
    loss_fn:Callable[
        [jax_md.rigid_body.RigidBody, jdna_types.Arr_N],
        tuple[jnp.ndarray, tuple[str, typing.Any]]
    ],
    ref_states:jax_md.rigid_body.RigidBody,
    new_states:jnp.ndarray,
) -> tuple[float, tuple[float, jnp.ndarray]]:
    weights, neff = compute_weights_and_neff(
        energy_fn_builder(opt_params),
        beta,
        ref_states,
        new_states,
    )
    loss, measured_value = loss_fn(new_states, weights)
    return loss, (neff, measured_value)


@ray.remote
class DiffTReObjective(Objective):

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        logging_observables:list[str],
        grad_or_loss_fn:typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        energy_fn_builder:Callable[[jdna_types.Params], Callable[[jnp.ndarray], jnp.ndarray]],
        opt_params:jdna_types.Params,
        trajectory_key:str,
        beta:float,
        **kwargs:dict[str, typing.Any],
    ):
        super().__init__(
            required_observables,
            needed_observables,
            logging_observables,
            grad_or_loss_fn,
            **kwargs,
        )
        self._energy_fn_builder = energy_fn_builder
        self._opt_params = opt_params
        self._trajectory_key = trajectory_key
        self._beta = beta

        self.n_eff_factor = kwargs.get("min_n_eff_factor", 0.95)
        self.accept_expired_trajectory_grads = kwargs.get("accept_expired_trajectory_grads", False)
        self.expired_trajectory_value = kwargs.get("expired_trajectory_value", 0.0)

        self._reference_states = None


    def calculate(self) -> list[jdna_types.Grads]:
        # we need override the grads calculation to check if the trajectory
        # is still valid and if so ask for a new trajectory
        if not self.is_ready():
            raise ValueError(ERR_OBJECTIVE_NOT_READY)


        sorted_obs = list(map(
            lambda x: x[1],
            sorted(
                self._obtained_observables,
                key=lambda x: self._required_observables.index(x[0]),
            )
        ))
        new_trajectory = sorted_obs
        if self._reference_states is None:
            self._reference_states = new_trajectory
        # n_eff_measured here should be the normalized effective sample size
        # of the trajectory.
        (loss, (n_eff_measured, measured_value)), grads = compute_loss(
            self._opt_params,
            self._energy_fn_builder,
            self._beta,
            self.grad_or_loss_fn,
            self._reference_states,
            new_trajectory,
        )

        self._obtained_observables = [
            ("loss", loss),
            measured_value,
            *[(ro, so) for (ro, so) in zip(self.required_observables, sorted_obs)],

        ]

        invalid_trajectory = n_eff_measured < self.n_eff_factor
        if invalid_trajectory:
            # we need to regenerate the trajectory
            self._needed_observables.append(self._trajectory_key)
            self._reference_states = None
            if not self.accept_expired_trajectory_grads:
                # if we are not accepting expired trajectory grads
                # we zero them out.
                # maybe change this to `nan` or make it a settable value/
                grads = jax.tree.map(
                    lambda x: jnp.zeros_like(x) + self.expired_trajectory_value,
                    grads,
                )

        return grads



    def post_step(self, opt_params:jdna_types.Params) -> None:
        self._needed_observables = self.required_observables
        self._obtained_observables = []
        self._opt_params = opt_params
