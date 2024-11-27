
import typing

import jax
import ray

import jax.numpy as jnp

import jax_dna.input.tree as jdna_tree
import jax_dna.utils.types as jdna_types

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."

ERR_DIFFTRE_MISSING_KWARGS = "`opt_params` and `trajectory_key` not provided."


class Objective:

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        logging_observables:list[str],
        grad_fn:typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        **kwargs:dict[str, typing.Any],
    ):
        self._required_observables = required_observables
        self._needed_observables = needed_observables
        self.grad_fn = grad_fn
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
        return all([obs in obtained_keys for obs in self.required_observables])


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

        grads, loss = self.grad_fn(*sorted_obs)

        self._obtained_observables = [
            ("loss", loss),
            *[(ro, so) for (ro, so) in zip(self.required_observables, sorted_obs)],
        ]

        return grads


    def post_step(self) -> None:
        self._needed_observables = self._required_observables
        self._obtained_observables = []


@ray.remote
class SimGradObjective(Objective):
    pass


@ray.remote
class DiffTReObjective(Objective):

    def __init__(
        self,
        required_observables:list[str],
        needed_observables:list[str],
        logging_observables:list[str],
        grad_fn:typing.Callable[[tuple[jdna_types.SimulatorActorOutput]], jdna_types.Grads],
        **kwargs:dict[str, typing.Any],
    ):
        super().__init__(
            required_observables,
            needed_observables,
            logging_observables,
            grad_fn,
            **kwargs,
        )
        self.n_eff_factor = kwargs.get("min_n_eff_factor", 0.95)
        self.accept_expired_trajectory_grads = kwargs.get("accept_expired_trajectory_grads", False)

        if "opt_params" not in kwargs or "trajectory_key" not in kwargs:
            raise ValueError(ERR_DIFFTRE_MISSING_KWARGS)

        self._opt_params = kwargs.get("opt_params")
        self._trajectory_key = kwargs.get("trajectory_key")
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
                key=lambda x: self.required_observables.index(x[0]),
            )
        ))

        grads, loss, is_valid_trajectory = self.grad_fn(
            self._reference_states,
            *sorted_obs,
        )

        self._obtained_observables = [
            ("loss", loss),
            *[(ro, so) for (ro, so) in zip(self.required_observables, sorted_obs)],
        ]

        if not is_valid_trajectory:
            # we need to regenerate the trajectory
            self._needed_observables.append(self._trajectory_key)
            self._reference_states = None
            if not self.accept_expired_trajectory_grads:
                # if we are not accepting expired trajectory grads
                # we zero them out
                grads = jax.tree.map(lambda x: jnp.zeros_like(x), grads)

        return grads



    def post_step(self) -> None:
        self._needed_observables = self.required_observables
        self._obtained_observables = []
