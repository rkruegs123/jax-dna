
import typing

import ray

import jax.numpy as jnp

import jax_dna.input.tree as jdna_tree
import jax_dna.utils.types as jdna_types

ERR_OBJECTIVE_NOT_READY = "Not all required observables have been obtained."


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
        self.n_eff_factor = kwargs.get("min_n_eff", 0.95)
        self.opt_params = kwargs.get("opt_params")


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

        ref_states_index = self.required_observables.index("trajectory")
        ref_states = sorted_obs[ref_states_index]

        new_energies = energy_fn_builder(params)(ref_states)
        diffs = new_energies - ref_energies
        boltz = jnp.exp(-beta * diffs)
        weights = boltz / jnp.sum(boltz)
        n_eff = jnp.exp(-jnp.sum(weights * jnp.log(weights)))


        grads, loss = self.grad_fn(*sorted_obs)

        self._obtained_observables = [
            ("loss", loss),
            *[(ro, so) for (ro, so) in zip(self.required_observables, sorted_obs)],
        ]

        # if this is a batch of runs we need the second index, otherwise the first
        # TODO(ryanhausen): is this correct?
        t_index = 0 if len(new_energies.shape) == 2 else 1

        if n_eff < self.n_eff_factor * new_energies.shape[0]:
            self._needed_observables = self.required_observables
            self._obtained_observables

        return grads


    def post_step(self) -> None:
        self._needed_observables = self.required_observables
        self._obtained_observables = []
