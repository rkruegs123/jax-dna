"""Logger logging jax_dna optimization results to the console."""

from pathlib import Path

import typing_extensions

from jax_dna.ui.loggers import logger
from jax_dna.ui.loggers.logger import Status


class ConsoleLogger(logger.Logger):
    """Console logger."""

    def __init__(self, log_dir: str | Path | None = None) -> "ConsoleLogger":
        """Initialize the console logger."""
        super().__init__(log_dir)

    @typing_extensions.override
    def log_metric(
        self,
        name: str,
        value: float,
        step: int,
    ) -> None:
        super().log_metric(name, value, step)

        print(f"Step: {step}, {name}: {value}")  # noqa: T201 -- we intend to print to the console

    def __update_status(self, name: str, status: Status) -> None:
        return print(name, status)  # noqa: T201 -- we intend to print to the console

    @typing_extensions.override
    def update_simulator_status(self, name: str, status: Status) -> None:
        self.__update_status(name, status)

    @typing_extensions.override
    def set_simulator_started(self, name: str) -> None:
        self.update_simulator_status(name, Status.STARTED)

    @typing_extensions.override
    def set_simulator_running(self, name: str) -> None:
        self.update_simulator_status(name, Status.RUNNING)

    @typing_extensions.override
    def set_simulator_complete(self, name: str) -> None:
        self.update_simulator_status(name, Status.COMPLETE)

    @typing_extensions.override
    def set_simulator_error(self, name: str) -> None:
        self.update_simulator_status(name, Status.ERROR)

    @typing_extensions.override
    def update_objective_status(self, name: str, status: Status) -> None:
        self.__update_status(name, status)

    @typing_extensions.override
    def set_objective_started(self, name: str) -> None:
        self.update_objective_status(name, Status.STARTED)

    @typing_extensions.override
    def set_objective_running(self, name: str) -> None:
        self.update_objective_status(name, Status.RUNNING)

    @typing_extensions.override
    def set_objective_complete(self, name: str) -> None:
        self.update_objective_status(name, Status.COMPLETE)

    @typing_extensions.override
    def set_objective_error(self, name: str) -> None:
        self.update_objective_status(name, Status.ERROR)

    @typing_extensions.override
    def update_observable_status(self, name: str, status: Status) -> None:
        self.__update_status(name, status)

    @typing_extensions.override
    def set_observable_started(self, name: str) -> None:
        self.update_observable_status(name, Status.STARTED)

    @typing_extensions.override
    def set_observable_running(self, name: str) -> None:
        self.update_observable_status(name, Status.RUNNING)

    @typing_extensions.override
    def set_observable_complete(self, name: str) -> None:
        self.update_observable_status(name, Status.COMPLETE)

    @typing_extensions.override
    def set_observable_error(self, name: str) -> None:
        self.update_observable_status(name, Status.ERROR)
