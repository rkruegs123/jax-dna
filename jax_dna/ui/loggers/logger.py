"""Base logger protocol."""

import warnings
from enum import Enum
from pathlib import Path

MISSING_LOGDIR_WANING = "`log_dir` not results might not be saved to disk."


class Status(Enum):
    """Status of a simulator, objective, or observable."""

    STARTED = 0
    RUNNING = 1
    COMPLETE = 2
    ERROR = 3


def convert_to_fname(name: str) -> str:
    """Convert a metric name to a valid filename."""
    return name.replace("/", "_").replace(" ", "_") + ".csv"


class Logger:
    """Base Logger that logs to disk."""

    def __init__(self, log_dir: str | Path | None = None) -> "Logger":
        """Initialize the logger."""
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        else:
            warnings.warn(MISSING_LOGDIR_WANING, stacklevel=1)
            log_dir = None
        self.log_dir = log_dir

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log the `value` for `name` at `step`.

        Args:
            name (str): the name of the metric
            value (float): the value of the metric
            step (int): the step at which the metric was recorded
        """
        if self.log_dir is not None:
            fname = self.log_dir / convert_to_fname(name)
            with fname.open(mode="a") as f:
                f.write(f"{step},{value}\n")

    def __update_status(self, name: str, status: Status) -> None:
        """Updates the status of a simulator, objective, or observable."""
        if self.log_dir is not None:
            fname = self.log_dir / convert_to_fname(name)
            with fname.open(mode="a") as f:
                f.write(f"{name},{status}\n")

    def update_simulator_status(self, name: str, status: Status) -> None:
        """Updates the status of a simulator."""
        self.__update_status(name, status)

    def set_simulator_started(self, name: str) -> None:
        """Sets the status of a simulator to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_simulator_running(self, name: str) -> None:
        """Sets the status of a simulator to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_simulator_complete(self, name: str) -> None:
        """Sets the status of a simulator to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_simulator_error(self, name: str) -> None:
        """Sets the status of a simulator to ERROR."""
        self.__update_status(name, Status.ERROR)

    def update_objective_status(self, name: str, status: Status) -> None:
        """Updates the status of an objective."""
        self.__update_status(name, status)

    def set_objective_started(self, name: str) -> None:
        """Sets the status of an objective to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_objective_running(self, name: str) -> None:
        """Sets the status of an objective to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_objective_complete(self, name: str) -> None:
        """Sets the status of an objective to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_objective_error(self, name: str) -> None:
        """Sets the status of an objective to ERROR."""
        self.__update_status(name, Status.ERROR)

    def update_observable_status(self, name: str, status: Status) -> None:
        """Updates the status of an observable."""
        self.__update_status(name, status)

    def set_observable_started(self, name: str) -> None:
        """Sets the status of an observable to STARTED."""
        self.__update_status(name, Status.STARTED)

    def set_observable_running(self, name: str) -> None:
        """Sets the status of an observable to RUNNING."""
        self.__update_status(name, Status.RUNNING)

    def set_observable_complete(self, name: str) -> None:
        """Sets the status of an observable to COMPLETE."""
        self.__update_status(name, Status.COMPLETE)

    def set_observable_error(self, name: str) -> None:
        """Sets the status of an observable to ERROR."""
        self.__update_status(name, Status.ERROR)
