"""Tests for the jax_dna.ui.loggers.jupyter module."""

import dataclasses as dc
import importlib

import pytest

from jax_dna.ui.loggers import jupyter


@dc.dataclass
class MockFigureData:
    name: str
    x: list[float]
    y: list[float]


class MockFigureWidget:
    def __init__(self, *args, **kwargs):
        pass

    def add_trace(self, *args, **kwargs):
        pass

    def update_traces(self, *args, **kwargs):
        pass

    def update_layout(self, *args, **kwargs):
        pass

    @property
    def layout(self) -> dict:
        return self

    @property
    def domain(self) -> dict:
        return self

    def show(self):
        return self

    @property
    def data(self) -> list[MockFigureData]:
        return [MockFigureData("test", [1, 2, 3], [2, 3, 4])]

    def __getitem__(self, index: str | int):
        return self


@pytest.mark.parametrize(
    ("n_plots", "nrows", "ncols", "expected"),
    [
        (1, 1, 1, (1, 1)),
        (2, 2, 1, (2, 1)),
        (2, None, None, (2, 2)),
        (3, 2, 2, (2, 2)),
        (3, -1, -1, (2, 2)),
        (4, 2, 2, (2, 2)),
        (5, 2, 2, (3, 3)),
    ],
)
def test_calc_rows_and_columns(
    n_plots: int,
    nrows: int,
    ncols: int,
    expected: tuple[int, int],
):
    """Test the calculation of rows and columns for a given number of plots."""
    assert jupyter.calc_rows_and_columns(n_plots, nrows, ncols) == expected


def test_setup_figure_layout():
    """Test the setup of the figure layout."""
    fig = MockFigureWidget()
    jupyter.setup_figure_layout(fig, 1, 1, ["test"])
    # not sure what we want te test here or how to test it, maybe refactor


def test_plotly_logger_log_metric():
    """Test logging a metric."""
    importlib.reload(jupyter)
    jupyter.figure_widget_f = MockFigureWidget

    logger = jupyter.PlotlyLogger(["test"], None, None)
    logger.log_metric("test", 1.0, 0)
    # not sure what we want te test here or how to test it, maybe refactor


def test_plotly_logger_change_size():
    """Test change size"""
    importlib.reload(jupyter)
    jupyter.figure_widget_f = MockFigureWidget

    logger = jupyter.PlotlyLogger(["test"], None, None)
    logger.change_size(100, 100)
    # not sure what we want te test here or how to test it, maybe refactor


def test_plotly_logger_show():
    """Test change size"""
    importlib.reload(jupyter)
    jupyter.figure_widget_f = MockFigureWidget

    logger = jupyter.PlotlyLogger(["test"], None, None)
    logger.show()
    # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_log_metric():
    """Test logging a metric."""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    logger.log_metric("test metric 1", 1.0, 0)
    # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_show():
    """Test show size"""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    logger.show()
    # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_increment_prog_bar():
    """Test increment_prog_bar"""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    logger.increment_prog_bar()
    # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_update_simulators():
    """Test update_simulators"""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    funcs = {
        jupyter.Status.STARTED: logger.set_simulator_started,
        jupyter.Status.RUNNING: logger.set_simulator_running,
        jupyter.Status.COMPLETE: logger.set_simulator_complete,
        jupyter.Status.ERROR: logger.set_simulator_error,
    }

    for status in jupyter.Status:
        logger.update_simulator_status("test_simulator", status)
        # not sure what we want te test here or how to test it, maybe refactor

        funcs[status]("test_simulator")
        # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_update_objectives():
    """Test update_objectives"""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    for status in jupyter.Status:
        logger.update_objective_status("test_objective", status)
        # not sure what we want te test here or how to test it, maybe refactor

    funcs = {
        jupyter.Status.STARTED: logger.set_objective_started,
        jupyter.Status.RUNNING: logger.set_objective_running,
        jupyter.Status.COMPLETE: logger.set_objective_complete,
        jupyter.Status.ERROR: logger.set_objective_error,
    }

    for status in jupyter.Status:
        logger.update_objective_status("test_objective", status)
        # not sure what we want te test here or how to test it, maybe refactor

        funcs[status]("test_objective")
        # not sure what we want te test here or how to test it, maybe refactor


def test_jupyter_logger_update_observables():
    """Test update_observables"""
    importlib.reload(jupyter)
    logger = jupyter.JupyterLogger(
        [
            "test_simulator",
        ],
        [
            "test_observable",
        ],
        [
            "test_objective",
        ],
        [
            "test metric 1",
            "test metric 2",
        ],
        max_opt_steps=10,
    )

    for status in jupyter.Status:
        logger.update_observable_status("test_observable", status)
        # not sure what we want te test here or how to test it, maybe refactor

    funcs = {
        jupyter.Status.STARTED: logger.set_observable_started,
        jupyter.Status.RUNNING: logger.set_observable_running,
        jupyter.Status.COMPLETE: logger.set_observable_complete,
        jupyter.Status.ERROR: logger.set_observable_error,
    }

    for status in jupyter.Status:
        logger.update_observable_status("test_observable", status)
        # not sure what we want te test here or how to test it, maybe refactor

        funcs[status]("test_observable")
        # not sure what we want te test here or how to test it, maybe refactor


if __name__ == "__main__":
    test_setup_figure_layout()
