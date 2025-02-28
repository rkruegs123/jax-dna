"""Tests for the ui.logger module."""

import pytest

from jax_dna.ui.loggers import logger


@pytest.mark.parametrize(
    ("fname", "expected"),
    [
        ("test", "test.csv"),
        ("test/first", "test_first.csv"),
        ("test/first/second", "test_first_second.csv"),
        ("test first", "test_first.csv"),
        ("test first second", "test_first_second.csv"),
        ("test first/second", "test_first_second.csv"),
    ],
)
def test_convert_to_fname(fname: str, expected: str) -> None:
    """Test the convert_to_fname function."""
    assert logger.convert_to_fname(fname) == expected


@pytest.mark.parametrize(
    ("name", "value", "step", "save_to_disk"),
    [
        ("test", 1.0, 0, True),
        ("test", 1.0, 0, False),
        ("test", 1.0, 1, True),
        ("test", 1.0, 1, False),
    ],
)
def test_log_metric(tmpdir, name: str, value: float, step: int, save_to_disk: bool) -> None:  # noqa: FBT001 -- for testing
    """Test the log_metric function."""
    log_dir = tmpdir.mkdir("logs")

    log = logger.Logger(log_dir=log_dir if save_to_disk else None)

    log.log_metric(name, value, step)
    fname = log_dir / logger.convert_to_fname(name)
    if save_to_disk:
        assert fname.exists()
        assert fname.readlines()[-1] == f"{step},{value}\n"
    else:
        assert not fname.exists()


@pytest.mark.parametrize(
    ("save_to_disk"),
    [
        (True),
        (False),
    ],
)
def test_simulator_status_updates(tmpdir, save_to_disk: bool) -> None:  # noqa: FBT001 -- for testing
    """Test the simulator status update functions."""
    log_dir = tmpdir.mkdir("logs") if save_to_disk else None
    log = logger.Logger(log_dir=log_dir)

    name = "test_sim"
    log.update_simulator_status(name, logger.Status.STARTED)

    if save_to_disk:
        fname = log_dir / logger.convert_to_fname(name)
        assert fname.exists()

    funcs = {
        logger.Status.STARTED: log.set_simulator_started,
        logger.Status.RUNNING: log.set_simulator_running,
        logger.Status.COMPLETE: log.set_simulator_complete,
        logger.Status.ERROR: log.set_simulator_error,
    }

    for status in logger.Status:
        log.update_simulator_status(name, status)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"

        funcs[status](name)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"


@pytest.mark.parametrize(
    ("save_to_disk"),
    [
        (True),
        (False),
    ],
)
def test_objective_status_updates(tmpdir, save_to_disk: bool) -> None:  # noqa: FBT001 -- for testing
    """Test the objective status update functions."""

    log_dir = tmpdir.mkdir("logs") if save_to_disk else None
    log = logger.Logger(log_dir=log_dir)

    name = "test_obj"
    log.update_objective_status(name, logger.Status.STARTED)

    if save_to_disk:
        fname = log_dir / logger.convert_to_fname(name)
        assert fname.exists()

    funcs = {
        logger.Status.STARTED: log.set_objective_started,
        logger.Status.RUNNING: log.set_objective_running,
        logger.Status.COMPLETE: log.set_objective_complete,
        logger.Status.ERROR: log.set_objective_error,
    }

    for status in logger.Status:
        log.update_objective_status(name, status)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"

        funcs[status](name)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"


@pytest.mark.parametrize(
    ("save_to_disk"),
    [
        (True),
        (False),
    ],
)
def test_observable_status_updates(tmpdir, save_to_disk: bool) -> None:  # noqa: FBT001 -- for testing
    """Test the observable status update functions."""

    log_dir = tmpdir.mkdir("logs") if save_to_disk else None
    log = logger.Logger(log_dir=log_dir)

    name = "test_obs"
    log.update_observable_status(name, logger.Status.STARTED)

    if save_to_disk:
        fname = log_dir / logger.convert_to_fname(name)
        assert fname.exists()

    funcs = {
        logger.Status.STARTED: log.set_observable_started,
        logger.Status.RUNNING: log.set_observable_running,
        logger.Status.COMPLETE: log.set_observable_complete,
        logger.Status.ERROR: log.set_observable_error,
    }

    for status in logger.Status:
        log.update_observable_status(name, status)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"

        funcs[status](name)
        if save_to_disk:
            assert fname.readlines()[-1] == f"{name},{status}\n"
