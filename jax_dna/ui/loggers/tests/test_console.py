"""Tests for the console logger."""

from jax_dna.ui.loggers import console


def test_log_metric(capfd):
    """Test logging a metric."""
    logger = console.ConsoleLogger()
    logger.log_metric("test", 1.0, 0)
    out, _ = capfd.readouterr()
    assert out == "Step: 0, test: 1.0\n"


def test_simulator_status_updates(capfd) -> None:
    """Test the simulator status update functions."""
    log = console.ConsoleLogger(log_dir=None)

    name = "test_sim"
    log.update_simulator_status(name, console.Status.STARTED)
    out, _ = capfd.readouterr()
    assert out == f"{name} {console.Status.STARTED}\n"

    funcs = {
        console.logger.Status.STARTED: log.set_simulator_started,
        console.logger.Status.RUNNING: log.set_simulator_running,
        console.logger.Status.COMPLETE: log.set_simulator_complete,
        console.logger.Status.ERROR: log.set_simulator_error,
    }

    for status in console.logger.Status:
        log.update_simulator_status(name, status)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"

        funcs[status](name)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"


def test_objective_status_updates(capfd) -> None:
    """Test the objective status update functions."""
    log = console.ConsoleLogger(log_dir=None)

    name = "test_sim"
    log.update_objective_status(name, console.Status.STARTED)
    out, _ = capfd.readouterr()
    assert out == f"{name} {console.Status.STARTED}\n"

    funcs = {
        console.logger.Status.STARTED: log.set_objective_started,
        console.logger.Status.RUNNING: log.set_objective_running,
        console.logger.Status.COMPLETE: log.set_objective_complete,
        console.logger.Status.ERROR: log.set_objective_error,
    }

    for status in console.logger.Status:
        log.update_objective_status(name, status)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"

        funcs[status](name)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"


def test_observable_status_updates(capfd) -> None:
    """Test the observable status update functions."""
    log = console.ConsoleLogger(log_dir=None)

    name = "test_sim"
    log.update_observable_status(name, console.Status.STARTED)
    out, _ = capfd.readouterr()
    assert out == f"{name} {console.Status.STARTED}\n"

    funcs = {
        console.logger.Status.STARTED: log.set_observable_started,
        console.logger.Status.RUNNING: log.set_observable_running,
        console.logger.Status.COMPLETE: log.set_observable_complete,
        console.logger.Status.ERROR: log.set_observable_error,
    }

    for status in console.logger.Status:
        log.update_observable_status(name, status)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"

        funcs[status](name)
        out, _ = capfd.readouterr()
        assert out == f"{name} {status}\n"
