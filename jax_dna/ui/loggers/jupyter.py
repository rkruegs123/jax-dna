"""Implements a plotly logger for use in Jupyter notebooks."""

import functools
import itertools
import math
import typing
import warnings
from pathlib import Path

import ipywidgets as widgets
import plotly
import plotly.graph_objects as go
import plotly.subplots
import typing_extensions

from jax_dna.ui.loggers import logger
from jax_dna.ui.loggers.logger import Status

LBL_TOP_HEADER = "Optimization Status"
LBL_PROG_BAR = "Optimizing"
LBL_SIM_HEADER = "Simulators"
LBL_OBS_HEADER = "Observables"
LBL_OBJ_HEADER = "Objectives"

WARN_INVALID_NCOLS_NROWS = (
    "The number of rows and columns is less than the number of plots. Adjusting the number of rows and columns."
)

figure_widget_f = go.FigureWidget
scatter_f = go.Scatter
make_subplots_f = plotly.subplots.make_subplots


def calc_rows_and_columns(
    n_plots: int,
    nrows: int | None,
    ncols: int | None,
) -> tuple[int, int]:
    """Calculate the number of rows and columns for the plot.

    Args:
        n_plots: the number of plots
        nrows: the number of rows in the plot
        ncols: the number of columns in the plot

    Returns:
        tuple[int, int]: the number of rows and columns
    """
    is_valid_nrows = nrows is not None and nrows > 0
    is_valid_ncols = ncols is not None and ncols > 0
    if is_valid_nrows and is_valid_ncols and nrows * ncols < n_plots:
        warnings.warn(WARN_INVALID_NCOLS_NROWS, UserWarning, stacklevel=1)
        is_valid_ncols = is_valid_nrows = False

    if not is_valid_nrows and not is_valid_ncols:
        nrows = ncols = int(math.ceil(math.sqrt(n_plots)))
    else:
        nrows = nrows if is_valid_nrows else int(math.ceil(n_plots / ncols))
        ncols = ncols if is_valid_ncols else int(math.ceil(n_plots / nrows))

    return nrows, ncols


def setup_figure_layout(
    fig: go.FigureWidget,
    nrows: int,
    ncols: int,
    trace_names: list[str | list[str]],
) -> None:
    """Setup the layout of the plotly figure.

    Args:
        fig: the plotly figure
        nrows: the number of rows in the plot
        ncols: the number of columns in the plot
        trace_names: the names of the traces
    """
    for i, ((row, col), names) in enumerate(
        zip(itertools.product(range(1, nrows + 1), range(1, ncols + 1)), trace_names, strict=False), start=1
    ):
        for name in (
            [
                names,
            ]
            if not isinstance(names, list)
            else names
        ):
            fig.add_trace(scatter_f(x=[], y=[], name=name), row=row, col=col)

        legend_name = f"legend{i+2}"
        axis_num = str(i) if i > 1 else ""
        fig.update_traces(row=row, col=col, legend=legend_name)
        fig.update_layout(
            {
                legend_name: {
                    "x": fig.layout["xaxis" + axis_num].domain[0],
                    "y": fig.layout["yaxis" + axis_num].domain[1],
                    "xanchor": "left",
                    "yanchor": "top",
                    "bgcolor": "rgba(0,0,0,0)",
                }
            }
        )


class PlotlyLogger:
    """A logger for use in Jupyter notebooks that uses plotly."""

    def __init__(
        self,
        observable_plots: list[str | list[str]],
        nrows: int | None,
        ncols: int | None,
        width_px: int | None = None,
        height_px: int | None = None,
    ) -> "PlotlyLogger":
        """Create a plotly logger for use in Jupyter notebooks.

        Args:
            observable_plots (list[str | list[str]]): a list of the names of the observables to plot
            nrows (int | None): the number of rows in the plot
            ncols (int | None): the number of columns in the plot
            width_px (int | None): the width of the figure in pixels
            height_px (int | None): the height of the figure in pixels
            log_dir (str|Path|None): the directory to save the logs to
        """
        nrows, ncols = calc_rows_and_columns(len(observable_plots), nrows, ncols)
        self.fig = figure_widget_f(make_subplots_f(rows=nrows, cols=ncols))
        if width_px is not None or height_px is not None:
            self.fig.update_layout(
                autosize=False,
                width=width_px,
                height=height_px,
            )

        self.observable_plots = observable_plots
        setup_figure_layout(self.fig, nrows, ncols, observable_plots)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log a metric to the plotly figure."""
        graph_obj = next(filter(lambda f: f.name == name, self.fig.data))
        graph_obj.x += (step,)
        graph_obj.y += (value,)

    def change_size(
        self,
        width_px: int | None = None,
        height_px: int | None = None,
    ) -> None:
        """Change the size of the plotly figure.

        Args:
            width_px (int | None): the width of the figure in pixels
            height_px (int | None): the height of the figure in pixels
        """
        self.fig.update_layout(
            autosize=False,
            width=width_px,
            height=height_px,
        )

    def show(self) -> go.FigureWidget:
        """Show the plotly figure in a Jupyter notebook."""
        return self.fig


class JupyterLogger(logger.Logger):
    """A logger for use in Jupyter notebooks."""

    STATUS_STYLE: typing.ClassVar[dict[Status, dict[str, str]]] = {
        Status.STARTED: {"button_style": "primary", "icon": ""},
        Status.RUNNING: {"button_style": "info", "icon": "hourglass-half"},
        Status.COMPLETE: {"button_style": "success", "icon": "check"},
        Status.ERROR: {"button_style": "danger", "icon": "exclamation"},
    }

    def __init__(
        self,
        simulators: list[str],
        observables: list[str],
        objectives: list[str],
        metrics_to_log: list[list[str] | str],
        max_opt_steps: int,
        plots_size_px: tuple[int, int] | None = None,
        plots_nrows_ncols: tuple[int, int] | None = None,
        log_dir: str | Path | None = None,
    ) -> "JupyterLogger":
        """Initialize the Jupyter dashboard.

        Args:
            simulators (list[str]): the names of the simulators
            observables (list[str]): the names of the observables
            objectives (list[str]): the names of the objectives
            metrics_to_log (list[list[str]|str]): the metrics to log
            max_opt_steps (int): the maximum number of optimization steps
            plots_size_px (tuple[int,int]|None): the size of the plots in pixels
            plots_nrows_ncols (tuple[int,int]|None): the number of rows and columns in the plots
            log_dir (str|Path|None): the directory to save the logs to, passed to logger.Logger
        """
        super().__init__(log_dir)

        self.prog_bar = widgets.IntProgress(
            min=0, max=max_opt_steps, description=LBL_PROG_BAR, bar_style="info", orientation="horizontal"
        )

        btn_f = functools.partial(
            widgets.Button,
            disabled=True,
            **JupyterLogger.STATUS_STYLE[Status.STARTED],
        )

        self.sim_btns = [
            btn_f(
                description=sim,
            )
            for sim in simulators
        ]
        self.obs_btns = [btn_f(description=obs) for obs in observables]
        self.obj_btns = [btn_f(description=obj) for obj in objectives]

        nrows, ncols = plots_nrows_ncols if plots_nrows_ncols else (None, None)
        width_px, height_px = plots_size_px if plots_size_px else (None, None)

        self.plots = PlotlyLogger(
            metrics_to_log,
            nrows=nrows,
            ncols=ncols,
            width_px=width_px,
            height_px=height_px,
        )
        self.percent_complete = widgets.Label(value="0%")
        self.dashboard = widgets.VBox(
            [
                widgets.Label(value=LBL_TOP_HEADER),
                widgets.HBox([self.prog_bar, self.percent_complete]),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.Label(value=LBL_SIM_HEADER),
                                *self.sim_btns,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.Label(value=LBL_OBS_HEADER),
                                *self.obs_btns,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.Label(value=LBL_OBJ_HEADER),
                                *self.obj_btns,
                            ]
                        ),
                    ]
                ),
                self.plots.show(),
            ]
        )

    def show(self) -> widgets.DOMWidget:
        """Show the Jupyter dashboard."""
        return self.dashboard

    def increment_prog_bar(self, value: int = 1) -> None:
        """Increment the progress bar by `value`."""
        self.prog_bar.value += value
        self.percent_complete.value = f"{(self.prog_bar.value / self.prog_bar.max) * 100:.2f}%"

    @typing_extensions.override
    def log_metric(self, name: str, value: float, step: int) -> None:
        self.plots.log_metric(name, value, step)

    def _update_status(self, btns: list[widgets.Button], name: str, status: Status) -> None:
        """Updates the status of a simulator, objective, or observable."""
        next(filter(lambda btn: btn.description == name, btns)).set_state(JupyterLogger.STATUS_STYLE[status])

    @typing_extensions.override
    def update_simulator_status(self, name: str, status: Status) -> None:
        """Updates the status of a simulator."""
        self._update_status(self.sim_btns, name, status)

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
        self._update_status(self.obj_btns, name, status)

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
        self._update_status(self.obs_btns, name, status)

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
