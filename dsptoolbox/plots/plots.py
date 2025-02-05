"""
Includes some basic plotting templates
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from numpy import array, max, min, arange
from numpy.typing import NDArray

try:
    from seaborn import set_style

    set_style("whitegrid")
except ModuleNotFoundError as e:
    print("Seaborn will not be used for plotting: ", e)
    pass


def show():
    """Show created plots by using this wrapper around matplotlib's show."""
    plt.show()


def general_plot(
    x: NDArray | None,
    matrix: NDArray,
    range_x=None,
    range_y=None,
    log: bool = True,
    labels=None,
    xlabel: str = "Frequency / Hz",
    ylabel: str | None = None,
    info_box: str | None = None,
    tight_layout: bool = True,
) -> tuple[Figure, Axes]:
    """Generic plot template.

    Parameters
    ----------
    x : array-like
        Vector for x axis. Pass `None` to generate automatically.
    matrix : NDArray[np.float64]
        Matrix with data to plot.
    range_x : array-like, optional
        Range to show for x axis. Default: None.
    range_y : array-like, optional
        Range to show for y axis. Default: None.
    log : bool, optional
        Show x axis as logarithmic. Default: `True`.
    labels : list or str, optional
        Labels for the drawn lines as list of strings. Default: `None`.
    xlabel : str, optional
        Label for x axis. Default: None.
    ylabel : str, optional
        Label for y axis. Default: None.
    info_box : str, optional
        String containing extra information to be shown in a info box on the
        plot. Default: None.
    tight_layout: bool, optional
        When `True`, tight layout is activated. Default: `True`.

    Returns
    -------
    fig, ax

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if matrix.ndim == 1:
        matrix = matrix[..., None]
    elif matrix.ndim > 2:
        raise ValueError("Only 1D and 2D-arrays are supported")
    if x is None:
        x = arange(matrix.shape[0])
    if labels is not None:
        if type(labels) not in (list, tuple):
            assert type(labels) is str, "labels should be a list or a string"
            labels = [labels]
    if labels is not None:
        ax.plot(x, matrix, label=labels)
    else:
        ax.plot(x, matrix)
    if log:
        ax.set_xscale("log")
        ticks = array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if range_x is not None:
            ticks = ticks[(ticks > range_x[0]) & (ticks < range_x[-1])]
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.grid(True, which="minor")
    if range_x is not None:
        ax.set_xlim(range_x)
    if range_y is not None:
        ax.set_ylim(range_y)
    if labels is not None:
        ax.legend()
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if info_box is not None:
        ax.text(
            0.1,
            0.5,
            info_box,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
        )
    if tight_layout:
        fig.tight_layout()
    return fig, ax


def general_plot_two_axes(
    x1: NDArray | None,
    matrix1: NDArray,
    x2: NDArray | None,
    matrix2: NDArray,
    range_x=None,
    range_y1=None,
    range_y2=None,
    log_x: bool = True,
    labels1=None,
    labels2=None,
    xlabel: str = "Frequency / Hz",
    y1label: str | None = None,
    y2label: str | None = None,
    y1_linestyle: str | None = None,
    y2_linestyle: str | None = None,
    y1_alpha: float = 1.0,
    y2_alpha: float = 1.0,
    info_box: str | None = None,
    tight_layout: bool = True,
) -> tuple[Figure, Axes]:
    """Plot template for two y-axis with the same x-axis.

    Parameters
    ----------
    x1 : array-like
        Vector for x axis. Pass `None` to generate automatically.
    matrix1 : NDArray[np.float64]
        Matrix with data to plot. The first dimension is expected to match the
        x data.
    x2 : array-like
        Vector for x axis. Pass `None` to generate automatically.
    matrix2 : NDArray[np.float64]
        Matrix with data to plot. The first dimension is expected to match the
        x data.
    range_x : array-like, optional
        Range to show for x axis. Default: None.
    range_y1 : array-like, optional
        Range to show for y axis. Default: None.
    range_y2 : array-like, optional
        Range to show for second y axis. Default: None.
    log_x : bool, optional
        Show x axis as logarithmic. Default: `True`.
    labels1 : list or str, optional
        Labels for the drawn lines as list of strings. Default: `None`.
    labels2 : list or str, optional
        Labels for the drawn lines as list of strings. Default: `None`.
    xlabel : str, optional
        Label for x axis. Default: None.
    y1label : str, optional
        Label for y axis. Default: None.
    y2label : str, optional
        Label for y axis. Default: None.
    y1_linestyle : str, None, optional
        Line style for first axis. Default: None.
    y2_linestyle : str, None, optional
        Line style for second axis. Default: None.
    y1_alpha : float, optional
        Alpha value for first axis. Default: 1.
    y2_alpha : float, optional
        Alpha value for second axis. Default: 1.
    info_box : str, optional
        String containing extra information to be shown in a info box on the
        plot. Default: None.
    tight_layout: bool, optional
        When `True`, tight layout is activated. Default: `True`.

    Returns
    -------
    fig, ax

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax.twinx()

    # ====== First axis
    if matrix1.ndim == 1:
        matrix1 = matrix1[..., None]
    elif matrix1.ndim > 2:
        raise ValueError("Only 1D and 2D-arrays are supported")
    if x1 is None:
        x1 = arange(matrix1.shape[0])
    if labels1 is not None:
        if type(labels1) not in (list, tuple):
            assert type(labels1) is str, "labels should be a list or a string"
            labels1 = [labels1]
    if labels1 is not None:
        ax.plot(
            x1, matrix1, label=labels1, linestyle=y1_linestyle, alpha=y1_alpha
        )
    else:
        ax.plot(x1, matrix1, linestyle=y1_linestyle, alpha=y1_alpha)
    # ======

    # ====== Second axis
    if matrix2.ndim == 1:
        matrix2 = matrix2[..., None]
    elif matrix2.ndim > 2:
        raise ValueError("Only 1D and 2D-arrays are supported")
    if x2 is None:
        x2 = arange(matrix2.shape[0])
    if labels2 is not None:
        if type(labels2) not in (list, tuple):
            assert type(labels2) is str, "labels should be a list or a string"
            labels2 = [labels2]
    if labels2 is not None:
        ax2.plot(
            x2, matrix2, label=labels2, linestyle=y2_linestyle, alpha=y2_alpha
        )
    else:
        ax2.plot(x2, matrix2, linestyle=y2_linestyle, alpha=y2_alpha)
    # ======
    if log_x:
        ax.set_xscale("log")
        ticks = array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if range_x is not None:
            ticks = ticks[(ticks > range_x[0]) & (ticks < range_x[-1])]
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.grid(True, which="minor")
    if range_x is not None:
        ax.set_xlim(range_x)
    if range_y1 is not None:
        ax.set_ylim(range_y1)
    if range_y2 is not None:
        ax2.set_ylim(range_y2)
    if y1label is not None:
        ax.set_ylabel(y1label)
    if y2label is not None:
        ax2.set_ylabel(y2label)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if info_box is not None:
        ax.text(
            0.1,
            0.5,
            info_box,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="grey", alpha=0.75),
        )
    if labels1 is not None:
        ax.legend()
    if labels2 is not None:
        ax2.legend()
    if tight_layout:
        fig.tight_layout()
    return fig, ax


def general_subplots_line(
    x: NDArray | None,
    matrix: NDArray,
    column: bool = True,
    sharex: bool = True,
    sharey: bool = False,
    log: bool = False,
    xlabels=None,
    ylabels=None,
    range_x=None,
    range_y=None,
) -> tuple[Figure, list[Axes]]:
    """Generic plot template with subplots in one column or row.

    Parameters
    ----------
    x : array-like
        Vector for x axis. The same x vector is used for all subplots. Pass
        `None` to generate automatically.
    matrix : NDArray[np.float64]
        Matrix with data to plot.
    column : bool, optional
        When `True`, the subplots are organized in one column. Default: `True`.
    sharex : bool, optional
        When `True`, all subplots share the same values for the x axis.
        Default: `True`.
    sharey : bool, optional
        When `True`, all subplots share the same values for the y axis.
        Default: `False`.
    log : bool, optional
        Show x axis as logarithmic. Default: `False`.
    xlabels : array_like, optional
        Labels for x axis. Default: None.
    ylabels : array_like, optional
        Labels for y axis. Default: None.
    range_x : array-like, optional
        Range to show for x axis. Default: None.
    range_y : array-like, optional
        Range to show for y axis. Default: None.

    Returns
    -------
    fig, ax

    """
    if matrix.ndim == 1:
        matrix = matrix[..., None]
    elif matrix.ndim > 2:
        raise ValueError("Unsupported dimension. Matrix must be a 2D-array")
    number_of_channels = matrix.shape[1]
    if column:
        fig, ax = plt.subplots(
            number_of_channels,
            1,
            sharex=sharex,
            figsize=(8, 2 * number_of_channels),
            sharey=sharey,
        )
    else:
        fig, ax = plt.subplots(
            1,
            number_of_channels,
            sharex=sharex,
            figsize=(2 * number_of_channels, 8),
            sharey=sharey,
        )
    if number_of_channels == 1:
        ax = [ax]
    if x is None:
        x = arange(matrix.shape[0])
    for n in range(number_of_channels):
        ax[n].plot(x, matrix[:, n])
        if log:
            ax[n].set_xscale("log")
            ticks = array(
                [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            )
            if range_x is not None:
                ticks = ticks[(ticks > range_x[0]) & (ticks < range_x[-1])]
            ax[n].set_xticks(ticks)
            ax[n].get_xaxis().set_major_formatter(ScalarFormatter())
        if ylabels is not None:
            ax[n].set_ylabel(ylabels[n])
        if xlabels is not None:
            if not type(xlabels) is str and len(xlabels) > 1:
                ax[n].set_xlabel(xlabels[n])
        if range_x is not None:
            ax[n].set_xlim(range_x)
        if range_y is not None:
            ax[n].set_ylim(range_y)
    if type(xlabels) is str or len(xlabels) == 1:
        ax[-1].set_xlabel(xlabels)
    fig.tight_layout()
    return fig, ax


def general_matrix_plot(
    matrix,
    range_x=None,
    range_y=None,
    range_z: float | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zlabel: str | None = None,
    xlog: bool = False,
    ylog: bool = False,
    colorbar: bool = True,
    cmap: str = "magma",
    lower_origin: bool = True,
) -> tuple[Figure, Axes]:
    """Generic plot template for a matrix's heatmap.

    Parameters
    ----------
    matrix : NDArray[np.float64]
        Matrix with data to plot.
    range_x : array-like, optional
        Range to show for x axis. Default: `None`.
    range_y : array-like, optional
        Range to show for y axis. Default: `None`.
    range_z : float, optional
        Dynamic range to show. Default: `None`.
    xlabel : str, optional
        Label for x axis. Default: `None`.
    ylabel : str, optional
        Label for y axis. Default: `None`.
    zlabel : str, optional
        Label for z axis. Default: `None`.
    xlog : bool, optional
        Show x axis as logarithmic. Default: `False`.
    ylog : bool, optional
        Show y axis as logarithmic. Default: `False`.
    colorbar : bool, optional
        When `True`, a colorbar for zaxis is shown. Default: `True`.
    cmap : str, optional
        Type of colormap to use from matplotlib.
        See https://matplotlib.org/stable/tutorials/colors/colormaps.html.
        Default: `'magma'`.
    lower_origin : bool, optional
        When `True`, the origin of the vertical axis of the matrix is put
        below. Default: `True`.

    Returns
    -------
    fig, ax

    """
    assert matrix.ndim == 2, "Only 2D-arrays are supported for this plot type"
    extent = None
    if range_x is not None:
        assert range_y is not None, (
            "When x range is given, y range is " + "also necessary"
        )
        assert len(range_x) == 2 and len(range_y) == 2, (
            "xrange and or yrange are invalid. Please give a list "
            + "containing (min, max) values"
        )
        extent = (range_x[0], range_x[1], range_y[0], range_y[1])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    cmap2 = cm.get_cmap(cmap)
    if range_z is not None:
        max_val = max(matrix)
        min_val = max_val - range_z
    else:
        max_val = max(matrix)
        min_val = min(matrix)

    if lower_origin:
        origin = "lower"
    else:
        origin = "upper"

    if extent is None:
        col = ax.imshow(
            matrix,
            alpha=0.95,
            cmap=cmap2,
            vmin=min_val,
            vmax=max_val,
            origin=origin,
            aspect="auto",
        )
    else:
        col = ax.imshow(
            matrix,
            extent=extent,
            alpha=0.95,
            cmap=cmap2,
            vmin=min_val,
            vmax=max_val,
            origin=origin,
            aspect="auto",
        )
    if colorbar:
        if zlabel is not None:
            fig.colorbar(col, ax=ax, label=zlabel)
        else:
            fig.colorbar(col, ax=ax)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
        ticks = array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if range_y is not None:
            ticks = ticks[(ticks > range_y[0]) & (ticks < range_y[-1])]
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    return fig, ax
