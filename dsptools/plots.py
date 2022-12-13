'''
Includes some basic plotting templates
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import cm
import numpy as np
import seaborn
seaborn.set_style('whitegrid')


def show():
    '''
    Wrapper around matplotlib's show
    '''
    plt.show()


def general_plot(f, matrix, range_x=None, range_y=None, log: bool = True,
                 labels=None, xlabel: str = 'Frequency / Hz',
                 ylabel: str = None, info_box: str = None,
                 returns: bool = False):
    '''
    Generic plot for data.

    Parameters
    ----------
    f : array-like
        Vector for x axis.
    matrix : np.ndarray
        Matrix with data to plot.
    range_x : array-like, optional
        Range to show for x axis. Default: None.
    range_y : array-like, optional
        Range to show for y axis. Default: None.
    log : bool, optional
        Show x axis as logarithmic. Default: True.
    xlabel : str, optional
        Label for x axis. Default: None.
    ylabel : str, optional
        Label for y axis. Default: None.
    info_box : str, optional
        String containing extra information to be shown in a info box on the
        plot. Default: None.
    returns : bool, optional
        When `True`, the figure and axis are returned. Default: `False`.

    Returns
    -------
    When returns is activated, figure and axis are returned.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for n in range(matrix.shape[1]):
        if labels is not None:
            ax.plot(f, matrix[:, n], label=labels[n])
        else:
            ax.plot(f, matrix[:, n])
    if log:
        ax.set_xscale('log')
        ticks = \
            np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if range_x is not None:
            ticks = ticks[(ticks > range_x[0]) & (ticks < range_x[-1])]
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.xaxis.grid(True, which='minor')
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
        ax.text(0.1, 0.5, info_box, transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='grey', alpha=0.75))
    fig.tight_layout()
    if returns:
        return fig, ax


def general_subplots_line(x, matrix, column: bool = True,
                          sharex: bool = True, sharey: bool = False,
                          log: bool = False, ylabels=None, xlabels=None,
                          xlims=None, ylims=None,
                          returns: bool = False):
    '''
    Creates a generic plot with different subplots for each vector from the
    matrix.
    '''
    number_of_channels = matrix.shape[1]
    if column:
        fig, ax = plt.subplots(number_of_channels, 1, sharex=sharex,
                               figsize=(8, 2*number_of_channels),
                               sharey=sharey)
    else:
        fig, ax = plt.subplots(1, number_of_channels, sharex=sharex,
                               figsize=(2*number_of_channels, 8),
                               sharey=sharey)
    if number_of_channels == 1:
        ax = [ax]
    for n in range(number_of_channels):
        ax[n].plot(x, matrix[:, n])
        if log:
            ax[n].set_xscale('log')
            ticks = \
                np.array([20, 50, 100, 200, 500, 1000,
                          2000, 5000, 10000, 20000])
            if xlims is not None:
                ticks = ticks[(ticks > xlims[0]) & (ticks < xlims[-1])]
            ax[n].set_xticks(ticks)
            ax[n].get_xaxis().set_major_formatter(ScalarFormatter())
        if ylabels is not None:
            ax[n].set_ylabel(ylabels[n])
        if xlabels is not None:
            if not type(xlabels) == str and len(xlabels) > 1:
                ax[n].set_xlabel(xlabels[n])
        if xlims is not None:
            ax[n].set_xlim(xlims)
        if ylims is not None:
            ax[n].set_ylim(ylims)
    if type(xlabels) == str or len(xlabels) == 1:
        ax[-1].set_xlabel(xlabels)
    fig.tight_layout()

    if returns:
        return fig, ax


def general_matrix_plot(matrix, xrange=None, yrange=None, zrange=None,
                        xlabel=None, ylabel=None, zlabel=None,
                        xlog: bool = False, ylog: bool = False,
                        colorbar: bool = True, cmap: str = 'magma',
                        returns: bool = False):
    extent = None
    if xrange is not None:
        assert yrange is not None, 'When x range is given, y range is also ' +\
            'necessary'
        assert len(xrange) == 2 and len(yrange) == 2, \
            'xrange and or yrange are invalid. Please give a list ' +\
            'containing (min, max) values'
        extent = (xrange[0], xrange[1], yrange[0], yrange[1])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    cmap = cm.get_cmap(cmap)
    # cmap._init()
    # cmap._lut[-3, -1]
    if zrange is not None:
        max_val = np.max(matrix)
        min_val = max_val - zrange
    else:
        max_val = np.max(matrix)
        min_val = np.min(matrix)

    if extent is None:
        col = ax.imshow(
            matrix,
            alpha=0.95, cmap=cmap, vmin=min_val, vmax=max_val, origin='lower',
            aspect='auto')
    else:
        col = ax.imshow(
            matrix, extent=extent,
            alpha=0.95, cmap=cmap, vmin=min_val, vmax=max_val, origin='lower',
            aspect='auto')
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
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
        ticks = \
            np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        if yrange is not None:
            ticks = ticks[(ticks > yrange[0]) & (ticks < yrange[-1])]
        ax.set_yticks(ticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
    fig.tight_layout()
    if returns:
        return fig, ax
