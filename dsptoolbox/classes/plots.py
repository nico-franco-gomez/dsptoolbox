"""
Very specific plots which are harder to create from the general templates
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from ..helpers.other import find_nearest_points_index_in_vector
from ..helpers.gain_and_level import to_db


def _zp_plot(z, p):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    circle = plt.Circle(
        (0, 0),
        1,
        fill=False,
        linestyle="dashed",
        alpha=0.6,
        color="xkcd:grey",
        label="Unit circle",
    )
    ax.add_patch(circle)
    ax.plot(np.real(z), np.imag(z), "o", label="Zeros")
    ax.plot(np.real(p), np.imag(p), "x", label="Poles")
    ax.set_xlabel(r"$Re(z)$")
    ax.set_ylabel(r"$Im(z)$")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def _csm_plot(f, csm, range_x=None, log=True, with_phase=True):
    """Function to plot cross-spectral matrix. Since it is very specialized,
    it is not in the plots module.

    """
    ch = csm.shape[1]
    if range_x is not None:
        id0, id1 = find_nearest_points_index_in_vector(range_x, f)
    else:
        id0, id1 = 0, -1
    f = f[id0:id1]
    csm = csm[id0:id1]
    fig, ax = plt.subplots(
        ch, ch, figsize=(2.5 * ch, 2.5 * ch), sharex=True, sharey=True
    )
    for c1 in range(ch):
        ax[c1, 0].set_ylabel("dB")
        for c2 in range(ch):
            if log:
                ax[c1, c2].set_xscale("log")
                ticks = np.array(
                    [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
                )
                if range_x is not None:
                    ticks = ticks[(ticks > range_x[0]) & (ticks < range_x[-1])]
                ax[c1, c2].set_xticks(ticks)
                ax[c1, c2].get_xaxis().set_major_formatter(ScalarFormatter())
            ax[c1, c2].plot(f, to_db(csm[:, c1, c2], False))
            if c1 != c2:
                axRight = ax[c1, c2].twinx()
                axRight.plot(
                    f,
                    np.unwrap(np.angle(csm[:, c1, c2])),
                    alpha=0.6,
                    color="xkcd:orange",
                    linestyle="dotted",
                )
                axRight.grid(False)
            if c1 == ch - 1:
                ax[c1, c2].set_xlabel("Hz")
            if c2 == ch - 1:
                axRight.set_ylabel("rad")
    fig.tight_layout()
    return fig, ax
