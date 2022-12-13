'''
Very specific plots which are harder to create from the general templates
'''
import matplotlib.pyplot as plt
import numpy as np


def _zp_plot(z, p, returns: bool = False):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    x = np.linspace(-1, 1, 100, endpoint=True)
    yP = np.sqrt(1 - x**2)
    yM = -np.sqrt(1 - x**2)
    ax.plot(x, yP, linestyle='dashed', alpha=0.6, color='xkcd:grey',
            label='Unit circle')
    ax.plot(x, yM, linestyle='dashed', alpha=0.6, color='xkcd:grey')
    ax.plot(np.real(z), np.imag(z), 'o', label='Zeros')
    ax.plot(np.real(p), np.imag(p), 'x', label='Poles')
    ax.legend()
    fig.tight_layout()
    if returns:
        return fig, ax
