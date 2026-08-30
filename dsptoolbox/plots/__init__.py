"""
Plots
-----
This module contains plotting templates that use matplotlib for styling. Every
template and every `plot_*` method takes an `ax` argument, so that several
results can be drawn onto the same axes.

- `general_matrix_plot()`
- `general_plot()`
- `general_plot_two_axes()`
- `general_subplots_line()`
- `show()`
- `use_default_style()`: opt into the seaborn styling used for the plots in
  the documentation. Importing this library does not change matplotlib's
  global settings.

"""

from ._style import use_default_style
from .plots import (
    general_matrix_plot,
    general_plot,
    general_plot_two_axes,
    general_subplots_line,
    show,
)

__all__ = [
    "general_matrix_plot",
    "general_plot",
    "general_plot_two_axes",
    "general_subplots_line",
    "show",
    "use_default_style",
]
