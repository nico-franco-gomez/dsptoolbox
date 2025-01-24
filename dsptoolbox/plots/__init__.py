"""
Plots
-----
This module contains plotting templates that use matplotlib and seaborn for
styling.

- `general_matrix_plot()`
- `general_plot()`
- `general_subplots_line()`
- `show()`

"""

from .plots import (
    general_matrix_plot,
    general_plot,
    general_subplots_line,
    general_plot_two_axes,
    show,
)

__all__ = [
    "general_matrix_plot",
    "general_plot",
    "general_plot_two_axes",
    "general_subplots_line",
    "show",
]
