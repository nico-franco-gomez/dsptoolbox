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

from ._style import _apply_default_plot_style
from .plots import (
    general_matrix_plot,
    general_plot,
    general_plot_two_axes,
    general_subplots_line,
    show,
)

_apply_default_plot_style()

__all__ = [
    "general_matrix_plot",
    "general_plot",
    "general_plot_two_axes",
    "general_subplots_line",
    "show",
]
