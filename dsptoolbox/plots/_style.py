def _apply_default_plot_style():
    """Apply the package's default seaborn style.

    Importing any module that plots reaches this once through the `plots`
    package, so the style is set in a single place. Seaborn is a hard
    dependency, but plotting must still work if only it is missing.

    """
    try:
        from seaborn import set_style

        set_style("whitegrid")
    except ModuleNotFoundError as e:
        print("Seaborn will not be used for plotting: ", e)
