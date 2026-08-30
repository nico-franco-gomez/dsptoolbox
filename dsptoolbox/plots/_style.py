def use_default_style():
    """Apply the package's default seaborn style to matplotlib's global
    settings.

    This is not done when importing the library, since it would change the
    look of every other plot in the user's session. Call it explicitly to get
    the styling that the plots in this package were designed with.

    Returns
    -------
    bool
        True when the style was applied, False when seaborn is not installed.

    """
    try:
        from seaborn import set_style
    except ModuleNotFoundError:
        return False

    set_style("whitegrid")
    return True
