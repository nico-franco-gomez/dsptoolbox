import numpy as np

RngLike = np.random.Generator | np.random.SeedSequence | int | None
"""Accepted specifications for the random number generator of every stochastic
entry point: a `numpy.random.Generator` to use directly, a seed (`int` or
`numpy.random.SeedSequence`) to build one from, or None for a fresh,
unpredictably seeded generator."""


def _get_rng(rng: RngLike) -> np.random.Generator:
    """Turn a user-facing `rng` argument into a `numpy.random.Generator`.

    Parameters
    ----------
    rng : RngLike
        Generator, seed or None.

    Returns
    -------
    numpy.random.Generator
        The generator to draw from. A passed generator is returned as is, so
        that consecutive calls advance the same stream.

    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
