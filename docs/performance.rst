Performance Optimizations
=========================

DSPToolbox keeps the public Python API while moving selected hot loops to
compiled backends:

* Numba is an optional Python dependency. When it is installed, selected
  numerical kernels are compiled and parallelized at runtime.
* Rust functions are compiled into the ``dsptoolbox._rust`` PyO3 extension.
  The package prefers these functions for selected realtime filters and
  transforms, and falls back to Python implementations when the extension is
  unavailable.

Optional Numba Backend
----------------------

Numba is an optional dependency. Install it with the package extra to enable
the accelerated paths::

    python -m pip install "dsptoolbox[use-numba]"

It accelerates computationally intensive loops such as the direct DFT,
complex spectral smoothing, and image-source room impulse response generation.
Without it, these operations use NumPy/Python implementations, so Numba is
not required at runtime.

The first call to a Numba-backed function can include compilation overhead.
Repeated calls can be faster because the kernels use caching where appropriate.

Compiled Rust Backend
---------------------

The Rust backend is part of the package build, rather than a Python optional
extra. ``maturin`` uses the PyO3 configuration in ``pyproject.toml`` to
compile the Rust crate in ``src/`` as ``dsptoolbox._rust``. Installed wheels
already contain this extension, so end users do not need Rust or Cargo when a
matching wheel is available.

The compiled functions cover performance-sensitive realtime filtering and
selected transforms, including FIR/IIR and parallel filter operations, the
Laguerre and warping transforms, and wavelet-related calculations. Python
modules import these functions conditionally and select them at runtime. A
source checkout without a built extension can therefore still use the Python
fallbacks for the affected operations, although realtime or large transform
workloads may be slower.

The Rust build and release workflow is described in :doc:`deployment`.