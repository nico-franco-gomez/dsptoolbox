Deployment
==========

This project contains a PyO3 extension written in Rust. The Python package is
built with ``maturin`` and the Rust crate is compiled into the private
``dsptoolbox._rust`` extension module.

End users who install a wheel from PyPI do not need Rust or Cargo. The release
maintainer is responsible for building and uploading wheels for the supported
Python versions, operating systems, and architectures.

Prerequisites
-------------

Install the following before making a release:

* A supported Python interpreter.
* Rust and Cargo, available on ``PATH``.
* The project development requirements, including ``maturin`` and ``twine``.

From the repository root, create or activate the project environment and run::

    python -m pip install -r requirements-dev.txt

Check the tools before continuing::

    rustc --version
    cargo --version
    maturin --version
    twine --version

Prepare A Release
-----------------

Update the package version in both places that currently define it:

* ``dsptoolbox/__init__.py``: ``__version__``
* ``Cargo.toml``: ``package.version``

Use the same version in both files. PyPI versions should be unique and should
not be reused after an upload.

Run the checks before building::

    python -m pytest
    cargo fmt --check
    cargo check

Build The Distribution
----------------------

Remove artifacts from an earlier build and create a release wheel and source
distribution::

    rm -rf dist target
    mkdir -p dist
    maturin build --release --interpreter python --out dist
    maturin sdist --out dist

The wheel is specific to the Python interpreter, operating system, and
architecture used for the build. For example, a macOS ARM64 build for Python
3.14 produces a wheel with a name similar to::

    dsptoolbox-0.10.3-cp314-cp314-macosx_11_0_arm64.whl

Because this repository does not use GitHub Actions, repeat the wheel build
manually on each supported platform and architecture. Build once for each
supported Python version on that machine, using the corresponding interpreter
with ``--interpreter``. Keep the source distribution from only one build; it is
not platform-specific.

Inspect the artifacts before uploading::

    ls -lh dist
    python -m twine check dist/*

It is also useful to install the wheel into a clean virtual environment and
verify that the compiled module is present::

    python -m venv /tmp/dsptoolbox-release-check
    /tmp/dsptoolbox-release-check/bin/python -m pip install dist/*.whl
    /tmp/dsptoolbox-release-check/bin/python -c \
        "from dsptoolbox._rust import warp_time_series; print(warp_time_series)"

Upload To TestPyPI
------------------

Use TestPyPI when a release needs an upload-level check. Configure PyPI
credentials through Twine's supported credential mechanisms; do not commit
tokens to the repository. Upload the artifacts with::

    python -m twine upload --repository testpypi dist/*

Install the uploaded version from TestPyPI, while using the main PyPI index for
runtime dependencies::

    python -m pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        dsptoolbox==VERSION

Run the focused transform tests in that clean environment before publishing to
the main index.

Upload To PyPI
--------------

After the artifacts have been checked, upload them to the production index::

    python -m twine upload dist/*

Verify the release from a fresh environment rather than the repository
checkout::

    python -m venv /tmp/dsptoolbox-pypi-check
    /tmp/dsptoolbox-pypi-check/bin/python -m pip install dsptoolbox==VERSION
    /tmp/dsptoolbox-pypi-check/bin/python -c \
        "from dsptoolbox._rust import warp_time_series; print(warp_time_series)"

If a user's platform does not have a matching wheel, ``pip`` may fall back to
the source distribution. Building from that source distribution requires a
working Rust toolchain and Cargo, so publishing wheels for the supported target
matrix is important.