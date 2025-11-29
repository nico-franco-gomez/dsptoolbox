.. image:: docs/logo/logo.png
   :width: 800
   :align: center

------------------------------------------------------------------------------

.. image:: https://readthedocs.org/projects/dsptoolbox/badge/?version=latest
    :target: https://dsptoolbox.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/l/dsptoolbox?color=gr
    :target: https://en.wikipedia.org/wiki/MIT_License
    :alt: License

.. image:: https://img.shields.io/pypi/pyversions/dsptoolbox
    :target: https://www.python.org/downloads/release/python-3100/
    :alt: Python version

.. image:: https://img.shields.io/pypi/v/dsptoolbox?color=orange
    :target: https://pypi.org/project/dsptoolbox/
    :alt: PyPI version

Readme
======

This is a toolbox in form of a python package that contains algorithms to be used in dsp (digital signal processing) research projects.

This is kind of a "sandbox" project with many different experimental implementations across a variety of DSP-related topics. Some parts are more
thoroughly tested and validated than others, so "caution" is advised. Please feel free to reach out in case you find bugs or want
to talk about certain functionality.

It is under active development and it will take some time until it reaches a certain level of maturity. Beware that backwards compatibility is not an actual concern and significant
changes to the API might come in the future. If you find some implementations interesting or useful, please feel free to use it for your projects
and expand or change functionalities.

Getting Started
===============

Check out the `examples`_ for some basic examples of the dsptoolbox package
and refer to the `documentation`_ for the complete description of classes and functions.

Installation
============

Use pip to install dsptoolbox

.. code-block:: console

    $ pip install dsptoolbox

    # Or this for activating numba parallelization
    $ pip install "dsptoolbox[use-numba]"

(Requires Python 3.11 or higher)

In order to install the package successfully using Linux, you need to install
PortAudio manually, since installing `sounddevice`_ will not do it automatically. To do this,
run the following commands on your console:

.. code-block:: console

    $ sudo apt-get install libasound-dev libportaudio2 libsndfile1

If this does not work properly for some reason, refer to the documentation for
`sounddevice`_ or `PortAudio`_.

.. _documentation: http://dsptoolbox.readthedocs.io/
.. _examples: https://github.com/nico-franco-gomez/dsptoolbox/tree/main/examples
.. _sounddevice: https://python-sounddevice.readthedocs.io/en/0.4.5/
.. _PortAudio: http://www.portaudio.com
