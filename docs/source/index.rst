Welcome to dsptool's documentation!
===================================

**dsptools** is a Python library for hosting general use Digital
Signal Processing (DSP) algorithms and some basic classes to keep
a consistent and easy-to-use and easy-to-expand data structure.

The design philosophy behind this library is that the end user does not
have to modify numpy arrays directly but the library should always offer
a function for general procedures in DSP tasks. It should also be
easily expandable and written with well-documented, easy-to-read code.
Its main use is for rapid-prototyping of offline dsp algorithms, though
some real-time functionalities might be added with time.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development. It is primarily a personal
   project that started as a way to facitilate learning and applying
   different dsp algorithms with consistent data structures as foundation
   for this.
   
.. disclaimer::
    Some of the algorithms are almost directly taken from other libraries.
    This is always marked with a reference to the original source. There are
    two main reasons for not just using external libraries:
        1. Avoid dependencies that could cause trouble over time.
        2. Personal learning of the implementations and mathematical methods
            used.

Contents
--------

.. toctree::

   usage
   api
