Welcome to dsptool's documentation!
============================

**dsptools** is a Python library for hosting general use Digital
Signal Processing (DSP) algorithms and some basic classes to keep
a consistent, easy-to-use and easy-to-expand data structures.

The design philosophy behind this library is that the end user never has
to modify numpy arrays directly, but the library should always offer
a function for general procedures in DSP tasks. It should also be
easily expandable and written with well-documented, easy-to-read code.
Its main use is for rapid-prototyping of offline dsp algorithms, though
some real-time functionalities might get added with time.

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development. Important changes might come
   in the future and backward compatibility is not a concern at the moment.
   
.. disclaimer::
    Some of the algorithms are almost directly taken from other libraries.
    This is always marked with a reference to the original source. There are
    two main reasons for not just using external libraries:
        1. Avoid dependencies that could cause trouble over time.
        2. Personal learning of the implementations and mathematical methods
            used.

Documentation
============

.. toctree::

   classes
   
.. toctree::
    
   modules
   
   
Other
=====
