======
Readme
======

This is a toolbox in form of a python package that contains algorithms to be used in dsp (digital signal processing) projects.

This project is under active development and it will take some time until it reaches a certain level of maturity. Beware that backwards compatibility is not an actual concern and important changes to the API might come in the future. If you find some implementations interesting or useful, please feel free to use it for your projects and expand or change
functionalities.

Getting Started
===============

Check out the `examples`_ for some basic examples of the dsptoolbox package
and refer to the `documentation`_ for the complete description of classes and functions.

Installation
============

Use pip to install dsptoolbox

.. code-block:: console

    $ pip install dsptoolbox

(Requires Python 3.11 or higher)

In order to install the package successfully using Linux, you need to install
PortAudio manually, since installing `sounddevice`_ will not do it automatically. To do this,
run the following commands on your console:

.. code-block:: console

    $ sudo apt-get install libasound-dev libportaudio2 libsndfile1

If this does not work properly for some reason, refer to the documentation for
`sounddevice`_ or `PortAudio`_.

For ASIO support on Windows, refer to `sounddevice`_.

.. _documentation: http://dsptoolbox.readthedocs.io/
.. _examples: https://github.com/nico-franco-gomez/dsptoolbox/tree/main/examples
.. _sounddevice: https://python-sounddevice.readthedocs.io/en/0.4.5/
.. _PortAudio: http://www.portaudio.com
