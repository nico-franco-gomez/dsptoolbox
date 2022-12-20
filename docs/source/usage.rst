Usage
=====

.. _installation:

Installation
------------

To use dsptools, first install it using pip:

.. code-block:: console

   (.venv) $ pip install dsptools

Creating signals
----------------

To start creating a signal object, you can provide a path to a
wav or flac file. Alternatively, you can just pass a time series vector/matrix (numpy array)
and a sampling rate in Hz.

.. autofunction:: dsptools.Signal(path='path/to/file/music.wav')
.. autofunction:: dsptools.Signal(time_data=my_time_series, sampling_rate_hz=44100)

For example:

>>> import dsptools as dsp
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

