Changelog
=========

All notable changes to `dsptoolbox
<https://github.com/nico-franco-gomez/dsptoolbox>`_ will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

`Unreleased`_
-------------

- Automated testing with pytest
- Validation of transfer functions and coherence with some external tool
- Fractional time delay filter
- Beamforming module

`0.0.3`_ - 2023-01-05
---------------------

Added
~~~~~

- added resampling using ``scipy.signal.resample_poly``
- added distance measures: snr, si-sdr
- added ``normalize`` function
- added ``get_ir`` method to ``FilterBank`` class
- added function to load pickle objects
- added changelog
- added support for ``MultiBandSignal`` input in ``reverb_time`` function
- added ``get_channel`` method in ``Signal`` class for retrieving specific channels from signal as signal objects
- introduced support for 1d-arrays in plot functions and raise error if ndim>2
- added property and specialized setter for multiple sampling rates in FilterBank and MultiBandSignal
- ``get_stream_samples`` added in ``Signal`` class for streaming purposes
- added ``fade`` method for signals

Bugfix
~~~~~~

- corrected a bug regarding filter order
- corrected documentation for ``__init__`` Filter biquad, ``find_room_modes``, 
- change assert order in merge signal function
- corrected errors in test file
- corrected copying signals in `_filter.py` functions and ``MultiBandSignal.collapse`` method
- references in pyfar functions corrected
- bug fix in normalize function
- minor bug fixes

Misc
~~~~

- dropped multichannel parameter in spectral deconvolve and get transfer function
- changed to dynamic versioning to building package with hatch
- when plotting, general plot can now take flat arrays as arguments
- readme edited
- package structure updated
- general updates to docstrings
- extended merging signals while trimming or padding in the end and in the beginning
- changed module name from `measure` to `audio_io`
- refactored ``time_vector_s`` handling in ``Signal`` class