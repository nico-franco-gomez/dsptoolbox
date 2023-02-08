Changelog
=========

All notable changes to `dsptoolbox
<https://github.com/nico-franco-gomez/dsptoolbox>`_ will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

`To Do's for future releases`_
------------------------------

- Validation for results from tests in every module (so far many tests are
  only regarding functionality)

`0.2.1 <https://pypi.org/project/dsptoolbox/0.2.1>`_ - 2023-02-08
---------------------

Added
~~~~~~
- ``plot_waterfall`` in special module
- beamforming algorithms added as a module called beamforming
- number of filters property in ``FilterBank``
- vectorized ``generators.noise`` for faster multi channel noise generation
- quadrature mirror filters crossovers

Bugfix
~~~~~~
- now the original signal length is used everywhere as an argument to ``numpy.fft.irfft``
  to avoid reconstruction issues for odd-length signals
- now ``Signal`` and ``Filter`` can not be created without explicitely passing a
  sampling rate
- corrected scaling when using ``_welch`` for spectrum and now clearer scalings
  can be passed
- allowed for 0 percent overlap when computing spectrum, csm or stft
- other minor fixes

Misc
~~~~~
- added automated testing using pytest (and changed requirements)
- added support for python 3.11
- extended and corrected docstrings
- change to warning instead of assertion error after not passing the COLA condition
  for stft, welch or csm
- optimized computation of cross-spectral matrix
- relocated some functions from standard to transfer functions module

`0.1.1 <https://pypi.org/project/dsptoolbox/0.1.1/>`_ - 2023-01-20
---------------------

Added
~~~~~~
- the method for finding room modes now includes the ``prune_antimodes`` 
  parameter which checks for modes that are dips in the room impulse response and leaves these out
- filter class can now plot magnitude directly with zero_phase filtering
- ``activity_detector`` added in standard module
- ``spectral_average`` in transfer_functions module
- ``generate_synthetic_rir`` in room_acoustics module

Bugfix
~~~~~~
- start of impulse responses for multibandsignals is now done for each signal separately
  since filtering could lead to different group delays in each band
- assertion that ``start_stop_hz`` is ``None`` when standard method is selected in ``transfer_functions.spectral_deconvolve()``
- _biquad_coefficients can now take strings as eq_type
- refactored part of filtering function in Linkwitz-Riley filter bank such that
  no unnecessary loops are used

Misc
~~~~~
- turned off warning if time_data_imaginary is called and happens to be None
- corrected or extended docstrings
- moved linear and minimum phase system generation from special to transfer_functions module

`0.1.0 <https://pypi.org/project/dsptoolbox/0.1.0/>`_ - 2023-01-13
---------------------

Added
~~~~~~
- GammaToneFilterBank with reconstruction capabilities
- fractional time delay in standard module
- delay_samples parameter for dirac signal
- polyphase representations in `_general_helpers.py`
- filtering and resampling has been implemented in the ``Filter`` class:
  if filter is iir normal filtering and downsampling (or the other way around
  for upsampling) is done. If filter is fir, an efficient polyphase representation is used
- ``log_mel_spectrogram`` and ``mel_filterbank`` added in special module

Bugfix
~~~~~
- time_data_imaginary gives now a copy of the time data
- energy normalization in distance measures now allows for scale-invariant comparison
- corrected sampling rate in plot generation for FilterBank

Misc
~~~~
- add image in the beginning of repository's readme


`0.0.5 <https://pypi.org/project/dsptoolbox/0.0.5/>`_ - 2023-01-11
---------------------

Added
~~~~~~
- stop_flag for ``stream_samples`` method of ``Signal`` class
- ``get_ir`` method for Linkwitz-Riley Filterbank class
- possibility to define a start for the RIR in the ``reverb_time`` method. Also
  the same start index is now used for all channels and bands
- sleep and output_stream to audio_io (wrappers around sounddevice's functions)
- ``min_phase_from_mag`` and ``lin_phase_from_mag`` in the special module.
- ``auditory_filters_gammatone`` filter bank.
- harmonic tone generator added in ``generators`` module
- grey noise in noise generator function
- ``find_ir_start`` in room_acoustics module
- ``Signal`` class can now handle complex time data by splitting real and imaginary
  parts in different properties (time_data and time_data_imaginary)
- ``swap_bands`` in ``MultiBandSignal`` class that allows reordering the bands
- ``swap_filters`` in ``FilterBank`` class that allows reordering the filters

Bug fixes
~~~~~~~~~~
- bug in _get_normalized_spectrum helper function
- bug in the order of the [filter] order vector in Linkwitz-Riley FliterBank class
- bug in ``Signal`` class where unwrapped phase could not be plotted correctly
- plots.general_plot can now use tight_layout() or not. Activating it could be
  counterproductive in cases where the legend is very large since it squishes the axes
- changed spectrum array dtype to cfloat to ensure that complex spectrum is always created

Misc
~~~~~
- changed function name ``play_stream`` to ``play_through_stream`` in audio_io module and the way it works
- extended and corrected docstrings
- ``Filter`` class can now handle complex output: a warning can be printed or not and the imaginary output is saved in the 
  ``Signal`` class' ``time_data_imaginary``. The warning is defined through ``warning_if_complex`` bool attribute
- newly improved filtering function for FIR filters that uses ``scipy.signal.convolve`` instead of ``numpy.convolve``


`0.0.4 <https://pypi.org/project/dsptoolbox/0.0.4/>`_ - 2023-01-05
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
- documentation fixed

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
