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

`0.4.7 <https://pypi.org/project/dsptoolbox/0.4.7>`_ - 
---------------------
Added
~~~~~
- new `dft` in ``transforms`` for computing DFTs with any resolution
- `lpc` in ``transforms``
- `ExponentialAverageFilter` in ``filterbanks``
- support for python 3.13

Misc
~~~~
- improved precision of parallel filter by adding a third feed-forward
  coefficient to least-squares approximation
- replaced convolve with oaconvolve in multiple places for optimal handling
  with different signal lengths
- made framed signal methods available in ``dsptoolbox.tools``
- general doc corrections and additions
- added numba as new dependency for parallelizing some functions. It will be
  installed and used automatically if the current python environment is 3.12 or
  below. Support for numba and python 3.13 is not yet available.

Bugfix
~~~~~~
- fixed problem with group delay designer
- fixed a problem with array dimensions in autoregressive coefficients estimation

`0.4.6 <https://pypi.org/project/dsptoolbox/0.4.6>`_ - 
---------------------

Bugfix
~~~~~~
- corrected `excess_group_delay` due to different padding cases when removing
  the IR latency

`0.4.5 <https://pypi.org/project/dsptoolbox/0.4.5>`_ - 
---------------------
Added
~~~~~
- `FilterChain` in ``filterbanks`` for use in real-time applications
- `arma` in ``filterbanks`` for obtaining arbitrary IIR filter approximations
  to an impulse response

Misc
~~~~
- renamed smoothe to smoothing across the library
- zeros, poles and gain are now saved in `Filter`. They are returned instead
  of recomputing from the coefficients
- general doc improvements

Bugfix
~~~~~~
- corrected a bug where the time window of an impulse response did not match
  after some time-domain operation was applied to it
- fixed a problem with normalization in ``audio_io``
- fixed a problem with `Distortion` in ``effects``

`0.4.4 <https://pypi.org/project/dsptoolbox/0.4.4>`_ - 
---------------------
Added
~~~~~
- bark and erb approximations to warping factor
- `ParallelFilter` in ``filterbanks``
- `KautzFilter` in ``filterbanks``
- Realtime capabilities for filter `LatticeLadderFilter`, `StateVariableFilter`,
  `IIR`, `FIR`, `KautzFilter`
- `warp_filter` in ``transforms``
- `resample_filter` in ``standard``

Misc
~~~~
- moved `kautz` and `kautz_filters` functionality to `KautzFilter`

Bugfix
~~~~~~
- use the peak for `combine_ir_with_dirac` instead of delay with minimum-phase

`0.4.3 <https://pypi.org/project/dsptoolbox/0.4.3>`_ - 
---------------------
Added
~~~~~
- added `laguerre` to ``transforms``
- added `kautz` and `kautz_filters` to ``transforms``

Misc
~~~~
- energy decay curve is not corrected with compensation energy or pruned from
  noise when something during the estimation goes wrong (fallback strategy)
- updated README
- moved `warp_ir` to ``transforms`` and renamed to `warp`
- general documentation additions and fixes
- finding the end of an IR now also allows for defining a distance to noise
  floor

Bugfix
~~~~~~
- fixed a bug during the computation of the energy decay curve where
  phase-inverted peaks were not taken into account for the start of the
  impulse response

`0.4.2 <https://pypi.org/project/dsptoolbox/0.4.2>`_ - 
---------------------
Added
~~~~~~~
- `apply_gain` utility function in ``standard``
- beta parameter for arbitrary noise generation
- `GroupDelayDesigner` in ``filterbanks``
- nomalization of signals now accepts rms values

Misc
~~~~~
- frequency response interpolation with more interpolation modes
- refactored `PhaseLinearizer`

Bugfix
~~~~~~
- corrected a case where scaling of spectrum while plotting was wrong


`0.4.1 <https://pypi.org/project/dsptoolbox/0.4.1>`_ - 
---------------------

Bugfix
~~~~~~
- channel handling of ImpulseResponse


`0.4.0 <https://pypi.org/project/dsptoolbox/0.4.0>`_ - 
---------------------
Added
~~~~~~
- `ImpulseResponse` as a subclass of `Signal`. It handles time windows, coherence
  and plotting of those windows. Assertions for expected `ImpulseResponse` instead
  of `Signal` were added as well
- new module ``tools`` for computations with primitive data types, added time
  smoothing, interpolation of frequency response
- `get_transfer_function` in Filter and FilterBank
- analog-matched biquads in ``filterbanks``
- `gaussian_kernel` approximation in ``filterbanks``
- gain parameter functionality for some biquads
- new biquad types (lowpass and highpass first order, inverter)
- new explicit constructors for signal and filter
- pearson correlation as part quality estimator for latency computation
- new scaling parameter in synchrosqueezing of `cwt`
- new parameter in `window_frequency_dependent`

Bugfix
~~~~~~
- bugfix in `window_frequency_dependent` when querying a single frequency bin
- corrected plotting of spl when calibrated signal is passed

Misc
~~~~~~~
- got rid of signal type attribute. Use now `ImpulseResponse`
- general doc additions and fixes, type annotations
- `fractional_octave_smoothing` performance improved
- renamed some files of code base for consistency

`0.3.9 <https://pypi.org/project/dsptoolbox/0.3.9>`_ - 
---------------------
Added
~~~~~~
- `pinking_filter` in ``filterbanks`` module

Bugfix
~~~~~~
- fixed framed signal representation such that the last frames that need zero-padding
  can be left out
- biquad filter coefficients now use double precision by default
- minor fix in `window_frequency_dependent`

Misc
~~~~~~~
- added zero-padding while computing minimum phase ir for better results
- compatibility with numpy v2.0 has been ensured

`0.3.8 <https://pypi.org/project/dsptoolbox/0.3.8>`_ - 
---------------------

Misc
~~~~~~~
- renamed paramater `remove_impulse_delay` to `remove_ir_latency`
- changed default values in `PhaseLinearizer`
- general documentation improvements

Bugfix
~~~~~~
- `find_ir_latency` now searches for the latency in comparison to the minimum
  phase ir
- `harmonic_distortion_analysis` was fixed so that it can succesfully trim
  the fundamental ir

`0.3.7 <https://pypi.org/project/dsptoolbox/0.3.7>`_ - 
---------------------

Misc
~~~~~~~
- `trim_rir` has an improved approach where users do not need to set any
  parameters. It was also migrated to the ``transfer_functions`` module

Bugfix
~~~~~~
- `harmonics_from_chirp_ir` was fixed since it only searched for positive peaks
  in the IR to determine the impulse

`0.3.6 <https://pypi.org/project/dsptoolbox/0.3.6>`_ - 
---------------------

Added
~~~~~~~
- `set_latency` and `set_blocksize` in ``audio_io``
- `dither` in ``standard``

Misc
~~~~~~
- general documentation and small performance improvements

`0.3.5 <https://pypi.org/project/dsptoolbox/0.3.5>`_ - 
---------------------

Added
~~~~~~~
- `harmonic_distortion_analysis` in ``transfer_functions``
- added possibility of scaling the spectrogram
- calibration using any dBSPL value

Bugfix
~~~~~~~
- `reverb_time` now uses indices of peaks instead of -20 dBFS threshold since
  it delivers more accurate results
- now scaling a spectrum of a signal with a window is done correctly (taking
  the window into account)

Misc
~~~~~~
- general documentation and small performance improvements

`0.3.4 <https://pypi.org/project/dsptoolbox/0.3.4>`_ - 
---------------------

Added
~~~~~~~
- added support for `MultiBandSignal` in `hilbert` in module ``transforms``
- plot momentary spl added in `Signal`
- `PhaseLinearizer` can now adapt to an input group delay
- `find_modes` in ``room_acoustics`` can now find antiresonances and use a
  prominence value in dB for finding peaks in the CMIF
- `plot_phase` in signal class can now apply smoothing to the phase and also
  remove the delay of the impulse response
- `MultiBandSignal` can now return its time data

Bugfix
~~~~~~~
- a new criterion was added to `trim_rir` to reliably find the end of aqs RIR.
  It now looks at non-overlapping windows and expects the energy to decay
  monotonically after the impulse has arrived
- `window_centered_ir` fixed for certain lengths
- `generate_synthetic_rir` has been fixed after previous refactoring changed
  some underlying functions
- `noise` in ``generators`` has been now fixed since its previous slopes were
  erroneously defined in the amplitude spectrum instead of the power spectrum

Misc
~~~~~~
- general documentation and small performance improvements
- `window_frequency_dependent` is now optimized to be faster and can apply a
  window-dependent scaling to its output
- `MultiBandSignal` checks now for complex time data and ensures it is
  consistent in every band
- if `Signal` has `time_data_imaginary`, it is now also plotted in the
  `plot_time` method
- `get_spectrum` now returns the correctly scaled spectrum also when the method
  is standard
- updated some example notebooks
- `group_delay` functions in ``transfer_functions`` can apply now smoothing
- `reverb_time` now returns correlation coefficients as well
- corrected smoothing behavior in signal class when plotting


`0.3.3 <https://pypi.org/project/dsptoolbox/0.3.3>`_ - 
---------------------

Added
~~~~~~~
- added state variable filter `StateVariableFilter` discretized with a
  topology-preserving transform

Misc
~~~~~~
- Corrected orders for `linkwitz_riley_crossover` and added 2nd order

`0.3.1 <https://pypi.org/project/dsptoolbox/0.3.1>`_ - 
---------------------

Added
~~~~~~
- added returning the indices for start and stop in `trim_rir` in ``room_acoustics``

`0.3.0 <https://pypi.org/project/dsptoolbox/0.3.0>`_ - 
---------------------

Added
~~~~~~
- added `complementary_fir_filter` in ``filterbanks`` module
- `window_ir` in ``transfer_functions`` is now adaptive to the impulse
- added automatic trimming of room impulse responses for reverberation time
  and descriptors using a smooth envelope of the energy time curve. Additionally,
  added warning if `reverb_time` with Topt does not seem to find a good
  linear fit for the energy decay curve
- partly refactored `linkwitz_riley_crossover` and allow for odd order
  crossovers
- `PhaseLinearizer` in ``filterbanks`` module is now available for designing
  FIR filters to linearize a given phase response
- added `trim_rir` in ``room_acoustics`` for trimming RIRs in a parametrized
  manner

Bugfix
~~~~~~
- corrected scaling of spectrum in the case of amplitude spectrum in `signal`
  class
- corrected computation of minimum phase using log hilbert method
- corrected a case in `window_centered_ir` where padding was needed
- fixed a bug for `MultiBandSignal` where it could not add new bands in a
  multirate configuration

Misc
~~~~~
- docs and tests
- refactored `window_ir` for more flexibility and consistency
- now `compute_transfer_function` also returns the coherence
- change `LatticeLadderFilter` to be part of ``filterbanks`` module

`0.2.16 <https://pypi.org/project/dsptoolbox/0.2.16>`_ - 
---------------------
Added
~~~~~~
- renamed `spectral_average` into `average_irs` in ``transfer_functions``
  module. Now also a time-aligned average of irs can be done

Misc
~~~~~
- Refactored some backend functions

`0.2.14 <https://pypi.org/project/dsptoolbox/0.2.14>`_ - 
---------------------
Added
~~~~~~
- Distortion analysis of IR when measured with an exponential chirp

Bugfix
~~~~~~
- Selecting a bit depth for saving wav and flac files is now possible

`0.2.13 <https://pypi.org/project/dsptoolbox/0.2.13>`_ - 
---------------------
Added
~~~~~~
- ``reverb_time`` now has option ``Topt``

Bugfix
~~~~~~
- ``fade`` in ``log`` mode has been corrected to have the correct length
- ``istft`` in `transforms` module can handle different fft lengths

Misc
~~~~~~
- ``_welch`` is now faster when the autospectrum is computed

`0.2.12 <https://pypi.org/project/dsptoolbox/0.2.12>`_ - 
---------------------
Bugfix
~~~~~~
- ``window_frequency_dependent`` now handles frequency boundaries in vector
  properly

`0.2.11 <https://pypi.org/project/dsptoolbox/0.2.11>`_ - 
---------------------
Bugfix
~~~~~~
- bugfix in ``_check_ir_start_reverb``. Now any integer type can be used for the
  start indices
- ``combine_ir_with_dirac`` now takes into account the polarity of the original
  impulse response
- ``fractional_octave_smoothing`` can now clip values below 0

`0.2.10 <https://pypi.org/project/dsptoolbox/0.2.10>`_ - 
---------------------
Bugfix
~~~~~~
- bugfix in ``find_ir_latency``

`0.2.9 <https://pypi.org/project/dsptoolbox/0.2.9>`_ - 
---------------------
Added
~~~~~~
- ``find_ir_latency`` in `transfer_functions` module

Misc
~~~~~
- corrected and updated doc

`0.2.8 <https://pypi.org/project/dsptoolbox/0.2.8>`_ - 
---------------------
Added
~~~~~~
- ``warp_ir`` in the `transfer_functions` module
- ``LatticeLadderFilter`` in classes and standard module

Bugfix
~~~~~~~
- general bugfixes
- flake8 new standards applied, black formatter

Misc
~~~~~
- corrected and updated doc
- support for python 3.12 added

`0.2.7 <https://pypi.org/project/dsptoolbox/0.2.7>`_ - 
---------------------

Added
~~~~~~
- ``envelope`` function in standard module
- ``latency`` can now also compute subsample latency and handle multiband
  signals
- ``window_centered_ir``, ``spectrum_with_cycles`` and
  ``combine_ir_with_dirac`` in `transfer_functions`
- continuous wavelet transform with complex morlet wavelet and synchrosqueezing
  ``cwt``, ``MorletWavelet`` in `transforms`
- ``chroma_stft``, ``vqt``, ``hilbert`` and ``stereo_mid_side`` transforms in
  `transforms` module

Bugfix
~~~~~~~
- general bugfixes
- only local paths within package
- solved a bug where lfilter was not working properly for filtering IIR filters
  in ba mode
- biquads now only use ba and not sos
- ``reverb_time`` now can handle different options for the start of the IR
- now linkwitz-riley crossovers can also be done for odd orders since
  zero-phase filtering still gives perfect magnitude reconstruction. A warning
  is shown

Misc
~~~~~
- ``fractional_octave_smoothing`` is now done more efficiently and uses a
  hann window instead of hamming
- ``min_phase_ir``` uses now a real cepstrum method for obtaining the minimum
  phase. API has been modified
- ``window_ir`` now returns the start sample of the IR as well
- renamed `special` module into `transforms`
- ``chirp`` function now accepts a phase offset
- from now on, python 3.10 is no longer actively supported
- corrected and updated documentation
- dependencies have been updated

`0.2.6 <https://pypi.org/project/dsptoolbox/0.2.6>`_ - 
---------------------

Added
~~~~~~
- effects module with basic implementations for standard audio effects
- extra functionalities in the audio io module

Bugfix
~~~~~~~
- general bug fixes

Misc
~~~~~
- made seaborn optional

`0.2.5 <https://pypi.org/project/dsptoolbox/0.2.5>`_ - 
---------------------

Added
~~~~~~
- mel-frequency cepstral coefficients ``mfcc`` in ``special`` module
- spectrogram of a signal can now be plotted with a selected dynamic range
- ``audio_io`` has now more port functionalities to ``sounddevice``

Bugfix
~~~~~~~
- plotting for the ``qmf`` Crossover is now possible without downsampling
- Linkwitz-Riley crossovers plotting functions have been updated and corrected
- corrected some tests

Misc
~~~~~
- docstrings corrected and extended
- computation of steering vectors in ``beamforming`` has been optimized

`0.2.4 <https://pypi.org/project/dsptoolbox/0.2.4>`_ - 
---------------------

Added
~~~~~~
- ``rms`` function
- ``constrain_amplitude`` property to signal class is now used to enable
  or disable normalizing audio data that has higher amplitudes than 1. Also
  the factor by which the data is multiplied is now saved as the attribute
  ``amplitude_scale_factor``
- ``get_analytical_transfer_function`` in the ``ShoeboxRoom`` class
- ``ShoeboxRoom`` now can take additional information about absorption through
  the method ``add_detailed_absorption``. This is automatically used by both
  ``get_analytical_transfer_function`` and ``generate_synthetic_rir``
- ``generate_synthetic_rir`` can now limit the order of reflections to take
  into account and make use of the detailed absorption information stored
  in ``ShoeboxRoom``

Bugfix
~~~~~~~
- corrected a bug that caused saving an object to crash if the path contained
  a point that was not the format of the file

Misc
~~~~~
- docstrings corrected and extended

`0.2.3 <https://pypi.org/project/dsptoolbox/0.2.3>`_ - 2023-03-05
---------------------

Added
~~~~~~
- ``detrend`` function
- ``fractional_octave_bands`` filter bank in ``filterbanks`` module
- ``ShoeboxRoom`` class in ``room_acoustics``. Some basic room acoustics
  parameters can be computed. Used also for ``generate_synthetic_rir``

Bugfix
~~~~~~~
- corrected scaling in ``BeamformerFunctional`` so that the source power is
  not underestimated
- corrected ``plot_magnitude`` in ``FilterBank`` class where the second and
  subsequent bands were plotted with an offset

Misc
~~~~~
- docstrings corrected and extended
- renamed ``sinus`` to ``harmonic`` in ``generators`` module

`0.2.2 <https://pypi.org/project/dsptoolbox/0.2.2>`_ - 2023-02-21
---------------------

Added
~~~~~~
- New beamforming formulations added in ``beamforming`` module and renamed
  some formulations for better clarity

Bugfix
~~~~~~~
- minor fixes
- minimum phase IR now done for equiripple filters, linear-phase filters and
  general IR's with different methods

Misc
~~~~~
- docstrings corrected and extended
- refactored beamformer formulations for clearer inheritance structure

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
