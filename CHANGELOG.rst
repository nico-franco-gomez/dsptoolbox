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

Unreleased
---------------------
API breaks
~~~~~~~~~~
- The 14 realtime filter structures moved from ``filterbanks`` into the new
  ``dsptoolbox.realtime`` module, together with their `RealtimeFilter` base
  class. The FIR designers (`FirDesigner`, `PhaseLinearizer`,
  `GroupDelayDesigner`) stay in ``filterbanks``
- `Filter` is created through its factory methods (`from_ba`, `from_sos`,
  `from_zpk`, `fir_from_file`, `iir_filter`, `fir_filter`, `biquad`). The
  coefficients dictionary is now a private second constructor argument
- Functions whose return type was decided by an argument were split:
  `generators.chirp` always returns a `Signal` and the synchronized sweep has
  its own `generators.sync_log_chirp`, which also returns its effective
  duration; `MultiBandSignal.get_all_bands` and `get_all_time_data` require a
  common sampling rate and have `*_multirate` counterparts; `remove_filter`
  and `remove_band` return only the new object, `pop_filter` and `pop_band`
  return the removed one as well
- `add_filter`, `add_band` and `remove_channel` take `None` instead of `-1`
  to mean "at the end"
- `constrain_amplitude` now defaults to `False` everywhere, including
  `Signal.from_time_data` and `ImpulseResponse`. `activate_cache` is
  available on every constructor and factory
- Every `save_*` method and `load_pkl_object` take the format from the path's
  extension, which has to be present and correct. `LRFilterBank`'s stricter
  no-extension convention is gone
- `AudioEffect` is an abstract base class; effects keep no state between
  applications
- `audio_io.set_device()` requires a device: the interactive `input()` prompt
  is now `audio_io.list_devices()`. `default_config` became
  `get_default_config()`
- Importing the library no longer changes matplotlib's global settings nor
  sets `SD_ENABLE_ASIO`. Use `plots.use_default_style()` and
  `audio_io.enable_asio()`. sounddevice is imported on first use
- The last string selectors became enums: `IrLatencyRemoval` for
  `remove_ir_latency`, `SpectrumAverageMethod` for `average`, and
  `Power2Rounding` for `tools.next_power_2`
- `transforms.istft` takes either the original signal or a
  `SpectrogramParameters` with a sampling rate; the loose keyword arguments
  are gone. `distances.*` take a `SpectrumParameters` instead of a dictionary

Added
~~~~~
- `process_block()` on every realtime filter. The base implementation loops
  over `process_sample`; `IIRFilter`, `FIRFilter` and `FilterChain` filter a
  whole block at once
- `ax` on every plot template and `plot_*` method, so that several results
  can be drawn onto the same axes
- `MultiBandSignal` is a `MultichannelData`, so it has channel operations
  (`get_channels`, `remove_channel`, `swap_channels`, `sum_channels`), and it
  gained `resample`, `fade`, `trim_with_level_threshold`, `plot_magnitude`
  and `plot_time`
- `SpectrumParameters` and `SpectrogramParameters` frozen dataclasses, read
  from `Signal.spectrum_parameters` / `spectrogram_parameters` and applied
  with `with_spectrum_parameters()` / `with_spectrogram_parameters()`
- `verbose` on the beamformers, which are silent by default
- `rng` parameter on every stochastic entry point, so that results can be
  reproduced and seeded independently: `generators.noise`,
  `generators.oscillator`, `Signal.dither`, `effects.LFO`, `transforms.lpc`
  and `room_acoustics.generate_synthetic_rir`. It accepts a
  `numpy.random.Generator` (used and advanced as is), a seed, or None for the
  previous unpredictable behaviour
- `zero_phase` parameter on `FilterBank.plot_phase` and
  `FilterBank.plot_group_delay`, which the sibling `plot_magnitude` and
  `get_ir` already accepted

Bugfix
~~~~~~
- `Window.with_extra_parameter` stored the parameter on the enum member itself,
  so it leaked process-wide across unrelated objects; it now returns a
  `ParametrizedWindow`. Calling `to_scipy_format` on a window that needs a
  parameter raises a clear error instead of an `AttributeError`
- `MultiBandSignal.collapse` accumulated the sum into the first band's own
  time data, corrupting the band on every call
- `_rms` computed the standard deviation instead of the RMS. `rms`,
  `crest_factor`, `snr` and RMS normalization change accordingly; a DC
  component now contributes to the result, so detrend beforehand if only the
  AC power is of interest
- `crest_factor` converted to dB twice for a `MultiBandSignal`
- `Filter` accepted a coefficients dictionary carrying all three coefficient
  types and silently discarded some of them
- the `Signal.spectrum_smoothing` setter did not invalidate the spectrum
  cache, making the setting and `ImpulseResponse.plot_bode(smoothing=...)`
  no-ops while caching was active
- `MagnitudeNormalization.OneKhzFirstChannel` normalized each channel by its
  own value at 1 kHz instead of the first channel's
- `Spectrum.apply_octave_smoothing` crashed for non-uniform frequency vectors
  and did not update the frequency vector in that case
- `spectral_deconvolve` reused the first channel's regularization band for
  every subsequent channel
- `average_irs(normalize_energy=True)` applied the energy ratio `E_i / E_0`
  instead of the amplitude factor `sqrt(E_0 / E_i)`, and had no effect at all
  when `time_average=False`
- `find_modes` changed the spectrum method of the signal passed to it
- `IIRFilter` normalized the caller's coefficient arrays in place and raised
  for integer coefficients
- `MultiBandSignal.sampling_rate_hz` never validated the number of sampling
  rates against the number of bands, and assigned before validating
- separator rules in `metadata_str` were missing or did not match the header
- `latency` checked the type of its second argument only after using it and
  swallowed every exception while printing it
- `Signal.plot_spectrogram` compared an array against a scalar in its lower
  frequency bound guard
- `Signal.time_vector_s` and the chirps in ``generators`` were spaced by
  `length / (N - 1)` instead of the sampling period, accumulating about one
  sample of drift over the signal
- `Signal.fade` used the last sample's time instead of the signal length as
  the reference for the default fade length
- `CalibrationData(high_snr=False)` raised an `AttributeError` because it
  still passed the pre-enum spectrum parameters
- `mix_sources_on_array` emptied the list of sources passed to it
- `Signal.get_spectrum()` dropped the channel axis of a single-channel signal
  when using Welch's method, so its shape depended on the spectrum method.
  This also made `distances.log_spectral` and `distances.itakura_saito` fail
  with Welch's method
- `generators.chirp(ChirpType.SyncLog)` trimmed or padded the sweep to the
  requested length, which destroyed the synchronization whenever the
  effective length came out longer. `sync_log_chirp` keeps its natural length
- `Compressor` divided the caller's time data in place when no pre-gain was
  set
- `FIRFilter.reset_state()` left the write index of its circular buffer
  where it was
- the energy normalizations of `Spectrum.plot_magnitude` divided the
  integrated energy by the number of frequency bins instead of using the mean
  square, so the offset depended on the frequency resolution and differed
  from `Signal.plot_magnitude` by `10*log10(df)`
- the time vector of the spectrogram was spread linearly over the signal
  length instead of following the hop size, and ignored the offset introduced
  by padding. Each frame is now placed at its window centre relative to the
  start of the signal, so the first frames are at negative times when padding
  is active
- the time vectors used for the energy decay curve, the centre time and
  Lundeby's noise compensation in ``room_acoustics`` were spaced by
  `length / (N - 1)` instead of the sampling period

Misc
~~~~
- The documentation builds without warnings: the class-member summaries no
  longer ask for stub files, the orphaned `general_tools` page is gone, the
  new ``realtime`` module has a page, and the nested lists and references in
  the docstrings are valid reStructuredText
- The README's link to a non-existent `examples/` directory was replaced by a
  short usage example, and `docs/readme.rst` now includes the README instead
  of duplicating an outdated copy of it
- Group delay is plotted in ms everywhere; `plot_bode` used seconds
- The frequency ranges of the plotting methods are annotated as tuples, which
  is what their defaults are
- `Filter.plot_magnitude`, `plot_phase` and `plot_group_delay` share one
  length adaptation, which reports the length that was actually asked for and
  extends by 100 samples in all three (`plot_phase` extended by 1)
- `Signal.trim_with_level_threshold` and `find_frequencies_above_threshold`
  raise a clear `ValueError` when nothing crosses the threshold instead of a
  bare `IndexError`
- `Spectrum.spectral_difference(complex=False)` returns a magnitude spectrum
  even for complex inputs, and `transforms.cepstrum(complex=False)` returns a
  real array, as both annotations promised
- `Signal.plot_spl` clips the real and imaginary parts at the same 500 dB
- The beamformers' grid loops are vectorized with `einsum`
- `plots.plots` no longer shadows the `max` and `min` builtins module-wide
- In-place writes through property getters were replaced by assignments
  through the setters, so validation, complex-value handling and cache
  invalidation are reached
- `convolve_rir_on_signal` now selects the overlap-add convolution for length
  ratios outside `[1/15, 15]`; the previous condition was subsumed and made
  the direct convolution branch unreachable for similar lengths
- The magnitude normalizations of `Signal.plot_magnitude`,
  `ImpulseResponse.plot_bode` and `Spectrum.plot_magnitude` now come from one
  shared implementation. All three now interpolate the 1 kHz value in the
  power domain, so that the normalized curve carries unit power at 1 kHz
- The default seaborn style is applied in one place instead of four, and
  `beamforming._beamforming` no longer imports seaborn unguarded
- `ImpulseResponse.copy_with_new_time_data`, the `remove_ir_latency`
  dispatch and the length/impulse prologue of `FilterBank`'s plotting
  methods no longer duplicate their `Signal`, helper and `get_ir`
  counterparts

`0.9 <https://pypi.org/project/dsptoolbox/0.9>`_ -
---------------------
Added
~~~~~
- More methods to `arma` in ``filterbanks``
- Faster computation of group delay for impulse responses and FIR filters

Bugfix
~~~~~~
- some cases of `vqt` were crashing due to an index error
- corrected the frequency vector produced when obtaining the spectogram of a signal
- robustness checks while generating some windows in backend functions

Misc
~~~~
- Improved some type annotations
- Corrected and extended docstrings

`0.8 <https://pypi.org/project/dsptoolbox/0.8>`_ -
---------------------
Added
~~~~~
- New parameter to `spectrum_via_filterbank` in ``transforms``
- `lufs_integrated` in standard module
- Possibility of using true peak value in `crest_factor`
- `fractional_octave_bands` now returns frequencies as well

Bugfix
~~~~~~
- correction of synthesis filters in `qmf` filterbank
- robustness checks while generating some windows in backend functions

Misc
~~~~
- Updated dependencies and numba support for Python 3.14
- Extended test coverage for some functions
- Added length check for convolutions with similar length signals in `convolve_rir`
- Unified plotting parameters across all whole package and remove unnecessary parameters
- Improved some type annotations with overloads
- Corrected and extended docstrings

`0.7.4 <https://pypi.org/project/dsptoolbox/0.7.4>`_ -
---------------------
Added
~~~~~
- `window_ir_tukey` in ``transfer_functions``
- some channel handling methods to `Spectrum`
- `spectrum_via_filterbank` in ``transforms``

Misc
~~~~
- corrections and additions to documentation

`0.7.3 <https://pypi.org/project/dsptoolbox/0.7.3>`_ -
---------------------
Bugfix
~~~~~
- correct minus sign in `warp_frequency` to match the rest of the toolbox

Misc
~~~~
- corrections and additions to documentation

`0.7.2 <https://pypi.org/project/dsptoolbox/0.7.2>`_ -
---------------------
Added
~~~~~
- `warp_frequency` in ``tools``
- Spectrum class now has `normalize` and `add_gain` methods
- fractional delay via thiran's allpass filter in ``filterbanks``

Bugfix
~~~~~
- `arma` now has the correct order for burg's method
- avoid delivering unnecessary files in package distribution. 0.7.1 was yanked
  because of this

Misc
~~~~
- update dependencies and add support for python 3.14 (without numba)
- corrections to documentation
- default to ASIO usage with sounddevice

`0.7 <https://pypi.org/project/dsptoolbox/0.7>`_ -
---------------------
Added
~~~~~
- `trim_with_time_selection` in ``standard``
- `FIRUniformPartitionedMultichannel` in ``filterbanks``

Misc
~~~~
- made installation with numba optional
- renamed arguments of `spectral_deconvolve` for more clarity
- simplified and corrected the cepstrum computations
- removal of deprecated functions

`0.6.2 <https://pypi.org/project/dsptoolbox/0.6.2>`_ -
---------------------
Added
~~~~~
- new implementation for FIR filters in realtime: `FIROverlapSave` and
  `FIRUniformPartitioned`
- `crest_factor` in ``standard``
- `warp` in ``Spectrum`` class for warping using analytical function
- Warped filters in ``filterbanks`` both realtime and offline processing:
  `WarpedIIR` and `WarpedFIR`
- alternative constructors for realtime filters where it is sensible

Misc
~~~~
- made minimum phase computation more efficient
- added some type annotations

`0.6.1 <https://pypi.org/project/dsptoolbox/0.6.1>`_ -
---------------------
Added
~~~~~
- `alpha` parameter for minimum phase ir
- One-padding in spectrum
- `to_signal` in spectrum
- analytic computation can be now used for the excess group delay

Bugfix
~~~~~~
- fixed smoothing with logarithmic frequency vector
- room acoustics descriptors now do not modify the input time signal
- order for numba functions has been fixed so that it does not fail
  unexpectedly

Misc
~~~~
- `convolve_rir` now allows for any length of inputs
- use circle in `zp_plot` for filters instead of manually generating it
- clip dynamic range for minimum phase magnitude

`0.6 <https://pypi.org/project/dsptoolbox/0.6>`_ -
---------------------
Misc
~~~~
- renamed some arguments in `general_plot`, `plot_spectrogram` and `plot_spl`
- `lin_phase_from_mag` and `min_phase_from_mag` now use the spectrum class as
  an input
- most enums are checked exclusively. If a value is not regarded, a ValueError
  is raised
- refactored `window_frequency_dependent`. The computation is now parallelized
  and delivers a complex spectrum with a linear frequency vector

`0.5.3 <https://pypi.org/project/dsptoolbox/0.5.3>`_ -
---------------------
Added
~~~~~
- `combine_ir_with_dirac` now can take a gain value for the impulse when
  merging with the impulse
- `complex_smoothing` to ``transfer_functions``
- `MagnitudeNormalization` was extended to normalization of all channels using
  only the first channel as a reference. This maintains the gain relations
- `AllpassFirstOrder` was added to biquads
- thd_percent was added to `harmonic_distortion_analysis`

Bugfix
~~~~~~
- return for `two_axes` plot in ``plots`` now has also the second axis
- plots for the `LRFilterBank` were fixed after some parameters were no longer
  defaults

Misc
~~~~
- caching is now activated for certain parallelized computations
- most enums are checked exclusively. If a value is not regarded, a ValueError
  is raised

`0.5.2 <https://pypi.org/project/dsptoolbox/0.5.2>`_ -
---------------------
Added
~~~~~
- added new plot in ``plots`` with two y-axis on the same plot
- ``plot_bode`` to `ImpulseResponse`
- utility function `trim_with_level_threshold`
- `FirDesigner` in ``filter_banks``
- introduced `ensure_integer_delay` parameter for FirDesigner

Bugfix
~~~~~~
- `GroupDelayDesigner` was fixed when using interpolation to increase the
  frequency resolution. The threshold for triggering the interpolation step is
  now less strict
- fixed a bug where `harmonic_distortion_analysis` did not deliver the right
  type for the spectrum of the fundamental

Misc
~~~~
- made group delay computation faster
- fixes and additions to documentation
- requirements for devs were improved
- more type annotations in ``plots``

`0.5.1 <https://pypi.org/project/dsptoolbox/0.5.1>`_ -
---------------------
Added
~~~~~
- `get_next_power_2` in ``tools``
- `copy_with_new_time_data` for `Signal` and `ImpulseResponse`
- property `is_complex_signal` in `Signal`

Bugfix
~~~~~~
- fixed assertion where transfer function of `Filter` could not compute nyquist
- amplitude_scale_factor in `Signal` can only be read as property but not set
  by user. It is acquired when setting the time data
- metadata and metadata_str properties are computed on-demand for every class
- removed unused attribute `scale_factor` in `Signal`
- modify some f-Strings so that they are also compatible with python 3.11
- passing a single frequency to `linkwitz_riley_crossover` is now supported

Misc
~~~~
- memory footprint was reduced to minimum due to avoidance of copying multiple
  times at different stages
- added python 3.13 support
- file structure of helpers was rearranged for more clarity
- requirements for different python versions were specified
- fixed type annotations

`0.5 <https://pypi.org/project/dsptoolbox/0.5>`_ - 
---------------------
Added
~~~~~
- Various properties were added to `Signal` and `Filter` class. They are mostly
  computed dynamically
- `Spectrum` class that can handle magnitude and complex, multi-channel spectra
- Multiple enums in each module were added in order to replace all string
  parameters across the code base
- `Filter` has now `plot_taps()` for FIR filters
- `merge_filters` can now merge FIR or IIR filters with each other by
  convolving and appending SOS, respectively
- `Filter` can now start FIR filters from files
- `apply_gain` can now be applied to filters and filter banks
- `delay` function was applied to apply delay integer delays to signals. This
  is considerably more efficient than using `fractional_delay`
- `fractional_octave_smoothing` can now be applied to logarithmically spaced
  data
- `spectral_difference` computes the magnitude or complex difference between
  spectra
- `trim_ir` can now be applied to multichannel signals directly

Misc
~~~~
- `Filter` was thoroughly refactored. Its constructor is now simpler.
- `Signal` does not do implicit copies of the time data anymore and was largely
  refactored
- documentation and type annotations fixes
- renamed `merge_filterbanks` and `merge_signals` to `append_filterbanks` and
  `append_signals`. Now they can also get a list with more than 2 objects
- replaced binary string parameters with booleans across code base
- most classes have builder-pattern-like behavior that return the object when
  it has been modified


`0.4.8 <https://pypi.org/project/dsptoolbox/0.4.8>`_ - 
---------------------
Added
~~~~~
- `convert_sample_representation` in ``dsptoolbox.tools``
- `sum_all_channels` method in Signal class
- `get_group_delay` method in Filter class
- iterator in Signal class now iterates over the channels
- `StateSpaceFilter` in ``filterbanks``
- synchronized swept-sine was added a new type of chirp in ``generators``
- `clear_time_window` in Signal class
- `modify_signal_length` in ``dsptoolbox.*``

Misc
~~~~
- extended functionality of `find_ir_latency` in ``transfer_functions``
- dropped support for Python 3.10
- rescaling time data can be done directly in `resample`
- `PhaseLinearizer` and `GroupDelayDesigner` now can use two different
  integration methods. They also got a new parameter that allows for more
  flexible designs
- `trim_ir` can now trim the end of an IR without modifying the start

Bugfix
~~~~~~
- plotting in `LRFilterBank` now returns the plots just like the FilterBank
  class
- multiple docs fixes and type annotations
- `merge_filterbanks` was fixed so that the output is a (deep) copy of the
  input instead of a shallow one

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
