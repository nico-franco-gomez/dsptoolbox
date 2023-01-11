"""
Here are methods considered as somewhat special or less common.
"""
import numpy as np
from dsptoolbox.classes.signal_class import Signal
from dsptoolbox._standard import _minimum_phase, _group_delay_direct


def cepstrum(signal: Signal, mode='power'):
    """Returns the cepstrum of a given signal in the Quefrency domain.

    Parameters
    ----------
    signal : Signal
        Signal to compute the cepstrum from.
    mode : str, optional
        Type of cepstrum. Supported modes are `'power'`, `'real'` and
        `'complex'`. Default: `'power'`.

    Returns
    -------
    ceps : `np.ndarray`
        Cepstrum.

    References
    ----------
    https://de.wikipedia.org/wiki/Cepstrum

    """
    mode = mode.lower()
    assert mode in ('power', 'complex', 'real'), \
        f'{mode} is not a supported mode'

    ceps = np.zeros_like(signal.time_data)
    signal.set_spectrum_parameters(method='standard')
    _, sp = signal.get_spectrum()

    for n in range(signal.number_of_channels):
        if mode in ('power', 'real'):
            cp = np.abs(np.fft.irfft((2*np.log(np.abs(sp[:, n])))))**2
        else:
            phase = np.unwrap(np.angle(sp[:, n]))
            cp = np.fft.irfft(np.log(np.abs(sp[:, n])) + 1j*phase).real
        if mode == 'real':
            cp = (cp**0.5)/2
        ceps[:, n] = cp
    return ceps


def min_phase_from_mag(spectrum: np.ndarray, sampling_rate_hz: int,
                       signal_type: str = 'ir'):
    """Returns a minimal phase signal from a magnitude spectrum using
    the hilbert transform.

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_min_phase : `Signal`
        Signal with same magnitude spectrum but minimal phase.

    References
    ----------
    - https://en.wikipedia.org/wiki/Minimum_phase

    """
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, \
        'Spectrum should have shape (bins, channels)'
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)
    min_spectrum = np.empty(spectrum.shape, dtype='cfloat')
    for n in range(spectrum.shape[1]):
        phase = _minimum_phase(spectrum[:, n], False)
        min_spectrum[:, n] = spectrum[:, n]*np.exp(1j*phase)
    time_data = np.fft.irfft(min_spectrum, axis=0)
    sig_min_phase = Signal(
        None, time_data=time_data,
        sampling_rate_hz=sampling_rate_hz, signal_type=signal_type)
    return sig_min_phase


def lin_phase_from_mag(spectrum: np.ndarray, sampling_rate_hz: int,
                       group_delay_ms='minimal',
                       check_causality: bool = True,
                       signal_type: str = 'ir'):
    """Returns a linear phase signal from a magnitude spectrum. It is possible
    to return the smallest causal group delay by checking the minimal phase
    version of the signal and choosing a constant group delay that is never
    lower than minimum group delay (for each channel). A value for the group
    delay can be also passed directly and applied to all channels. If check
    causility is activated, it is assessed that the given group delay is not
    less than each minimal group delay. If deactivated, the generated phase
    could lead to a non-causal system!

    Parameters
    ----------
    spectrum : `np.ndarray`
        Spectrum with only positive frequencies and 0.
    sampling_rate_hz : int
        Signal's sampling rate in Hz.
    group_delay_ms : str or float, optional
        Constant group delay that the phase should have for all channels
        (in ms). Pass `'minimal'` to create a signal with the minimum linear
        phase possible (that is different for each channel).
        Default: `'minimal'`.
    check_causality : bool, optional
        When `True`, it is assessed for each channel that the given group
        delay is not lower than the minimal group delay. Default: `True`.
    signal_type : str, optional
        Type of signal to be returned. Default: `'ir'`.

    Returns
    -------
    sig_lin_phase : `Signal`
        Signal with same magnitude spectrum but linear phase.

    """
    # Check spectrum
    if spectrum.ndim < 2:
        spectrum = spectrum[..., None]
    assert spectrum.ndim < 3, \
        'Spectrum should have shape (bins, channels)'
    if spectrum.shape[0] < spectrum.shape[1]:
        spectrum = spectrum.T
    spectrum = np.abs(spectrum)

    # Check group delay ms parameter
    minimum_group_delay = False
    if type(group_delay_ms) == str:
        group_delay_ms = group_delay_ms.lower()
        assert group_delay_ms == 'minimal', \
            'Group delay should be set to minimal'
        minimum_group_delay = True
    elif type(group_delay_ms) in (float, int):
        group_delay_ms /= 1000
    else:
        raise TypeError('group_delay_ms must be either str, float or int')

    # Frequency vector
    f_vec = np.fft.rfftfreq(spectrum.shape[0]*2-1, 1/sampling_rate_hz)
    delta_f = f_vec[1]-f_vec[0]

    # New spectrum
    lin_spectrum = np.empty(spectrum.shape, dtype='cfloat')
    for n in range(spectrum.shape[1]):
        if check_causality or minimum_group_delay:
            min_phase = _minimum_phase(spectrum[:, n], False)
            min_gd = _group_delay_direct(min_phase, delta_f)
            gd = np.max(min_gd) + 1e-3  # add 1 ms as safety factor
            if check_causality and type(group_delay_ms) != str:
                assert gd <= group_delay_ms, \
                    f'Given group delay {group_delay_ms*1000} ms is lower ' +\
                    f'than minimal group delay {gd*1000} ms for channel {n}'
                gd = group_delay_ms
        else:
            gd = group_delay_ms
        lin_spectrum[:, n] = spectrum[:, n]*np.exp(
            -1j * 2 * np.pi * f_vec * gd)
    time_data = np.fft.irfft(lin_spectrum, axis=0)
    sig_lin_phase = Signal(
        None, time_data=time_data,
        sampling_rate_hz=sampling_rate_hz, signal_type=signal_type)
    return sig_lin_phase
