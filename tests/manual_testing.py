'''
Testing script:
So far, only 'manual' tests have been written here. The purpose of this script
is to check results from some functions.
'''
import dsptoolbox as dsp
import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt


def transfer_function_test():
    # recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized', padding=False,
        keep_original_length=True)
    tf_wind = dsp.transfer_functions.window_ir(tf, at_start=False)
    tf_wind.plot_time()
    tf.plot_time()
    tf.plot_magnitude()
    dsp.plots.show()


def distances_function_test():
    recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    print(dsp.distances.itakura_saito(raw, recorded))
    print(dsp.distances.log_spectral(recorded, raw))


def welch_method():
    from scipy.signal import welch, csd

    cross = False

    if cross:
        raw = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    else:
        raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    raw.set_spectrum_parameters(method='welch', scaling='power spectrum',
                                average='median')
    if cross:
        f2, pp = csd(
            raw.time_data[:, 0], raw.time_data[:, 1],
            fs=raw.sampling_rate_hz, nperseg=1024)
        f1, mine = raw.get_csm()
        mine = mine[:, 0, 1]
    else:
        f2, pp = welch(raw.time_data.squeeze(),
                       raw.sampling_rate_hz, nperseg=1024,
                       average='median', scaling='spectrum')
        f1, mine = raw.get_spectrum()
        mine = np.squeeze(mine)
    print('Same as scipy: ', np.all(np.isclose(np.abs(pp) - np.abs(mine), 0)))
    # print('Same as scipy: ', np.all(np.isclose(pp[100:], mine[100:])))
    plt.figure()
    # plt.semilogx(f2, np.angle(pp), label='Scipy')
    # plt.semilogx(f1, np.angle(mine), label='mine')
    plt.semilogx(f2, 10*np.log10(np.abs(pp)), label='Scipy')
    plt.semilogx(f1, 10*np.log10(np.abs(mine)), label='mine')
    plt.legend()
    dsp.plots.show()


def csm():
    raw = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw.plot_csm()
    dsp.plots.show()


def group_delay():
    # recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized')
    tf = dsp.transfer_functions.window_ir(tf)
    f, g1 = dsp.group_delay(tf, 'matlab')
    f, g2 = dsp.group_delay(tf, 'direct')

    plt.plot(f, g1, label='matlab')
    plt.plot(f, g2, label='direct')
    plt.legend()
    dsp.plots.show()


def stft():
    try:
        import librosa
    except ModuleNotFoundError as e:
        print(e)
        print('librosa not installed, because numba is not yet supported for' +
              ' python 3.11. Try reinstalling librosa or run in python ' +
              '3.10 environment!')
        exit()
    except Exception as e:
        print(e)
        exit()

    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    raw.set_spectrogram_parameters(
        window_length_samples=2048, scaling=False, overlap_percent=75,
        padding=False)
    t, f, stft = raw.get_spectrogram()
    D = librosa.stft(raw.time_data.squeeze(), center=False)
    print(np.all(np.isclose(stft[1:, :-4], D[1:])))
    # exit()
    plt.subplot(121)
    st_abs = 20*np.log10(np.abs(stft))
    plt.imshow(st_abs, origin='lower', aspect='auto',
               vmin=np.max(st_abs)-100, vmax=np.max(st_abs)+10)
    plt.colorbar()
    plt.subplot(122)
    D_abs = 20*np.log10(np.abs(D))
    plt.imshow(D_abs, origin='lower', aspect='auto',
               vmin=np.max(D_abs)-100, vmax=np.max(D_abs)+10)
    plt.colorbar()

    raw.plot_spectrogram()
    dsp.plots.show()


def minimum_phase_systems():
    # recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized')
    tf = dsp.transfer_functions.window_ir(tf)
    f, bla = dsp.minimum_phase(tf)
    plt.subplot(121)
    plt.semilogx(f, bla)
    plt.subplot(122)
    f, bla = dsp.minimum_group_delay(tf)
    f, bla2 = dsp.group_delay(tf)
    plt.semilogx(f, (bla2-bla)*1e3)
    dsp.plots.show()


def room_acoustics():
    # recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized')
    tf = dsp.transfer_functions.window_ir(tf)
    print(dsp.room_acoustics.reverb_time(tf, 'T20'))
    print(dsp.room_acoustics.reverb_time(tf, 'T30'))
    print(dsp.room_acoustics.reverb_time(tf, 'T60'))
    print(dsp.room_acoustics.reverb_time(tf, 'EDT'))
    print(dsp.room_acoustics.find_modes(tf, [30, 350]))


def new_transfer_functions():
    # recorded = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.compute_transfer_function(
        recorded_multi, raw, mode='h2')
    # tf.plot_magnitude(normalize=None)
    tf.plot_coherence()
    from scipy.signal import coherence
    import matplotlib.pyplot as plt
    # Trying
    x = raw.time_data[:, 0]
    y = recorded_multi.time_data[:, 1]
    freq, coh = coherence(x, y, raw.sampling_rate_hz, nperseg=1024)
    plt.plot(freq, coh)
    dsp.plots.show()


def multiband():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    r1 = dsp.Signal(None, recorded_multi.time_data[:, 0],
                    recorded_multi.sampling_rate_hz)
    r2 = dsp.Signal(None, recorded_multi.time_data[:, 1],
                    recorded_multi.sampling_rate_hz)
    multi = dsp.MultiBandSignal()
    # multi = dsp.MultiBandSignal([r1, r2])
    multi.show_info(True)
    multi.show_info(False)
    multi.add_band(r1)
    multi.add_band(r2)
    multi.show_info(False)
    multi.remove_band(-1)
    multi.show_info(False)


def save_objects():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    recorded_multi.save_signal(mode='wav')
    recorded_multi.save_signal(mode='flac')
    recorded_multi.save_signal(mode='pickle')
    print()


def generators():
    # Noise
    # wno = dsp.generators.noise(
    #     'white', length_seconds=3, peak_level_dbfs=-1, number_of_channels=4,
    #     sampling_rate_hz=44100)
    # wno = dsp.generators.noise('pink', sampling_rate_hz=44100)
    # wno = dsp.generators.noise(
    #     'red', peak_level_dbfs=-5, sampling_rate_hz=44100)
    # wno = dsp.generators.noise('blue', sampling_rate_hz=44100)
    # wno = dsp.generators.noise(
    #     'violet', number_of_channels=3, sampling_rate_hz=44100)
    # wno = dsp.generators.noise(
    #     'grey', length_seconds=2, number_of_channels=3,
    #     sampling_rate_hz=44100)
    # wno.plot_magnitude(normalize=None)

    # Chirps
    wno = dsp.generators.chirp(type_of_chirp='log', length_seconds=5,
                               fade='log',
                               padding_end_seconds=2, number_of_channels=1,
                               peak_level_dbfs=-20, range_hz=[20, 24e3],
                               sampling_rate_hz=44100)

    # Plots
    wno.plot_magnitude(range_hz=[20, 24e3])
    wno.plot_spectrogram()
    wno.plot_time()
    # wno.plot_phase()
    # wno.plot_group_delay()
    dsp.plots.show()


def recording():
    from time import sleep

    sleep(3)

    dsp.audio_io.set_device()
    chirp = dsp.generators.chirp(padding_end_seconds=2, sampling_rate_hz=44100)
    s2 = dsp.audio_io.play_and_record(chirp)
    tf = dsp.transfer_functions.spectral_deconvolve(s2, chirp)
    tf = dsp.transfer_functions.window_ir(tf)
    tf.plot_magnitude()
    dsp.plots.show()


def convolve_rir_signal():
    rir = dsp.Signal(join('examples', 'data', 'rir.wav'),
                     signal_type='rir')
    speech = dsp.Signal(join('examples', 'data', 'speech.flac'))
    dsp.audio_io.set_device(2)
    new_speech = \
        dsp.room_acoustics.convolve_rir_on_signal(
            speech, rir, keep_length=False)
    dsp.audio_io.play(new_speech)
    dsp.plots.show()


def cepstrum():
    speech = dsp.Signal(join('examples', 'data', 'speech.flac'))
    c = dsp.special.cepstrum(speech, mode='real')
    # import matplotlib.pyplot as plt
    # plt.plot(c)
    dsp.plots.general_plot(speech.time_vector_s, c, log=False)
    dsp.plots.show()


def merging_signals():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))

    fb = \
        dsp.filterbanks.linkwitz_riley_crossovers(
            [1000, 2000], [4, 6], raw.sampling_rate_hz)

    raw_b = fb.filter_signal(raw)
    recorded_multi_b = fb.filter_signal(recorded_multi)

    new_b = dsp.merge_signals(raw_b, recorded_multi_b)
    new_b.show_info()

    # Lengths and such
    # raw = dsp.pad_trim(raw, len(raw.time_data)-100)
    # print(recorded_multi.time_data.shape, raw.time_data.shape)

    # n = dsp.merge_signals(recorded_multi, raw)
    # print(n.time_data.shape)
    # n.plot_time()
    # dsp.plots.show()


def merging_fbs():
    # fb1 = \
    #     dsp.filterbanks.linkwitz_riley_crossovers([1000, 2000], [4, 6])
    fb1 = \
        dsp.filterbanks.reconstructing_fractional_octave_bands(2)
    fb2 = \
        dsp.filterbanks.reconstructing_fractional_octave_bands(1)

    fb_n = dsp.merge_filterbanks(fb1, fb2)
    fb_n.show_info()
    fb_n.plot_magnitude()
    dsp.plots.show()


def collapse():
    d = dsp.generators.dirac(length_samples=2**13, sampling_rate_hz=48_000)
    # d.plot_magnitude(normalize=None)
    # d.plot_time()
    fb = dsp.filterbanks.reconstructing_fractional_octave_bands()
    d_b = fb.filter_signal(d)
    # d_b.bands[2].set_spectrum_parameters(method='standard')
    # d_b.bands[2].plot_magnitude(normalize=None)
    # d_b.bands[0].plot_phase()
    d_rec = d_b.collapse()
    d_rec.set_spectrum_parameters(method='standard')
    d_rec.plot_magnitude(normalize=None, range_db=[-2, 2])
    # d_rec.plot_phase()
    # d_rec.plot_group_delay()
    dsp.plots.show()


def smoothing():
    psig2 = dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    psig2.set_spectrum_parameters(method='standard')
    psig2.plot_magnitude(smoothe=4)
    dsp.plots.show()


def min_phase_signal():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    # recorded_multi = \
    #     dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    ir = dsp.transfer_functions.spectral_deconvolve(recorded_multi, raw)
    ir = dsp.transfer_functions.window_ir(ir)
    # ir.plot_phase()
    _, sp = ir.get_spectrum()
    min_ir = dsp.transfer_functions.min_phase_from_mag(sp, ir.sampling_rate_hz)
    # min_ir.plot_phase()
    # min_ir.plot_time()
    ir.plot_group_delay()
    min_ir.plot_group_delay()
    dsp.plots.show()


def lin_phase_signal():
    # recorded_multi = \
    #     dsp.Signal(join('examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    ir = dsp.transfer_functions.spectral_deconvolve(recorded_multi, raw)
    ir = dsp.transfer_functions.window_ir(ir)
    _, sp = ir.get_spectrum()
    lin_ir = dsp.transfer_functions.lin_phase_from_mag(
        sp, ir.sampling_rate_hz, group_delay_ms='minimal',
        check_causality=True)
    # Phases
    # ir.plot_phase(unwrap=True)
    # lin_ir.plot_phase(unwrap=True)

    # Time signals
    # ir.plot_time()
    # lin_ir.plot_time()

    # Group delays
    ir.plot_group_delay()
    lin_ir.plot_group_delay()
    dsp.plots.show()


def gammatone_filters():
    fb = dsp.filterbanks.auditory_filters_gammatone()

    # Filtering and listening to result
    # speech = dsp.Signal(join('examples', 'data', 'speech.flac'))
    # speech = fb.filter_signal(speech, mode='parallel')
    # speech_band = speech.bands[7]
    # dsp.audio_io.play(speech_band)

    # Plotting filter bank
    fb.plot_magnitude(mode='summed')
    fb.plot_magnitude(mode='parallel', length_samples=2**13)
    dsp.plots.show()


def ir2filt():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))
    ir = dsp.transfer_functions.spectral_deconvolve(recorded_multi, raw)
    ir = dsp.transfer_functions.window_ir(ir)

    f = dsp.ir_to_filter(ir, channel=0, phase_mode='direct')
    f.plot_magnitude()
    dsp.plots.show()


def fwsnrseg():
    recorded_multi = \
        dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))
    # raw = dsp.Signal(join('examples', 'data', 'chirp.wav'))

    c0 = recorded_multi.get_channels(0)
    c1 = recorded_multi.get_channels(1)
    fw = dsp.distances.fw_snr_seg(c0, c1)
    print(fw)


def true_peak():
    s = dsp.Signal(join('examples', 'data', 'chirp_stereo.wav'))

    # Testing for multibandsignal
    # from dsptoolbox.filterbanks import auditory_filters_gammatone
    # fb = auditory_filters_gammatone([40, 1e3])
    # s = fb.filter_signal(s)
    t, p = dsp.true_peak_level(s)
    print(t, p)
    print('difference: ', t - p, ' dB')


def harmonic_tone():
    c = dsp.generators.harmonic(
        frequency_hz=500, length_seconds=2, number_of_channels=3,
        uncorrelated=True, fade='log', sampling_rate_hz=4_000)
    # c.plot_time()
    c.plot_magnitude()
    dsp.plots.show()


def band_swapping():
    s = dsp.generators.noise(
        'grey', number_of_channels=3, sampling_rate_hz=16_000,
        length_seconds=3)

    # fb = dsp.filterbanks.reconstructing_fractional_octave_bands(
    #     frequency_range=[250, 2000], sampling_rate_hz=s.sampling_rate_hz)
    fb = dsp.filterbanks.auditory_filters_gammatone(
        [50, 500], sampling_rate_hz=s.sampling_rate_hz)

    s_f = fb.filter_signal(s, mode='parallel')
    print(s_f.number_of_bands)
    bands = []
    bands.append(s_f.bands[0])
    bands.append(s_f.bands[1])

    s_multi = dsp.MultiBandSignal(bands)
    s_multi.show_info(True)
    s_multi.add_band(s_f.bands[2])
    s_multi.show_info(True)
    s_multi.remove_band(0)
    s_multi.show_info(True)

    s_multi.swap_bands([1, 0])
    s_multi.show_info(True)


def fractional_time_delay():
    # Signal
    s = dsp.generators.noise(
        'grey', length_seconds=2, number_of_channels=3, sampling_rate_hz=44100)
    s.plot_time()
    s_d = dsp.fractional_delay(s, 10.5e-3, channels=[-1])
    s_d.plot_time()

    # MultiBandSignal
    # fb = dsp.filterbanks.auditory_filters_gammatone([500, 1000])
    # sb = fb.filter_signal(s)
    # s_d = dsp.fractional_delay(sb, 10.5e-3, channels=1)
    # s_d.bands[0].plot_time()
    dsp.plots.show()


def synthetic_rir():
    rir = dsp.room_acoustics.generate_synthetic_rir(
        room_dimensions_meters=[4, 5, 6],
        source_position=[2, 2.5, 3], receiver_position=[2, 1, 5.5],
        total_length_seconds=0.5,
        sampling_rate_hz=48000, desired_reverb_time_seconds=None,
        apply_bandpass=True)
    rir.plot_time()
    rir = dsp.room_acoustics.generate_synthetic_rir(
        room_dimensions_meters=[4, 5, 6],
        source_position=[2, 2.5, 3], receiver_position=[2, 1, 5.5],
        total_length_seconds=0.5,
        sampling_rate_hz=48000, desired_reverb_time_seconds=None,
        apply_bandpass=False)
    rir.plot_time()
    # rir.plot_magnitude()
    # rir.plot_phase(unwrap=True)
    # rir.plot_group_delay()
    dsp.plots.show()


def beamforming_basics():
    # Mic Array
    path = join('examples', 'data', 'array.xml')
    ar = pd.read_xml(path)
    ma = dsp.beamforming.MicArray(ar)
    # ma.plot_points(projection='3d')
    print('Maximum range', ma.get_maximum_frequency_range())

    # Grid
    xval = np.arange(-0.5, 0.5, 0.1)
    yval = np.arange(-0.5, 0.5, 0.1)
    zval = 1
    g = dsp.beamforming.Regular2DGrid(
        xval, yval, ['x', 'y'], value3=zval)
    # g.plot_points()

    # Steering vector
    st = dsp.beamforming.SteeringVector(formulation='classic')
    k = np.linspace(800, 1000, 10) * 2 * np.pi / 343
    h = st.get_vector(k, g, ma)
    print(h.shape)
    st = dsp.beamforming.SteeringVector(formulation='inverse')
    h = st.get_vector(k, g, ma)
    print(h.shape)
    # print('Grid: ', g.coordinates.shape)
    # print('Mics: ', ma.coordinates.shape)
    # print('k: ', k.shape)

    # Beamformer
    noise = dsp.generators.noise(
        number_of_channels=ma.number_of_points, sampling_rate_hz=20_000)
    bf = dsp.beamforming.BeamformerDASFrequency(noise, ma, g, st)
    bf.plot_setting()
    bf.show_info()
    print(bf.get_frequency_range_from_he())
    dsp.plots.show()


def beamforming_steering_test():
    # ======== Mini example
    # ma = dsp.beamforming.MicArray(dict(x=[0, 0], y=[0, 0.1], z=[0, 0]))
    # g = dsp.beamforming.Grid(dict(x=[0, 0], y=[0, 0.1], z=[0.5, 0.5]))

    # ======== Full example
    # Array
    path = join('examples', 'data', 'array.xml')
    ar = pd.read_xml(path)
    ma = dsp.beamforming.MicArray(ar)
    # Grid
    xval = np.arange(-0.5, 0.5, 0.1)
    yval = np.arange(-0.5, 0.5, 0.1)
    zval = 1
    g = dsp.beamforming.Regular2DGrid(
        xval, yval, ['x', 'y'], value3=zval)

    # Check steering vector Classic
    r0 = ma.array_center_coordinates

    def dist(r1, r0):
        """Euclidean distance between two points"""
        return np.sqrt(np.sum((r1-r0)**2))

    k = np.array([1000, 1200]) * np.pi * 2 / 343
    rt0 = g.get_distances_to_point(r0)
    h = np.zeros((len(k), ma.number_of_points, g.number_of_points),
                 dtype='cfloat')
    N = ma.number_of_points
    for i0, kn in enumerate(k):
        for i1 in range(ma.number_of_points):
            for i2 in range(g.number_of_points):
                rti = dist(g.coordinates[i2, :], ma.coordinates[i1, :])
                rt0 = dist(g.coordinates[i2, :], r0)
                h[i0, i1, i2] = 1/N * np.exp(-1j*kn*(rti-rt0))

    st = dsp.beamforming.SteeringVector(formulation='classic')
    h_intern = st.get_vector(k, g, ma)

    # Test for difference
    print(np.all(np.isclose(h_intern - h, 0)))


def beamforming_virtual_source():
    from time import time
    single_source = True

    # Array
    path = join('examples', 'data', 'array.xml')
    ar = pd.read_xml(path)
    ma = dsp.beamforming.MicArray(ar)

    if single_source:
        ns = dsp.beamforming.MonopoleSource(
            dsp.generators.noise(length_seconds=3, sampling_rate_hz=20_000),
            [0, 0, 0.5])
        # Plot setting
        fig, ax = ma.plot_points(projection='3d')
        ax.scatter(ns.coordinates[0], ns.coordinates[1], ns.coordinates[2])
        s_ma = ns.get_signals_on_array(ma)
    else:
        sp = dsp.Signal(join('examples', 'data', 'speech.flac'))
        ns = dsp.generators.noise(
            length_seconds=3, sampling_rate_hz=sp.sampling_rate_hz)
        sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.4])
        ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
        # Plot setting
        fig, ax = ma.plot_points(projection='3d')
        ax.scatter(ns.coordinates[0], ns.coordinates[1], ns.coordinates[2])
        ax.scatter(sp.coordinates[0], sp.coordinates[1], sp.coordinates[2])
        s_ma = dsp.beamforming.mix_sources_on_array([sp, ns], ma)
    # print('Number of channels:',
    #       s_ma.number_of_channels == ma.number_of_points)

    # Listen to one channel
    # s1 = s_ma.get_channels([63])
    # dsp.audio_io.set_device()
    # dsp.audio_io.play(s1)

    # Assess computation time for cross-spectral matrix
    # s_ma = s_ma.get_channels([0, 1, 2, 3, 4])
    st = time()
    f, csm = s_ma.get_csm()
    # f, csm = s_ma.plot_csm()
    print(time() - st)
    # s_ma.plot_time()
    # dsp.plots.show()


def beamforming_complete_test_2D():
    # Mic Array
    micsx = np.arange(-0.25, 0.25, 0.1)
    dim1, dim2 = np.meshgrid(micsx, micsx)
    dim1 = dim1.flatten()
    dim2 = dim2.flatten()
    positions = np.append(dim1[..., None], dim2[..., None], axis=1)
    positions = np.append(
        positions, np.zeros((len(dim1), 1)), axis=1)
    positions = dict(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
    ma = dsp.beamforming.MicArray(positions)

    # path = join('examples', 'data', 'array.xml')
    # ar = pd.read_xml(path)
    # ma = dsp.beamforming.MicArray(ar)

    # Signal (simulated)
    ns = dsp.beamforming.MonopoleSource(
        dsp.generators.noise(length_seconds=2, sampling_rate_hz=20_000),
        [0, 0.4, 0.5])
    s = ns.get_signals_on_array(ma)

    # Grid
    xval = np.arange(-0.2, 0.2, 0.1)
    yval = np.arange(-0.5, 0.5, 0.1)
    zval = 0.5
    g = dsp.beamforming.Regular2DGrid(
        xval, yval, ['x', 'y'], value3=zval)

    # Steering vector
    st = dsp.beamforming.SteeringVector(formulation='true location')

    # Create beamformer and plot setting
    bf = dsp.beamforming.BeamformerDASFrequency(s, ma, g, st, )
    _, ax = bf.plot_setting()
    ax.scatter(ns.coordinates[0], ns.coordinates[1], ns.coordinates[2])

    # Get and show map
    m = bf.get_beamformer_map(2000, 3, remove_csm_diagonal=True)
    g.plot_map(m)
    dsp.plots.show()


def beamforming_complete_test_time():
    # Small mic array
    micsx = np.arange(-0.1, 0.11, 0.1)
    micsy = np.arange(-0.3, 0.31, 0.1)
    dim1, dim2 = np.meshgrid(micsx, micsy)
    dim1 = dim1.flatten()
    dim2 = dim2.flatten()
    positions = np.append(dim1[..., None], dim2[..., None], axis=1)
    positions = np.append(
        positions, np.zeros((len(dim1), 1)), axis=1)
    positions = dict(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
    ma = dsp.beamforming.MicArray(positions)

    # Signal (simulated)
    sp = dsp.Signal(join('examples', 'data', 'speech.flac'))
    ns = dsp.generators.noise(
        length_seconds=3, sampling_rate_hz=sp.sampling_rate_hz)
    sp = dsp.beamforming.MonopoleSource(sp, [0, -0.5, 0.5])
    ns = dsp.beamforming.MonopoleSource(ns, [0, 0, 0.5])
    s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)

    s_trial = s.get_channels(5)
    s_trial.save_signal('/Users/neumanndev/Downloads/direct')

    # Grid
    xval = np.arange(-0.5, 0.5, 0.1)
    g = dsp.beamforming.LineGrid(xval, 'y', 0.5, 0)

    bf = dsp.beamforming.BeamformerDASTime(s, ma, g)
    _, ax = bf.plot_setting()
    ax.scatter(ns.coordinates[0], ns.coordinates[1], ns.coordinates[2])
    ax.scatter(sp.coordinates[0], sp.coordinates[1], sp.coordinates[2])

    out_sig = bf.get_beamformer_output()
    out_sig.save_signal('/Users/neumanndev/Downloads/beamformer_time')
    dsp.plots.show()


def beamforming_3d_grid():
    # Small mic array
    micsx = np.arange(-0.1, 0.11, 0.1)
    micsy = np.arange(-0.3, 0.31, 0.1)
    dim1, dim2 = np.meshgrid(micsx, micsy)
    dim1 = dim1.flatten()
    dim2 = dim2.flatten()
    positions = np.append(dim1[..., None], dim2[..., None], axis=1)
    positions = np.append(
        positions, np.zeros((len(dim1), 1)), axis=1)
    positions = dict(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2])
    ma = dsp.beamforming.MicArray(positions)

    ap = ma.aperture

    linex = np.arange(-0.5*ap, 0.5*ap, ap/4)
    liney = np.arange(-0.3*ap, 0.3*ap, ap/4)
    linez = np.arange(-0.4*ap, 0.4*ap, ap/2)+ap

    # Signal (simulated)
    # sp = dsp.Signal(join('examples', 'data', 'speech.flac'))
    sp = dsp.generators.noise(
        type_of_noise='white', length_seconds=3, sampling_rate_hz=10_000)
    sp = dsp.normalize(sp, -100)
    ns = dsp.generators.noise(
        length_seconds=3, sampling_rate_hz=sp.sampling_rate_hz)
    # sp = dsp.beamforming.MonopoleSource(sp, [0, -0.3, 0.5])
    ns = dsp.beamforming.MonopoleSource(
        ns, [np.max(linex), np.max(liney), np.max(linez)])
    s = ns.get_signals_on_array(ma)
    # s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)
    # s = dsp.beamforming.mix_sources_on_array([sp, ns], ma)

    # Grid and steering vector
    g = dsp.beamforming.Regular3DGrid(linex, liney, linez)
    st = dsp.beamforming.SteeringVector(formulation='true location')

    # Define beamformer and plot setting
    bf = dsp.beamforming.BeamformerDASFrequency(s, ma, g, st, c=343)
    fig, ax = bf.plot_setting()
    ax.scatter(ns.coordinates[0], ns.coordinates[1], ns.coordinates[2])
    # ax.scatter(sp.coordinates[0], sp.coordinates[1], sp.coordinates[2])

    # Run
    map = bf.get_beamformer_map(2000, 3, remove_csm_diagonal=False)
    # print(map)
    # exit()
    # y seems trasposed
    g.plot_map(map, 'z', ns.coordinates[2])
    g.plot_map(map, 'y', ns.coordinates[1])
    # g.plot_map(map, 'x', ns.coordinates[0])

    dsp.plots.show()


def beamforming_frequency_formulations():
    s = dsp.generators.noise(
        type_of_noise='white', sampling_rate_hz=5_000, peak_level_dbfs=-6)
    s2 = dsp.generators.noise(
        type_of_noise='white', sampling_rate_hz=5_000, peak_level_dbfs=-6)

    line = np.arange(0, 1, 0.2)
    xx, yy = np.meshgrid(line, line, indexing='ij')
    ma = dsp.beamforming.MicArray(
        dict(x=xx.flatten(), y=yy.flatten(), z=np.zeros(len(xx.flatten()))))
    ap = ma.aperture

    ms = dsp.beamforming.MonopoleSource(s, coordinates=[0.4, 0.4, ap])
    ms2 = dsp.beamforming.MonopoleSource(s2, coordinates=[0, 0, ap])
    s_out = dsp.beamforming.mix_sources_on_array([ms, ms2], ma)
    grid = dsp.beamforming.Regular2DGrid(line, line, ('x', 'y'), ap)
    st = dsp.beamforming.SteeringVector('true power')

    bf = dsp.beamforming.BeamformerCleanSC(s_out, ma, grid, st)
    bf.plot_setting()
    map = bf.get_beamformer_map(
        center_frequency_hz=1500, octave_fraction=0, maximum_iterations=100,
        safety_factor=0.5, remove_diagonal_csm=True)
    grid.plot_map(map, range_db=60)
    bf = dsp.beamforming.BeamformerDASFrequency(s_out, ma, grid, st)
    map = bf.get_beamformer_map(center_frequency_hz=1500, octave_fraction=0)
    grid.plot_map(map, range_db=60)
    bf = dsp.beamforming.BeamformerOrthogonal(s_out, ma, grid, st)
    map = bf.get_beamformer_map(center_frequency_hz=1500, octave_fraction=0)
    grid.plot_map(map, range_db=60)
    bf = dsp.beamforming.BeamformerFunctional(s_out, ma, grid, st)
    map = bf.get_beamformer_map(
        center_frequency_hz=1500, octave_fraction=0, gamma=20)
    grid.plot_map(map, range_db=60)
    try:
        bf = dsp.beamforming.BeamformerMVDR(s_out, ma, grid, st)
        map = bf.get_beamformer_map(
            center_frequency_hz=1500, octave_fraction=0, gamma=5)
        grid.plot_map(map, range_db=60)
    except np.linalg.LinAlgError as e:
        print(e)
        pass
    dsp.plots.show()


def detrending():
    s = dsp.generators.harmonic(
        300, sampling_rate_hz=1500, peak_level_dbfs=-20,
        number_of_channels=2, uncorrelated=True)
    s.plot_time()
    n = 0.3*np.arange(len(s))/len(s)
    s.time_data += n[..., None]
    s.plot_time()
    s2 = dsp.detrend(s, polynomial_order=10)
    s2.plot_time()
    dsp.plots.show()


def iterators():
    s = dsp.generators.noise(sampling_rate_hz=10_000)
    fb = dsp.filterbanks.auditory_filters_gammatone(
        [500, 1000], sampling_rate_hz=10_000)
    mb = fb.filter_signal(s, mode='parallel')
    for n in mb:
        print(type(n))
    for n in fb:
        print(type(n))


if __name__ == '__main__':
    # transfer_function_test()
    # new_transfer_functions()
    # welch_method()
    # csm()
    # group_delay()
    # stft()
    # distances_function_test()
    # minimum_phase_systems()
    # room_acoustics()
    # multiband()
    # save_objects()
    # generators()
    # recording()
    # convolve_rir_signal()
    # merging_signals()
    # merging_fbs()
    # collapse()
    # smoothing()
    # min_phase_signal()
    # lin_phase_signal()
    # gammatone_filters()
    # ir2filt()
    # fwsnrseg()
    # true_peak()
    # harmonic_tone()
    # band_swapping()
    # fractional_time_delay()
    # synthetic_rir()
    # beamforming_basics()
    # beamforming_steering_test()
    # beamforming_virtual_source()
    # beamforming_complete_test_2D()
    # beamforming_complete_test_time()
    # cepstrum()
    # beamforming_3d_grid()
    # beamforming_frequency_formulations()
    # detrending()
    # iterators()

    # Next
    print()
