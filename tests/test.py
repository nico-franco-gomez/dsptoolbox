'''
Testing script
'''
from os.path import join


def plotting_test():
    import dsptools as dsp

    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    recorded_multi.plot_magnitude(show_info_box=True)
    recorded_multi.plot_time()
    dsp.plots.show()


def package_structure_test():
    import dsptools as dsp
    recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    print(dsp.latency(recorded, raw))


def transfer_function_test():
    import dsptools as dsp

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized', multichannel=True,
        padding=False, keep_original_length=True)
    tf_wind = dsp.transfer_functions.window_ir(tf, at_start=False)
    tf_wind.plot_time()
    tf.plot_time()
    tf.plot_magnitude()
    dsp.plots.show()


def distances_function_test():
    import dsptools as dsp
    recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    print(dsp.distances.itakura_saito(raw, recorded))
    print(dsp.distances.log_spectral(recorded, raw))


def welch_method():
    import dsptools as dsp
    import matplotlib.pyplot as plt

    # raw = dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    raw.set_spectrum_parameters(method='welch', overlap_percent=50)
    # raw.plot_magnitude()
    f1, mine = raw.get_spectrum()
    from scipy.signal import welch
    import numpy as np
    f2, pp = welch(raw.time_data.squeeze(),
                   raw.sampling_rate_hz, nperseg=1024,
                   average='mean')
    # f2, pp = welch(raw.time_data.squeeze()[:, 0],
    #                raw.sampling_rate_hz, nperseg=1024,
    #                average='mean')
    plt.figure()
    plt.semilogx(f2, 10*np.log10(pp), label='Scipy')
    plt.semilogx(f1, 10*np.log10(mine), label='mine')
    plt.legend()
    dsp.plots.show()


def csm():
    import dsptools as dsp
    # import numpy as np

    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw.plot_csm()
    # f, csm = raw.get_csm()
    # fig, ax = plt.subplots(csm.shape[1], csm.shape[2], figsize=(6, 6),
    #                        sharey=True)
    # for n1 in range(csm.shape[1]):
    #     for n2 in range(csm.shape[1]):
    #         ax[n1, n2].semilogx(f, 10*np.log10(np.abs(csm[:, n1, n2])))
    #         # ax[n1, n2].plot(f, np.angle(csm[:, n1, n2]))
    # fig.tight_layout()
    dsp.plots.show()


def smoothing():
    import dsptools as dsp
    import numpy as np
    import matplotlib.pyplot as plt
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    f, bla = raw.get_spectrum()
    spec_db = np.abs(bla[100:, 0])
    spec_s = dsp.experimental._smoothing_log(
        spec_db, 15, 'hann')
    # import pyfar as pf
    # pf.dsp.smooth_fractional_octave()

    plt.plot(f[100:], spec_db)
    plt.plot(f[100:], spec_s)
    dsp.plots.show()


def group_delay():
    import dsptools as dsp
    import matplotlib.pyplot as plt

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized', multichannel=True)
    tf = dsp.transfer_functions.window_ir(tf)
    f, g1 = dsp.group_delay(tf, 'matlab')
    f, g2 = dsp.group_delay(tf, 'direct')

    plt.plot(f, g1, label='matlab')
    plt.plot(f, g2, label='direct')
    plt.legend()
    dsp.plots.show()


def add_channels():
    import dsptools as dsp
    import soundfile as sf

    audio, fs = sf.read(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    in_sig = dsp.Signal(None, audio[:, 0], fs)
    print(in_sig.time_data.shape)
    # import numpy as np
    # audio = np.concatenate([audio, np.zeros((200, 2))])
    in_sig.add_channel(None, audio[:-20, 1], fs)
    print(in_sig.time_data.shape)


def remove_channels():
    import dsptools as dsp

    audio = dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    first = audio.time_data[:, 0]
    # second = audio.time_data[:, 1]
    audio.remove_channel(1)
    print(audio.time_data.shape)
    print(all(first == audio.time_data[:, 0]))


def stft():
    import dsptools as dsp
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa

    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    raw.set_spectrogram_parameters(window_length_samples=2048)
    t, f, stft = raw.get_spectrogram()
    D = librosa.stft(raw.time_data.squeeze(), center=False)
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
    import dsptools as dsp
    import matplotlib.pyplot as plt

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized', multichannel=True)
    tf = dsp.transfer_functions.window_ir(tf)
    f, bla = dsp.minimal_phase(tf)
    plt.subplot(121)
    plt.semilogx(f, bla)
    plt.subplot(122)
    f, bla = dsp.minimal_group_delay(tf)
    f, bla2 = dsp.group_delay(tf)
    plt.semilogx(f, (bla2-bla)*1e3)
    dsp.plots.show()


def room_acoustics():
    import dsptools as dsp

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.spectral_deconvolve(
        recorded_multi, raw, mode='regularized', multichannel=True)
    tf = dsp.transfer_functions.window_ir(tf)
    print(dsp.room_acoustics.reverb_time(tf, 'T20'))
    print(dsp.room_acoustics.reverb_time(tf, 'T30'))
    print(dsp.room_acoustics.reverb_time(tf, 'T60'))
    print(dsp.room_acoustics.reverb_time(tf, 'EDT'))
    print(dsp.room_acoustics.find_modes(tf, [30, 350]))


def window_length():
    from scipy.signal import windows
    import matplotlib.pyplot as plt
    import numpy as np
    window_length = 1024
    s = np.zeros(20000)
    overlap = 0.75
    overlap_samples = int(overlap * window_length)
    step = window_length - overlap_samples

    w = windows.hann(window_length, sym=True)
    b = int(np.floor(len(s) / step)) + 1
    rem = int(len(s) % step)
    s = np.hstack([s, np.zeros(window_length - rem)])
    i = 0
    start = 0
    for n in range(b):
        s[start:start+window_length] += w
        start += step
        i += 1
    print(np.all(np.flip(s) == s))
    print(np.sum(s[-100:] == 0))
    print(np.sum(s[:100] == 0))

    plt.plot(s)
    plt.show()


def new_transfer_functions():
    import dsptools as dsp

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))
    tf = dsp.transfer_functions.compute_transfer_function(
        recorded_multi, raw, multichannel=True, mode='h2')
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


def spectrogram_plot():
    import dsptools as dsp

    # recorded = dsp.Signal(join('..', 'examples', 'data', 'chirp_mono.wav'))
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    recorded_multi.plot_spectrogram()
    dsp.plots.show()


def multiband():
    import dsptools as dsp
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
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
    import dsptools as dsp
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    recorded_multi.save_signal(mode='wav')
    recorded_multi.save_signal(mode='flac')
    recorded_multi.save_signal(mode='pickle')
    print()


def generators():
    import dsptools as dsp

    # Noise
    # wno = dsp.generators.noise('white', length_seconds=3, peak_level_dbfs=-1)
    # wno = dsp.generators.noise('pink')
    # wno = dsp.generators.noise('red', peak_level_dbfs=-5)
    # wno = dsp.generators.noise('blue')
    # wno = dsp.generators.noise('violet')
    # wno.plot_magnitude(normalize=None)

    # Chirps
    wno = dsp.generators.chirp(type_of_chirp='log', length_seconds=5,
                               fade='log',
                               padding_end_seconds=2, number_of_channels=1,
                               peak_level_dbfs=-20, range_hz=[20, 24e3])

    # Plots
    wno.plot_magnitude(range_hz=[20, 24e3])
    wno.plot_spectrogram()
    wno.plot_time()
    # wno.plot_phase()
    # wno.plot_group_delay()
    dsp.plots.show()


def recording():
    import dsptools as dsp
    from time import sleep

    sleep(3)

    dsp.measure.set_device()
    chirp = dsp.generators.chirp(padding_end_seconds=2)
    s2 = dsp.measure.play_and_record(chirp)
    tf = dsp.transfer_functions.spectral_deconvolve(s2, chirp)
    tf = dsp.transfer_functions.window_ir(tf)
    tf.plot_magnitude()
    dsp.plots.show()


def swapping_channels():
    import dsptools as dsp
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    recorded_multi.plot_time()
    recorded_multi.swap_channels([1, 0])
    recorded_multi.plot_time()
    dsp.plots.show()


def convolve_rir_signal():
    import dsptools as dsp
    rir = dsp.Signal(join('..', 'examples', 'data', 'rir.wav'),
                     signal_type='rir')
    speech = dsp.Signal(join('..', 'examples', 'data', 'speech.flac'))
    dsp.measure.set_device(2)
    # dsp.measure.play(speech)
    new_speech = \
        dsp.room_acoustics.convolve_rir_on_signal(
            speech, rir, keep_length=False)
    dsp.measure.play(new_speech)
    dsp.plots.show()


def cepstrum():
    import dsptools as dsp
    import matplotlib.pyplot as plt
    speech = dsp.Signal(join('..', 'examples', 'data', 'speech.flac'))
    c = dsp.special.cepstrum(speech, mode='real')
    plt.plot(c)
    # dsp.plots.general_plot(speech.get_time_vector(), c, log=False)
    dsp.plots.show()


def merging_signals():
    import dsptools as dsp
    
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    raw = dsp.Signal(join('..', 'examples', 'data', 'chirp.wav'))

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
    import dsptools as dsp

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
    import dsptools as dsp

    d = dsp.generators.dirac(length_samples=2**13)
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
    import dsptools as dsp
    import pyfar as pf
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np

    audio, fs = sf.read(join('..', 'examples', 'data', 'chirp_mono.wav'))
    psig2 = dsp.Signal(time_data=audio, sampling_rate_hz=fs)
    psig2.set_spectrum_parameters(method='standard')
    psig2.plot_magnitude(smoothe=4)
    dsp.plots.show()


if __name__ == '__main__':
    # plotting_test()
    # package_structure_test()
    # transfer_function_test()
    # welch_method()
    # csm()
    # group_delay()
    # add_channels()
    # remove_channels()
    # stft()
    # distances_function_test()
    # minimum_phase_systems()
    # room_acoustics()
    # window_length()
    # spectrogram_plot()
    # multiband()
    # save_objects()
    # generators()
    # recording()
    # swapping_channels()
    # convolve_rir_signal()
    # merging_signals()
    # merging_fbs()
    # collapse()

    # Weird results - needs validation
    # new_transfer_functions()  # -- coherence function from scipy (not quite)
    # cepstrum()  # -- scipy.signal.complex_cepstrum, real_cepstrum

    # Not working so far
    # smoothing()

    # Next
    smoothing()
    print()
