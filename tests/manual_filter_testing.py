"""
General tests for filter class
"""
import dsptoolbox as dsp
import numpy as np
from os.path import join


def linkwitz_riley():
    # Linkwitz-Riley filter bank
    fb = dsp.filterbanks.linkwitz_riley_crossovers([1000, 1500], [4, 6])
    fb.show_info()
    # fb.plot_phase(mode='summed', unwrap=True,
    #               test_zi=False, zero_phase=False)
    # fb.plot_group_delay(mode='summed', zero_phase=True)

    # Parallel
    # fb.plot_magnitude(test_zi=True)
    fb.plot_magnitude(test_zi=False, zero_phase=True)
    # fb.plot_magnitude(mode='parallel', test_zi=False, zero_phase=True)
    dsp.plots.show()


def perfect_reconstruction():
    fb = dsp.filterbanks.reconstructing_fractional_octave_bands(
        sampling_rate_hz=44100
    )
    fb.show_info()
    # fig, ax = \
    #     fb.plot_phase(
    #         mode='parallel', length_samples=2**12, unwrap=True)
    fig, ax = fb.plot_magnitude(mode="parallel")
    # ax.set_ylim([-5, 5])
    # fb.plot_phase(mode='parallel', unwrap=True)
    # fb.plot_group_delay(mode='summed')
    dsp.plots.show()


def gamma_tone_reconstruction():
    # s = dsp.generators.dirac(length_samples=2**12, number_of_channels=2)
    s = dsp.generators.noise(
        length_seconds=2, number_of_channels=1, sampling_rate_hz=44100
    )
    s.plot_magnitude()
    g_dsp = dsp.filterbanks.auditory_filters_gammatone(
        [20, 20e3], sampling_rate_hz=44100
    )
    # g_dsp.plot_magnitude()
    s_bla = g_dsp.filter_signal(s)
    s2 = g_dsp.reconstruct(s_bla)
    s_bla.collapse().plot_magnitude()
    s2.plot_magnitude()
    dsp.plots.show()


def qmf_crossovers():
    # Sampling rate
    fs_hz = 48000
    ny_hz = fs_hz // 2

    # Signal
    s = dsp.Signal(join("examples", "data", "speech.flac"))
    # s = dsp.generators.noise(sampling_rate_hz=fs_hz)

    # f = dsp.Filter('fir',
    #                dict(freqs=ny_hz//2, order=257, type_of_pass='lowpass'),
    #                sampling_rate_hz=fs_hz)
    f = dsp.Filter(
        "iir",
        dict(
            freqs=ny_hz // 2,
            order=5,
            type_of_pass="lowpass",
            filter_design_method="butter",
        ),
        sampling_rate_hz=fs_hz,
    )

    # Create second filter
    b, _ = f.get_coefficients(mode="ba")
    b2 = b.copy()
    b2[1::2] *= -1
    f2 = dsp.Filter("other", dict(ba=[b2, [1]]), sampling_rate_hz=fs_hz)

    # Create crossover
    fb = dsp.filterbanks.qmf_crossover(f)
    bafter, _ = fb.filters[1].get_coefficients(mode="ba")
    # Check that created filters are right
    print(np.all(b2 == bafter))

    s_manual1 = f.filter_signal(s)
    s_manual2 = f2.filter_signal(s)
    s_manual1.plot_magnitude()
    s_manual2.plot_magnitude()

    # When downsampling, aliasing leads the high-frequency signal to have an
    # spectrum just like the low-frequency part
    s_fb = fb.filter_signal(s, downsample=False)
    s_fb.bands[0].plot_magnitude()
    s_fb.bands[1].plot_magnitude()

    # s_rec = fb.reconstruct_signal(s_fb, upsample=True)

    # s.plot_magnitude()
    # s_rec.plot_magnitude()
    # s.plot_time()
    # s_rec.plot_time()
    # dsp.audio_io.set_device(1)
    # dsp.audio_io.play(s)
    # dsp.audio_io.play(s_rec)
    # print(dsp.distances.si_sdr(s, s_rec))

    # Take a look inside
    # fb.plot_magnitude(mode='parallel', downsample=False)
    # fb.plot_magnitude(mode='summed', downsample=True)
    # fb.plot_phase(mode='summed', unwrap=True)
    # fb.plot_group_delay(mode='summed')
    dsp.plots.show()


def fractional_octave_bands():
    fs = 44100
    length = 2**18
    fb = dsp.filterbanks.fractional_octave_bands(
        sampling_rate_hz=fs,
        # filter_order=14,
        frequency_range_hz=[32, 10e3],
    )

    fb.plot_magnitude(length_samples=length, test_zi=False)
    dsp.plots.show()


def test1():
    import pyfar as pf
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    # generate the data
    x = pf.signals.impulse(2**17)
    y = pf.dsp.filter.fractional_octave_bands(x, 1, freq_range=(20, 8e3))
    # frequency domain plot
    y_sum = pf.FrequencyData(np.sum(np.abs(y.freq) ** 2, 0), y.frequencies)
    pf.plot.freq(y)
    ax = pf.plot.freq(y_sum, color="k", log_prefix=10, linestyle="--")
    ax.set_title("Filter bands and the sum of their squared magnitudes")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # linkwitz_riley()
    # perfect_reconstruction()
    # gamma_tone_reconstruction()
    # qmf_crossovers()
    fractional_octave_bands()
    # test1()

    print()
