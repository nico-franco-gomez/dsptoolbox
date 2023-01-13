"""
General tests for filter class
"""
from os.path import join


def filter_functionalities():
    import dsptoolbox as dsp
    from matplotlib.pyplot import show
    config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
                  filter_design_method='bessel')
    rF = dsp.Filter('iir', config)
    # config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    # rF = dsp.Filter('fir', config)
    # config = dict(eq_type=7, freqs=1500, gain=10, q=0.7)
    # rF = dsp.Filter('biquad', config)
    # rF.show_filter_parameters()
    # rF.plot_magnitude(2048)
    # rF.plot_group_delay(2048)
    # rF.plot_phase(2048, unwrap=False)
    rF.plot_zp(show_info_box=True)
    rF.save_filter()
    show()


def filtering():
    import dsptoolbox as dsp

    # config = dict(eq_type='lowshelf', freqs=1500, gain=10, q=0.7)
    # filt = dsp.Filter('biquad', config)
    config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    filt = dsp.Filter('fir', config)
    # config = dict(order=10, freqs=[1500, 2000], type_of_pass='bandpass',
    #               filter_design_method='bessel')
    # filt = dsp.Filter('iir', config)
    recorded_multi = \
        dsp.Signal(join('..', 'examples', 'data', 'chirp_stereo.wav'))
    new_rec = filt.filter_signal(recorded_multi)
    recorded_multi.plot_magnitude(normalize=None)
    new_rec.plot_magnitude(normalize=None)
    dsp.plots.show()


def filter_bank_add_remove():
    import dsptoolbox as dsp
    fb = dsp.FilterBank()
    fb.show_info(True)

    # Filter 1
    config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
                  filter_design_method='bessel')
    fb.add_filter(dsp.Filter('iir', config))
    fb.show_info(True)
    # Filter 2
    config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    fb.add_filter(dsp.Filter('fir', config))
    # Filter 3
    config = dict(eq_type='highshelf', freqs=1500, gain=10, q=0.7)
    fb.add_filter(dsp.Filter('biquad', config))
    # Filter 4
    config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    fb.add_filter(dsp.Filter('fir', config), index=0)
    # Show info
    fb.show_info(True)

    # Remove
    fb.remove_filter(0)
    fb.show_info(False)

    new_order = [2, 1, 0]
    fb.swap_filters(new_order)
    fb.show_info(True)
    # fb.save_filterbank()


def filter_bank_filter():
    import dsptoolbox as dsp

    # Standard filter bank
    fb = dsp.FilterBank()
    config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
                  filter_design_method='bessel')
    fb.add_filter(dsp.Filter('iir', config))
    config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    fb.add_filter(dsp.Filter('fir', config))

    # Parallel
    # fb.plot_magnitude(test_zi=True)
    fb.plot_magnitude(test_zi=False)
    # fb.plot_magnitude(mode='sequential', test_zi=True)

    # Single filter
    # i = dsp.generators.dirac(1024, 2)
    # config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
    #               filter_design_method='bessel')
    # filt1 = dsp.Filter('iir', config)
    # filt1.filter_signal(i, activate_zi=True).plot_magnitude(normalize=None)
    # filt1.filter_signal(i, activate_zi=False).plot_magnitude(normalize=None)
    dsp.plots.show()


def linkwitz_riley():
    import dsptoolbox as dsp

    # Linkwitz-Riley filter bank
    fb = dsp.filterbanks.linkwitz_riley_crossovers(
        [1000, 1500], [4, 6])
    fb.show_info()
    # fb.plot_phase(unwrap=True, test_zi=True)
    # fb.plot_group_delay()

    # Parallel
    # fb.plot_magnitude(test_zi=True)
    # fb.plot_magnitude(test_zi=False)
    # fb.plot_magnitude(mode='sequential', test_zi=True)
    dsp.plots.show()


def perfect_reconstruction():
    import dsptoolbox as dsp

    fb = dsp.filterbanks.reconstructing_fractional_octave_bands()
    fb.show_info()
    # fig, ax = \
    #     fb.plot_phase(
    #         mode='parallel', length_samples=2**12, unwrap=True, returns=True)
    fig, ax = \
        fb.plot_magnitude(mode='parallel', returns=True)
    # ax.set_ylim([-5, 5])
    # fb.plot_phase(mode='parallel', unwrap=True)
    # fb.plot_group_delay(mode='summed')
    dsp.plots.show()


def gamma_tone_reconstruction():
    import dsptoolbox as dsp

    # s = dsp.generators.dirac(length_samples=2**12, number_of_channels=2)
    s = dsp.generators.noise(length_seconds=2, number_of_channels=1)
    s.plot_magnitude()
    g_dsp = dsp.filterbanks.auditory_filters_gammatone([20, 20e3])
    s_bla = g_dsp.filter_signal(s)
    s2 = g_dsp.reconstruct(s_bla)
    s_bla.collapse().plot_magnitude()
    s2.plot_magnitude()
    dsp.plots.show()


if __name__ == '__main__':
    # filter_functionalities()
    # filtering()
    # filter_bank_add_remove()
    # filter_bank_filter()
    # linkwitz_riley()
    # perfect_reconstruction()
    # gamma_tone_reconstruction()

    print()
