"""
General tests for filter class
"""


def filter_functionalities():
    import dsptools as dsp
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
    import dsptools as dsp

    # config = dict(eq_type='lowshelf', freqs=1500, gain=10, q=0.7)
    # filt = dsp.Filter('biquad', config)
    # config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    # filt = dsp.Filter('fir', config)
    config = dict(order=10, freqs=[1500, 2000], type_of_pass='bandpass',
                  filter_design_method='bessel')
    filt = dsp.Filter('iir', config)
    recorded_multi = \
        dsp.Signal('/Users/neumanndev/Library/CloudStorage/OneDrive-' +
                   'SennheiserelectronicGmbH&Co.KG/PPONS OneDrive/Polar ' +
                   'Picker Data/test_audio_5.wav')
    # recorded_multi = \
    #     dsp.Signal('/Users/neumanndev/Library/CloudStorage/' +
    #                'OneDrive-SennheiserelectronicGmbH&Co.KG/PPONS ' +
    #                'OneDrive/MORE/Holzmarkt/chirp_10cm/raw_twin.wav')
    new_rec = filt.filter_signal(recorded_multi)
    recorded_multi.plot_magnitude(normalize=None)
    new_rec.plot_magnitude(normalize=None)
    dsp.plots.show()


def filter_bank_add_remove():
    import dsptools as dsp

    fb = dsp.FilterBank()

    # filters = {}
    config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
                  filter_design_method='bessel')
    # filters[0] = dsp.Filter('iir', config)
    fb.add_filter(dsp.Filter('iir', config))
    config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    # filters[1] = dsp.Filter('fir', config)
    fb.add_filter(dsp.Filter('fir', config))
    config = dict(eq_type='highshelf', freqs=1500, gain=10, q=0.7)
    # filters[2] = dsp.Filter('biquad', config)
    fb.add_filter(dsp.Filter('biquad', config))
    fb.show_info(True)
    # fb = dsp.FilterBank(filters, None)
    fb.remove_filter(0)
    fb.show_info(False)
    fb.save_filterbank()


def filter_bank_filter():
    import dsptools as dsp

    # Standard filter bank
    # fb = dsp.FilterBank()
    # config = dict(order=5, freqs=[1500, 2000], type_of_pass='bandpass',
    #               filter_design_method='bessel')
    # fb.add_filter(dsp.Filter('iir', config))
    # config = dict(order=150, freqs=[1500, 2000], type_of_pass='bandpass')
    # fb.add_filter(dsp.Filter('fir', config))

    # Linkwitz-Riley
    fb = dsp.filterbanks.linkwitz_riley_crossovers(
        [1000, 1500], [4, 6])
    # i = dsp.generators.dirac(1024, 2)
    # i_out = fb.filter_signal(i, activate_zi=True)
    # i_out.bands[0].plot_magnitude()
    # i_out.bands[1].plot_magnitude()
    # i_out.bands[2].plot_magnitude()

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


if __name__ == '__main__':
    # filter_functionalities()
    # filtering()
    # filter_bank_add_remove()
    filter_bank_filter()

    print()
