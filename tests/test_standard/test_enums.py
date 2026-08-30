"""
Tests for the enums exposed by the standard module.
"""

import numpy as np
import pytest

import dsptoolbox as dsp
from dsptoolbox.standard.enums import ParametrizedWindow, Window


class TestWindowEnum:
    def test_extra_parameter_does_not_leak_between_callers(self):
        """Binding a parameter must not mutate the shared enum member."""
        first = Window.Kaiser.with_extra_parameter(8.0)
        second = Window.Kaiser.with_extra_parameter(2.0)
        assert isinstance(first, ParametrizedWindow)
        assert first.extra_parameter == 8.0
        assert second.extra_parameter == 2.0
        assert first.to_scipy_format() == ("kaiser", 8.0)
        assert second.to_scipy_format() == ("kaiser", 2.0)

    def test_missing_extra_parameter_raises_clear_error(self):
        with pytest.raises(ValueError, match="with_extra_parameter"):
            Window.Gaussian.to_scipy_format()

    def test_plain_window_rejects_extra_parameter(self):
        with pytest.raises(ValueError):
            Window.Hann.with_extra_parameter(3.0)

    def test_general_gaussian_needs_two_parameters(self):
        with pytest.raises(ValueError):
            Window.GeneralGaussian.with_extra_parameter(3.0)
        w = Window.GeneralGaussian.with_extra_parameter((1.5, 7.0))
        assert w.to_scipy_format() == ("general_gaussian", 1.5, 7.0)

    def test_parametrized_window_is_usable_like_a_window(self):
        w = Window.Chebwin.with_extra_parameter(60)
        values = w(64, True)
        assert len(values) == 64
        assert w.needs_extra_parameter()

        fs = 48000
        ir = dsp.ImpulseResponse.from_time_data(
            np.random.default_rng(0).normal(0, 0.05, (2048, 1)), fs
        )
        windowed, _ = dsp.transfer_functions.window_centered_ir(
            ir, 1024, window_type=Window.Gaussian.with_extra_parameter(500)
        )
        assert windowed.length_samples == 1024
