"""
Tests for audio io module should be manual since they have pauses used
for the streamings
"""

import os
from os.path import join

import numpy as np
import pytest
import sounddevice as sd

import dsptoolbox as dsp


class TestAudioIOModule:
    speech = dsp.Signal(
        join(os.path.dirname(__file__), "..", "example_data", "speech.flac")
    )

    def test_device_selection(self):
        devices = dsp.audio_io.print_device_info()
        if len(devices) == 0:
            pytest.skip("No PortAudio devices are available on this runner")
        d = dsp.audio_io.print_device_info(device_number=0)
        assert d is not None
        dsp.audio_io.set_device(0)

    def test_list_devices_replaces_the_interactive_prompt(self):
        """`set_device()` used to call `input()` when no device was given,
        which cannot be used from a script."""
        assert dsp.audio_io.list_devices() is not None
        with pytest.raises(TypeError):
            dsp.audio_io.set_device()

    def test_print_device_info_invalid_device_raises(self):
        # Pure PortAudio device query, no stream is opened -- safe without
        # real playback/recording hardware.
        with pytest.raises(sd.PortAudioError):
            dsp.audio_io.print_device_info(device_number=99_999)

    def test_set_device_invalid_parameters_raise(self):
        # `set_device` only queries devices and sets the (Python-side)
        # default device config; it never opens a stream.
        with pytest.raises(ValueError):
            dsp.audio_io.set_device("this-device-does-not-exist-xyz-123")
        with pytest.raises(AssertionError):
            dsp.audio_io.set_device([1, 2, 3])
        with pytest.raises(TypeError):
            dsp.audio_io.set_device(3.5)

    def test_play_invalid_parameters_raise(self):
        # All of these assertions fire before any stream is opened.
        sig = dsp.Signal(None, np.zeros((100, 2)), 8_000)
        with pytest.raises(AssertionError):
            dsp.audio_io.play(sig, play_channels=0)
        with pytest.raises(AssertionError):
            dsp.audio_io.play(sig, duration_seconds=-1.0)
        with pytest.raises(AssertionError):
            dsp.audio_io.play(sig, normalized_dbfs=5.0)

    def test_record_invalid_parameters_raise(self):
        with pytest.raises(AssertionError):
            dsp.audio_io.record(rec_channels=0)

    def test_play_and_record_invalid_parameters_raise(self):
        sig = dsp.Signal(None, np.zeros((100, 2)), 8_000)
        with pytest.raises(AssertionError):
            dsp.audio_io.play_and_record(sig, play_channels=[1, 2], rec_channels=0)
        with pytest.raises(AssertionError):
            dsp.audio_io.play_and_record(sig, play_channels=0)
        with pytest.raises(AssertionError):
            # Signal has 2 channels, play_channels only maps 1.
            dsp.audio_io.play_and_record(sig, play_channels=[1])
        with pytest.raises(AssertionError):
            dsp.audio_io.play_and_record(
                sig, play_channels=[1, 2], duration_seconds=-1.0
            )
        with pytest.raises(AssertionError):
            dsp.audio_io.play_and_record(sig, play_channels=[1, 2], normalized_dbfs=5.0)
