"""
Tests for audio io module should be manual since they have pauses used
for the streamings
"""
import dsptoolbox as dsp
from os.path import join


class TestAudioIOModule:
    speech = dsp.Signal(join("examples", "data", "speech.flac"))

    def test_device_selection(self):
        dsp.audio_io.print_device_info()
        d = dsp.audio_io.print_device_info(device_number=0)
        assert d is not None
        dsp.audio_io.set_device(0)
