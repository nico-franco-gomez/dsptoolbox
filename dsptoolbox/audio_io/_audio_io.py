"""
Backend for audio io module
"""

from sounddevice import CallbackStop
from .. import Signal, normalize, fade
from numpy import ndarray


def standard_callback(signal: Signal):
    """This standard callback function takes in a signal and does
    preprocessing (normalization and fade). After that, it returns a callback
    valid for the `sounddevice.OutputStream`.

    Parameters
    ----------
    signal : `Signal`
        Signal to be played through the audio stream.

    Returns
    -------
    call : callable
        Function to be used as callback for the output stream. The signature
        must be valid for sounddevice's callback::

            call(outdata: np.ndarray, frames: int, time, status) -> None

    """
    # Normalize
    signal = normalize(signal)
    # Fade in and fade out
    signal = fade(signal, length_fade_seconds=signal.time_vector_s[-1] * 0.05)

    def call(outdata: ndarray, frames: int, time, status) -> None:
        """Standard version of an audio callback with a signal object.

        Parameters
        ----------
        outdata : `np.ndarray`
            Samples as numpy array with shape (samples, channels).
        frames : int
            Block size in samples.
        time : CData
            See sounddevice's documentation.
        status : `sounddevice.CallbackFlags`
            Warnings and flags if errors happen during streaming.

        """
        if status:
            print(status)
        out, flag = signal.stream_samples(frames, signal_mode=False)
        if flag:
            outdata[: len(out)] = out
            outdata[len(out) :] = 0
            raise CallbackStop()
        outdata[:] = out

    return call
