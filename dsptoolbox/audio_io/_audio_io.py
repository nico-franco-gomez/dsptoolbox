"""
Backend for audio io module
"""
from sounddevice import CallbackStop
from dsptoolbox import Signal
from numpy import ndarray


def standard_callback(signal: Signal):
    """This is a standard callback that passes blocks of samples to an output
    stream. The arguments are fixed and must match the expected signature
    of the `sounddevice.OutputStream` object.

    Returns
    -------
    call : callable
        Function to be used as callback for the output stream. The signature
        must be valid for sounddevice's callback::

            call(outdata: ndarray, frames: int, time, status) -> None

    """

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
        # Out as a signal object
        out = signal.stream_samples(frames)
        # Cast to numpy array
        out = out.time_data
        chunksize = len(out)
        if chunksize < frames:
            outdata[:chunksize] = out
            outdata[chunksize:] = 0
            raise CallbackStop()
        outdata[:] = out
    return call
