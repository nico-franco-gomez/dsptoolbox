"""
Backend for the effects module
"""


class AudioEffect():
    """Base class for an audio effect

    """
    def __init__(self, blocking_mode_supported: bool,
                 non_blocking_mode_supported: bool, description: str = None):
        """Base constructor for an audio effect.

        Parameters
        ----------
        blocking_mode_supported : bool
            When `True`, the effect can be applied in a block-processing
            manner.
        non_blocking_mode_supported : bool
            When `True`, non blocking mode (with possibly anti-causal
            operations) is supported.
        description : str, optional
            A string containing a general description about the audio effect.
            Default: `None`.

        """
        self.blocking_mode_supported = blocking_mode_supported
        self.non_blocking_mode_supported = non_blocking_mode_supported
