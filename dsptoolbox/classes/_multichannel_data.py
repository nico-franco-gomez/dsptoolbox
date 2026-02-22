import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class MultichannelData(ABC):

    # ======== Multichannel Data Base Class Implementation ====================
    @abstractmethod
    def _get_data(self) -> NDArray[np.float64 | np.complex128]:
        pass

    @abstractmethod
    def _set_data(self, data: NDArray[np.float64 | np.complex128]) -> None:
        pass

    @abstractmethod
    def _create_copy_with_new_data(
        self, data: NDArray[np.float64 | np.complex128]
    ):
        pass

    @abstractmethod
    def _update_state(self) -> None:
        pass

    @property
    def number_of_channels(self) -> int:
        return self._get_data().shape[-1]

    def __len__(self):
        return self._get_data().shape[0]

    def remove_channel(self, channel_number: int = -1):
        """Removes a channel.

        Parameters
        ----------
        channel_number : int, optional
            Channel number to be removed. Default: -1 (last).

        Returns
        -------
        self

        """
        data = self._get_data()
        if channel_number == -1:
            channel_number = data.shape[1] - 1
        assert data.shape[1] > 1, "Cannot not erase only channel"
        assert data.shape[1] - 1 >= channel_number, (
            f"Channel number {channel_number} does not exist. Signal only "
            + f"has {self.number_of_channels - 1} channels (zero included)."
        )
        self._set_data(np.delete(data, channel_number, axis=-1))
        self._update_state()
        return self

    def swap_channels(self, new_order):
        """Rearranges the channels (inplace) in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of channels.

        Returns
        -------
        self

        """
        new_order = np.atleast_1d(np.asarray(new_order).squeeze())
        assert new_order.ndim == 1, (
            "Too many or too few dimensions are given in the new "
            + "arrangement vector"
        )
        assert self.number_of_channels == len(
            new_order
        ), "The number of channels does not match"
        assert all(new_order < self.number_of_channels) and all(
            new_order >= 0
        ), (
            "Indexes of new channels have to be in "
            + f"[0, {self.number_of_channels - 1}]"
        )
        assert len(np.unique(new_order)) == len(
            new_order
        ), "There are repeated indexes in the new order vector"
        self._set_data(self._get_data()[:, new_order])
        self._update_state()
        return self

    def get_channels(self, channels):
        """Returns a signal object with the selected channels. Beware that
        first channel index is 0!

        Parameters
        ----------
        channels : array-like or int
            Channels to be returned as a new Signal object.

        Returns
        -------
        new_sig : `Signal`
            New signal object with selected channels.

        """
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        return self._create_copy_with_new_data(self._get_data()[:, channels])

    def sum_channels(self):
        """Return a copy of the signal where all channels are summed into one.

        Returns
        -------
        Signal
            New signal with a single channel.

        """
        return self._create_copy_with_new_data(
            np.sum(self._get_data(), axis=1, keepdims=True)
        )
