from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


class MultichannelData(ABC):
    # ======== Multichannel Data Base Class Implementation ====================
    @abstractmethod
    def _get_data(self) -> NDArray[np.float64 | np.complex128]:
        pass

    @abstractmethod
    def _set_data(self, data: NDArray[np.float64 | np.complex128]) -> None:
        pass

    @abstractmethod
    def _create_copy_with_new_data(self, data: NDArray[np.float64 | np.complex128]):
        pass

    @abstractmethod
    def _update_state(self) -> None:
        pass

    @property
    def number_of_channels(self) -> int:
        return self._get_data().shape[-1]

    def __len__(self):
        return self._get_data().shape[0]

    def remove_channel(self, channel_number: int | None = None):
        """Return a copy with a channel removed.

        Parameters
        ----------
        channel_number : int, None, optional
            Channel number to be removed. Pass None to remove the last one.
            Default: None.

        Returns
        -------
        New object of the same type, with the channel removed.

        """
        data = self._get_data()
        if channel_number is None:
            channel_number = data.shape[-1] - 1
        assert data.shape[-1] > 1, "The only channel cannot be removed"
        assert channel_number in range(data.shape[-1]), (
            f"Channel number {channel_number} does not exist. There are "
            + f"{data.shape[-1]} channels (zero included)."
        )
        return self._create_copy_with_new_data(np.delete(data, channel_number, axis=-1))

    def swap_channels(self, new_order):
        """Return a copy with the channels rearranged in the new given order.

        Parameters
        ----------
        new_order : array-like
            New rearrangement of channels.

        Returns
        -------
        New object of the same type, with the channels rearranged.

        """
        new_order = np.atleast_1d(np.asarray(new_order).squeeze())
        assert new_order.ndim == 1, (
            "Too many or too few dimensions are given in the new "
            + "arrangement vector"
        )
        assert self.number_of_channels == len(new_order), (
            "The number of channels does not match"
        )
        assert all(new_order < self.number_of_channels) and all(new_order >= 0), (
            "Indexes of new channels have to be in "
            + f"[0, {self.number_of_channels - 1}]"
        )
        assert len(np.unique(new_order)) == len(new_order), (
            "There are repeated indexes in the new order vector"
        )
        return self._create_copy_with_new_data(self._get_data()[..., new_order])

    def get_channels(self, channels: int | ArrayLike):
        """Returns a new object with the selected channels. Beware that the
        first channel index is 0!

        Parameters
        ----------
        channels : ArrayLike or int
            Channels to be returned in the new object.

        Returns
        -------
        New object of the same type, with the selected channels.

        """
        channels = np.atleast_1d(np.asarray(channels).squeeze())
        return self._create_copy_with_new_data(self._get_data()[..., channels])

    def sum_channels(self):
        """Return a copy where all channels are summed into one.

        Returns
        -------
        New object of the same type, with a single channel.

        """
        return self._create_copy_with_new_data(
            np.sum(self._get_data(), axis=-1, keepdims=True)
        )
