import numpy as np
from numpy.typing import NDArray
import sys


def _array_to_bytes_24bits(vector: NDArray[np.int32 | np.uint32]):
    """This function turns an array with samples with type np.int32 or
    np.uint32 into i24 or u24 respectively. The endianness of the current
    platform is kept.

    Parameters
    ----------
    vector : NDArray[np.int32 | np.uint32]
        Vector to transform to bytes.

    Returns
    -------
    bytes

    """
    assert vector.dtype in (
        np.uint32,
        np.int32,
    ), "Vector data type is not supported"
    b = np.frombuffer(vector.tobytes(), dtype=np.uint8)
    if sys.byteorder == "little":
        indices = np.setdiff1d(np.arange(len(b)), np.arange(3, len(b), 4))
    else:
        indices = np.setdiff1d(np.arange(len(b)), np.arange(0, len(b), 4))
    b = b[indices]
    return b.tobytes()


def _bytes_to_array_24bits(vector: bytes, signed_input: bool):
    """Convert bytes into an array."""
    assert (
        len(vector) % 3 == 0
    ), "Vector should have a length with 3-bytes sized samples"
    output_format = eval(f"np.{"int" if signed_input else "uint"}32")
    values = [
        int.from_bytes(vector[n : n + 3], sys.byteorder, signed=signed_input)
        for n in range(0, len(vector), 3)
    ]
    return np.asarray(values, dtype=output_format)
