from enum import Enum, auto


class TransferFunctionType(Enum):
    """Types of transfer functions for stochastic signals:

    - H1: for noise in the output signal. `Gxy/Gxx`.
    - H2: for noise in the input signal. `Gyy/Gyx`.
    - H3: for noise in both signals. `G_xy / abs(G_xy) * (G_yy/G_xx)**0.5`.

    """

    H1 = auto()
    H2 = auto()
    H3 = auto()
