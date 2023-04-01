import numpy as np
import torch

from logging_config import get_logger

logger = get_logger(__name__)

def onehot2bit(X):
    """
    This function converts one-hot encoded data to bit data using a pre-defined bit sequence (BIT_16).
    """
    BIT_16 = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
         [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
         [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
         [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
        dtype=torch.float
    )
    if type(X) is not torch.Tensor:
        # X = torch.tensor(X, dtype=torch.float)
        raise TypeError(f"X is not Tensor")
    bit = torch.matmul(X, BIT_16)
    return bit


def generate_transmit_data(M, J, num, seed=0):
    torch.manual_seed(seed)
    symbol_index = torch.randint(M, size=(num * J,))
    X = torch.zeros((num * J, M), dtype=torch.float32)
    X[torch.arange(num * J), symbol_index] = 1
    X = X.reshape(num, M * J)
    Y = X
    return X, Y


def BER(X, sys, Y_pred, num_test):
    """
    This function calculates the bit error rate (BER) of the predicted data against the actual data.
    :param X: one-hot encoded data
    :param sys: communication system object
    :param Y_pred: predicted data
    :param num_test: number of test samples
    :return: ber: bit error rate
    """
    # Convert prediction to binary data
    Y_pred[Y_pred < 0.5] = 0
    Y_pred[Y_pred >= 0.5] = 1

    Y_pred = Y_pred.view(num_test * sys.Num_User, sys.k)
    X = X.view(num_test * sys.Num_User, sys.M)
    X_bit = onehot2bit(X)
    X_bit = X_bit.view(num_test * sys.Num_User, sys.k)
    err = torch.sum(torch.abs(Y_pred - X_bit))
    ber = err / (num_test * sys.Num_User * sys.k)
    return ber.item()


def generate_rate_data(M, J):
    """
    This function generates transmit data for calculating the channel capacity.
    :param M: number of symbols
    :param J: number of bits per symbol

    :return: symbol_index: array of symbol indices
        X: one-hot encoded data
        Y: one-hot encoded data (same as X)
    """
    logger.info(f'\tGenerating rate data: M = {M}')
    symbol_index = torch.arange(M)
    X = torch.tile(torch.eye(M), (1, J))
    Y = X
    return symbol_index, X, Y


def calcul_rate(y, Rece_Ampli, Num_Antenna):
    """
    This function calculates the channel capacity of the given data using a pre-defined QAM constellation.
    :param y: channel input data
    :param Rece_Ampli: receiver antenna amplification
    :param Num_Antenna: number of receiver antennas
    :return: mean_rate: mean channel capacity
        max_rate: maximum channel capacity
    """
    logger.info("\tCalculating rate..")
    z = y[:, 0:Num_Antenna] + 1j * y[:, Num_Antenna:2 * Num_Antenna]
    zt = y[:, 0:Num_Antenna] - 1j * y[:, Num_Antenna:2 * Num_Antenna]
    # rate = np.log2(1 + np.matmul(z, zt.T) / Rece_Ampli ** 2)
    rate = torch.log2(1 + torch.matmul(z, zt.t()) / Rece_Ampli ** 2)
    # mean_rate = torch.mean(torch.diag(rate))
    # max_rate = torch.max(torch.diag(rate).cpu())
    diag = torch.diag(rate).cpu().numpy()
    mean_rate = np.mean(diag)
    max_rate = np.max(diag)
    return mean_rate, max_rate
