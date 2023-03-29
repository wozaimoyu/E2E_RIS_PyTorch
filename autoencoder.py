import numpy as np


def onehot2bit(X):
    """
    This function converts one-hot encoded data to bit data using a pre-defined bit sequence (BIT_16).
    :param X: one-hot encoded data
    :return: bit: bit data
    """
    BIT_16 = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                       [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                       [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                       [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    bit = np.matmul(X, BIT_16)
    return bit


def generate_transmit_data(M, J, num, seed=0):
    """
    This function generates transmit data with a given number of symbols.

    Args:
        M (int): the number of modulation symbols.
        J (int): the number of antennas.
        num (int): the number of symbols.
        seed (int): seed for the random number generator.

    Returns:
        symbol_index (ndarray): an array of indices of the transmitted symbols.
        X (ndarray): an array of one-hot-encoded transmitted symbols.
        Y (ndarray): an array of one-hot-encoded received symbols, which is the same as X.
    """
    # print('Generate transmit data: M = %d, seed = %d' %(M, seed))
    np.random.seed(seed)
    symbol_index = np.random.randint(M, size=num * J)
    X = np.zeros((num * J, M), dtype='float32')
    # Y = np.zeros((num * J, M), dtype='float32')
    for i in range(num * J):
        X[i, symbol_index[i]] = 1
    X = np.reshape(X, [num, M * J])
    Y = X
    return symbol_index, X, Y


def BER(X, sys, Y_pred, num_test):
    """
    This function calculates the bit error rate (BER) of the predicted data against the actual data.
    :param X: one-hot encoded data
    :param sys: communication system object
    :param Y_pred: predicted data
    :param num_test: number of test samples
    :return: ber: bit error rate
    """
    Y_pred[Y_pred < 0.5] = 0
    Y_pred[Y_pred >= 0.5] = 1
    Y_pred = np.reshape(Y_pred, [num_test * sys.Num_User, sys.k])
    X = np.reshape(X, [num_test * sys.Num_User, sys.M])
    X_bit = onehot2bit(X)
    X_bit = np.reshape(X_bit, [num_test * sys.Num_User, sys.k])
    err = np.sum(np.abs(Y_pred - X_bit))
    ber = err / (num_test * sys.Num_User * sys.k)
    return ber


def generate_rate_data(M, J):
    """
    This function generates transmit data for calculating the channel capacity.
    :param M: number of symbols
    :param J: number of bits per symbol

    :return: symbol_index: array of symbol indices
        X: one-hot encoded data
        Y: one-hot encoded data (same as X)
    """
    symbol_index = np.arange(M)
    X = np.tile(np.eye(M), (1, J))
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
    z = y[:, 0:Num_Antenna] + 1j * y[:, Num_Antenna:2 * Num_Antenna]
    zt = y[:, 0:Num_Antenna] - 1j * y[:, Num_Antenna:2 * Num_Antenna]
    rate = np.log2(1 + np.matmul(z, zt.T) / Rece_Ampli ** 2)
    mean_rate = np.mean(np.diag(rate))
    max_rate = np.max(np.diag(rate))
    return mean_rate, max_rate
