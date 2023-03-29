import numpy as np
import torch


def rayleigh_chan(Ta: int, Ra: int, L: int):
    """
    Generates a Rayleigh channel of size (LxRaxTa)

    :param Ta:
    :param Ra:
    :param L:
    :return: Rayleigh Channel distribution
    """
    # rand = np.random.normal
    # H = np.empty((L, Ra, Ta), dtype=np.complex128)
    # for l in range(L):
    #     G = rand(scale=1 / np.sqrt(2), size=(Ra, Ta))
    #     J = rand(scale=1 / np.sqrt(2), size=(Ra, Ta))
    #     H[l, :, :] = G + 1j * J
    # return H
    rand = torch.randn
    H = torch.empty((L, Ra, Ta), dtype=torch.complex128)
    r2 = np.sqrt(2)
    for l in range(L):
        G = rand((Ra, Ta)) / r2
        J = rand((Ra, Ta)) / r2
        H[l, :, :] = G + 1j * J
    return H


def nakagami_chan(Ta: int, Ra: int, L: int, m: float = 1 / 2):
    """
    Generates a Nakagami-m channel of size (LxRaxTa)

    :param Ta:
    :param Ra:
    :param L:
    :param m: Nakagami-m parameter
    :return: Nakagami-channel distribution
    """
    z = np.random.gamma(shape=m, scale=1 / m, size=(L, Ra, Ta))
    x = np.sqrt(z)
    # Generate random phases
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=(L, Ra, Ta))
    # Combine magnitude and phase to get complex coefficients
    H = x * np.exp(1j * theta)
    return H


def rician_chan(Ta: int, Ra: int, L: int, K: float = 3):
    """
    Generated a Rician-K channel distribution of size (LxRaxTa)
    :param Ta:
    :param Ra:
    :param L:
    :param K: Rician K-factor
    :return:
    """
    # Generate complex-valued channel coefficients
    H = np.sqrt(0.5 * (K / (K + 1))) * np.random.randn(L, Ra, Ta) \
        + np.sqrt(0.5 / (K + 1)) * np.random.randn(L, Ra, Ta) * 1j
    return H
