import numpy as np
import torch


def rayleigh_chan(Ta: int, Ra: int, L: int):
    """
    Generates a Rayleigh channel of size (LxRaxTa)
    """
    print(f"Generating a ({L}x{Ra}x{Ta}) Rayleigh Channel Distribution")
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
    """
    # Generate complex-valued channel coefficients
    H = np.sqrt(0.5 * (K / (K + 1))) * np.random.randn(L, Ra, Ta) \
        + np.sqrt(0.5 / (K + 1)) * np.random.randn(L, Ra, Ta) * 1j
    return H
