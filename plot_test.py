import ctypes
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
except AttributeError:
    print("Unable to import windll..")

# Load the .mat files
BER_0_Trd = np.mean(sio.loadmat(r"figure_data/Tradition_Ber_0.mat")['BER_noRIS'], axis=0)
BER_128_Trd = np.mean(sio.loadmat(r"figure_data/Tradition_Ber_128.mat")['BER'], axis=0)
BER_256_Trd = np.mean(sio.loadmat(r"figure_data/Tradition_Ber_256.mat")['BER'], axis=0)
BER_128_e2e = np.mean(sio.loadmat(r"figure_data/E2E_Ber_128.mat")['Ber'], axis=0)
BER_256_e2e = np.mean(sio.loadmat(r"figure_data/E2E_Ber_256.mat")['Ber'], axis=0)


def ber_plot(Ber: torch.Tensor, fname: Path, x1=None, x2=None):
    fig, axs = plt.subplots(2, 1, figsize=(7, 10))

    # Compute the mean along the first axis
    # BER_my = np.mean(Ber, axis=0)
    BER_my = Ber.mean(dim=0)

    # Plot the figure
    x = np.arange(0, 110, 10)
    if x1 is None:
        x1 = torch.arange(0, 101, 10, device="cpu")
    k = 4
    axs[0].semilogy(x, BER_0_Trd[:, k], 'r-o', linewidth=1.7, label='No RIS')
    axs[0].semilogy(x, BER_128_Trd[:, k], 'b-s', linewidth=1.7, label='128 - Traditional')
    axs[0].semilogy(x, BER_256_Trd[:, k], 'r-v', linewidth=1.7, label='256 - Traditional')
    axs[0].semilogy(x, BER_128_e2e[:, k], 'b--s', linewidth=1.7, label='128 - E2E Scheme')
    axs[0].semilogy(x, BER_256_e2e[:, k], 'r--v', linewidth=1.7, label='256 - E2E Scheme')
    axs[0].semilogy(x1.cpu(), BER_my[:, k].cpu(), 'k--o', linewidth=1.7, label='Current Scheme')
    axs[0].grid(True)
    axs[0].axis([0, 100, 1e-6, 1])
    axs[0].set_xlabel('L (m)')
    axs[0].set_ylabel('BER')
    axs[0].legend()

    x = np.arange(-5, 21, 1)
    if x2 is None:
        x2 = torch.arange(-5, 21, 1, device="cpu")
    k = 2
    axs[1].semilogy(x, BER_0_Trd[k, :], 'r-o', linewidth=1.7, label='No RIS')
    axs[1].semilogy(x, BER_128_Trd[k, :], 'b-s', linewidth=1.7, label='128 - Traditional')
    axs[1].semilogy(x, BER_256_Trd[k, :], 'r-v', linewidth=1.7, label='256 - Traditional')
    axs[1].semilogy(x, BER_128_e2e[k, :], 'b--s', linewidth=1.7, label='128 - E2E Scheme')
    axs[1].semilogy(x, BER_256_e2e[k, :], 'r--v', linewidth=1.7, label='256 - E2E Scheme')
    axs[1].semilogy(x2.cpu(), BER_my[k, :].cpu(), 'k--o', linewidth=1.7, label='Current Scheme')
    axs[1].grid(True)
    axs[1].axis([-5, 20, 1e-6, 1])
    axs[1].set_xlabel('SNR (dB)')
    axs[1].set_ylabel('BER')
    axs[1].legend()
    plt.tight_layout()
    fname.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fname, bbox_inches='tight', dpi=200, )
    plt.close()


if __name__ == "__main__":
    t = time.time()
    ber_plot(
        Ber=torch.rand([300, 6, 6]),
        x1=torch.arange(0, 101, 20),
        x2=torch.arange(-5, 21, 5)
    )
    print(f"{time.time() - t:0.8f}")
# print("")
