import ctypes
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

# Load the .mat file
mat = sio.loadmat(r"figure\E2E_Ber_256.mat")
Ber_E2E = mat['Ber']
BER256_e2e = np.mean(Ber_E2E, axis=0)


def ber_plot(Ber: torch.Tensor, x1=None, x2=None):
    fig, axs = plt.subplots(2, 1, figsize=(7, 10))
    axs[0].clear()
    axs[1].clear()

    # Compute the mean along the first axis
    # BER256_my = np.mean(Ber, axis=0)
    BER256_my = Ber.mean(dim=0)

    # Plot the figure
    if x1 is None:
        x1 = torch.arange(0, 101, 10, device="cpu")
    k = 4
    axs[0].semilogy(np.arange(0, 110, 10), BER256_e2e[:, k], 'r--o', linewidth=1.7, label='256 - E2E Scheme')
    axs[0].semilogy(x1.cpu(), BER256_my[:, k].cpu(), 'b--^', linewidth=1.7, label='My Scheme')
    axs[0].grid(True)
    axs[0].axis([0, 100, 1e-6, 1])
    axs[0].set_xlabel('L (m)')
    axs[0].set_ylabel('BER')
    axs[0].legend()

    if x2 is None:
        x2 = torch.arange(-5, 21, 1, device="cpu")
    k = 2
    axs[1].semilogy(np.arange(-5, 21, 1), BER256_e2e[k, :], 'r--o', linewidth=1.7, label='256 - E2E Scheme')
    axs[1].semilogy(x2.cpu(), BER256_my[k, :].cpu(), 'b--^', linewidth=1.7, label='My Scheme')
    axs[1].grid(True)
    axs[1].axis([-5, 20, 1e-6, 1])
    axs[1].set_xlabel('SNR (dB)')
    axs[1].set_ylabel('BER')
    axs[1].legend()
    plt.tight_layout()
    # plt.pause(0.001)
    # plt.show()
    plt.savefig("test_fig.png", bbox_inches='tight', dpi=200, )
    plt.close()


if __name__ == "__main__":
    for _ in range(4):
        ber_plot(
            Ber=np.random.random([300, 6, 6]),
            x1=np.arange(0, 101, 20),
            x2=np.arange(-5, 21, 5)
        )
        # threading.Thread(target=ber_plot, args=(
        #     np.random.random([300, 6, 6]),
        #     np.arange(0, 101, 20),
        #     np.arange(-5, 21, 5)
        # )).start()
        print(f"Plotted {_}")
        time.sleep(3)
    # plt.show()
# print("")
