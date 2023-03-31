import random
import time

import torch
# import numpy as np
import scipy.io as sio
# from torch.autograd import Variable

import autoencoder as ae
import precoder
from channel_generator import rayleigh_chan
from plot_test import ber_plot
from sys_model import Para

use_cuda_if_available = True
device = "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu"
print(f"Using pytorch {torch.__version__} on {device}")
try:
    torch.set_default_device(device)
except AttributeError:
    print(f"Unable to set default device to {device}")

# Generate Channel Data
ChaData_BS2User = rayleigh_chan(Ta=8, Ra=2, L=1000)
ChaData_RIS2User = rayleigh_chan(Ta=1024, Ra=2, L=1000)
ChaData_BS2RIS = rayleigh_chan(Ta=8, Ra=1024, L=1000)

# ChaData_BS2User = torch.tensor(sio.loadmat('channel/ChaData_BS2User.mat')['Channel_BS2User'])
# ChaData_RIS2User = torch.tensor(sio.loadmat('channel/ChaData_RIS2User.mat')['Channel_RIS2User'])
# ChaData_BS2RIS = torch.tensor(sio.loadmat('channel/ChaData_BS2RIS.mat')['Channel_BS2RIS'])

# ChaData_BS2RIS.dtype = 'complex128'
# ChaData_RIS2User.dtype = 'complex128'
# ChaData_BS2User.dtype = 'complex128'

print('ChaData_BS2RIS:', ChaData_BS2RIS.shape)
print('ChaData_RIS2User:', ChaData_RIS2User.shape)
print('ChaData_BS2User:', ChaData_BS2User.shape)

SNR_dB = torch.arange(-5, 21, 1)  # Generate graph at these values
User_dis = torch.arange(0, 101, 10)
SNR_dB_train = torch.tensor([2, 1, 0, 0, -2, -5, -8, -6, 0, 2, 4])
# SNR_dB_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# SNR_dB_train = torch.tensor([2, 6, 7, 9, 10, 5, 0, 6, 13, 18, 20])

# How many training would be done, the output curve is mean of all training
Sample = 1

Ber = torch.zeros([Sample, User_dis.shape[0], SNR_dB.shape[0]])

# todo: figure out mean_rate and max_rate
mean_rate = torch.zeros([Sample, User_dis.shape[0]])
max_rate = torch.zeros([Sample, User_dis.shape[0]])

for s in range(Sample):
    Channel_BS2RIS = ChaData_BS2RIS[s]
    Channel_RIS2User = ChaData_RIS2User[s]
    Channel_BS2User = ChaData_BS2User[s]

    for x in range(User_dis.shape[0]):
        sys = Para(User_dis[x], SNR_dB_train[x], Channel_BS2RIS, Channel_RIS2User, Channel_BS2User)
        print(
            f'\n\nSample: {s}/{Sample - 1}, x: {x}/{User_dis.shape[0] - 1}, '
            f'position: {User_dis[x]}/{User_dis[-1]}, SNR {sys.SNR_train_db}'
        )

        X_train, Y_train = ae.generate_transmit_data(
            M=sys.M, J=sys.Num_User, num=sys.Num_train,
            seed=random.randint(0, 1000)
        )

        if x == 0:
            sys.Epoch_train = 200
            sys.LR_Factor = 1.05
            sys.load_model = 0

        time_start = time.time()
        precoder.train(X_train, Y_train, sys, sys.SNR_train)
        time_end = time.time()
        print(f'Time to train: {time_end - time_start}')

        # _, X_rate, Y_rate = ae.generate_rate_data(sys.M, sys.Num_User)
        # Y_pred, y_rate = precoder.test(X_rate, sys, torch.tensor(torch.inf))
        # mean_rate[s, x], max_rate[s, x] = torch.tensor(ae.calcul_rate(y_rate, sys.Rece_Ampli, sys.Num_User_Antenna))
        # print(f'The mean rate and max rate are {mean_rate[s, x]:0.8f}, {max_rate[s, x]:0.8f}, at x = {User_dis[x]} m')

        ber = torch.zeros(SNR_dB.shape)

        print("Calculating BER")
        for i_snr in range(SNR_dB.shape[0]):
            SNR = 10 ** (SNR_dB[i_snr] / 10) / sys.Rece_Ampli ** 2
            X_test, Y_test = ae.generate_transmit_data(
                M=sys.M, J=sys.Num_User, num=sys.Num_test,
                seed=random.randint(0, 1000)
            )
            Y_pred, y_receiver = precoder.test(X_test, sys, SNR)
            ber[i_snr] = ae.BER(X_test, sys, Y_pred, sys.Num_test)
            # print(f'The BER at SNR={SNR_dB[i_snr]} is {ber[i_snr]:0.8f}')
        print('-' * (5 + SNR_dB.shape[0] * 12))
        print(f'SNR | {"| ".join([f"{x:^10d}" for x in SNR_dB])}|')
        print(f'BER | {"| ".join([f"{x:0.8f}" for x in ber])}|')
        print('-' * (5 + SNR_dB.shape[0] * 12))

        Ber[s, x, :] = ber
        ber_plot(
            Ber=Ber[:s + 1, :, :],
            x1=User_dis,
            x2=SNR_dB
        )
        sio.savemat(f'figure/E2E_Ber_{sys.Num_RIS_Element}_{s}.mat', mdict={'Ber': Ber.cpu().numpy()})

    sio.savemat(f'figure/E2E_Ber_{sys.Num_RIS_Element}_{s}.mat', mdict={'Ber': Ber.cpu().numpy()})
    print("Saved temp ber!")
