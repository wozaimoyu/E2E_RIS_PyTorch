import random
import time

import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable

import autoencoder as ae
import precoder
from channel_generator import rayleigh_chan
from plot_test import ber_plot

use_cuda_if_available = True
device = "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu"
print(f"Using pytorch {torch.__version__} on {device}")
try:
    torch.set_default_device(device)
except AttributeError:
    print(f"Unable to set default device to {device}")


class Para:
    def __init__(self, Pos_User_x: torch.Tensor, SNR_train: int, Chan_BS2RIS, Chan_RIS2User, Chan_BS2User):
        self.M = 16  # number of message symbols
        self.k = 4  # bits per symbol

        self.Batch_Size = 128  # 1024 * 8
        self.Num_train = max(self.Batch_Size, 10000)  # number of training samples
        self.Num_test = max(self.Batch_Size, 100000)  # number of test samples
        self.Num_vali = max(self.Batch_Size, 10000)
        self.Epoch_train = 200
        self.LR_Factor = 1.2  # learning rate factor
        self.load_model = 1  # whether to load pre-trained models or not

        # Set the positions of the BS, RIS, and user in the 2D plane
        self.Pos_BS = torch.tensor([[0, -40]], dtype=torch.float)
        self.Pos_RIS = torch.tensor([[60, 10]], dtype=torch.float)
        self.Pos_User = torch.zeros([1, 2], dtype=torch.float)  # [[0, 0]]
        self.Pos_User[0, 0] = Pos_User_x  # user is placed on the x-axis, y=0; [[x, 0]]

        # Set the number of BS, RIS, and user in the simulation
        self.Num_BS = self.Pos_BS.shape[0]
        self.Num_BS_Antenna = 8
        self.Num_Subcarrier = 1
        self.Power_Max = 1  # maximum power limit of BS
        self.Num_User = 1
        self.Num_User_Antenna = 2
        self.Num_RIS = self.Pos_RIS.shape[0]  # number of RIS
        self.Num_RIS_Element = 256  # number of elements in the RIS
        self.Phase_Matrix = Variable(  # initial phase matrix
            torch.ones([1, self.Num_RIS_Element * 2]) / np.sqrt(2),
            requires_grad=False
        )
        self.RIS_radio = 1  # fraction of active RIS elements

        self.Active_Element = torch.randint(
            self.Num_RIS_Element,
            size=(int(self.RIS_radio * self.Num_RIS_Element),),
            dtype=torch.int64
        )
        self.Active_Element = torch.cat([
            self.Active_Element,
            self.Active_Element + int(self.Num_RIS_Element)
        ], dim=0)
        self.Active_Element, _ = torch.sort(self.Active_Element)

        # Calculate the distance matrix between the BS, RIS, and user
        self.Dis_BS2RIS = self.Distance_Matrix(self.Pos_BS, self.Pos_RIS)
        self.Dis_BS2User = self.Distance_Matrix(self.Pos_BS, self.Pos_User)
        self.Dis_RIS2User = self.Distance_Matrix(self.Pos_RIS, self.Pos_User)

        # Calculate the high-dimensional channel matrices
        self.Channel_BS2RIS = self.Channel_Matrix(
            Channel=Chan_BS2RIS[0:self.Num_RIS_Element, 0:self.Num_BS_Antenna],
            Nt=self.Num_BS_Antenna, Mr=self.Num_RIS_Element,
            Dis=self.Dis_BS2RIS, gain=2, LOS=1, fading=2
        )
        self.Channel_RIS2User = self.Channel_Matrix(
            Channel=Chan_RIS2User[0:self.Num_User_Antenna, 0:self.Num_RIS_Element],
            Nt=self.Num_RIS_Element, Mr=self.Num_User_Antenna,
            Dis=self.Dis_RIS2User, gain=2, LOS=0, fading=2
        )
        self.Channel_BS2User = self.Channel_Matrix(
            Channel=Chan_BS2User[0:self.Num_User_Antenna, 0:self.Num_BS_Antenna],
            Nt=self.Num_BS_Antenna, Mr=self.Num_User_Antenna,
            Dis=self.Dis_BS2User, gain=3, LOS=0, fading=3
        )

        self.Rece_Ampli = 10 ** (-3.5)  # amplitude of the received signal
        self.SNR_train = 10 ** (SNR_train / 10)  # convert the SNR from dB to a ratio of P_S to P_N (linear scale)
        # ensures that the received signal power is consistent across different SNR values during training.
        self.SNR_train = self.SNR_train / (self.Rece_Ampli ** 2)  # normalizing the received signal amplitude to 1

    @staticmethod
    def Distance_Matrix(A, B):
        NumofA, NumofB = A.shape[0], B.shape[0]
        Dis = torch.zeros((NumofA, NumofB), dtype=A.dtype, device=A.device)
        # Compute the distance between each row of matrix A and all rows of matrix B
        # and store the result in the Dis matrix
        for i in range(NumofA):
            Dis[i, :] = torch.norm(A[i, :] - B, dim=1)
        return Dis

    @staticmethod
    def Channel_Matrix(Channel, Nt, Mr, Dis, gain, LOS, fading):
        # Calculate path loss based on distance and fading parameters
        Path_Loss = torch.sqrt(10 ** (-fading) * Dis ** (- gain))
        # Repeat path loss matrix to match the size of Channel
        Path_Loss = Path_Loss.repeat(Mr, 1)
        Path_Loss = Path_Loss.repeat(1, Nt)
        # Multiply path loss with the channel matrix to obtain the final channel
        Channel = Path_Loss * Channel

        return Channel

    @property
    def SNR_train_db(self) -> torch.Tensor:
        return 10 * torch.log10(self.SNR_train * (self.Rece_Ampli ** 2))


# Generate Channel Data
# ChaData_BS2User = rayleigh_chan(Ta=8, Ra=2, L=1000)
# ChaData_RIS2User = rayleigh_chan(Ta=256, Ra=2, L=1000)
# ChaData_BS2RIS = rayleigh_chan(Ta=8, Ra=256, L=1000)

ChaData_BS2User = torch.tensor(sio.loadmat('channel/ChaData_BS2User.mat')['Channel_BS2User'])
ChaData_RIS2User = torch.tensor(sio.loadmat('channel/ChaData_RIS2User.mat')['Channel_RIS2User'])
ChaData_BS2RIS = torch.tensor(sio.loadmat('channel/ChaData_BS2RIS.mat')['Channel_BS2RIS'])

# ChaData_BS2RIS.dtype = 'complex128'
# ChaData_RIS2User.dtype = 'complex128'
# ChaData_BS2User.dtype = 'complex128'

print('ChaData_BS2RIS:', ChaData_BS2RIS.shape)
print('ChaData_RIS2User:', ChaData_RIS2User.shape)
print('ChaData_BS2User:', ChaData_BS2User.shape)

User_dis = torch.arange(0, 101, 10)
SNR_dB_train = torch.tensor([2, 1, 0, 0, -2, -5, -8, -6, 0, 2, 4])
# SNR_dB_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# SNR_dB_train = torch.tensor([2, 6, 7, 9, 10, 5, 0, 6, 13, 18, 20])

SNR_dB = torch.arange(-5, 21, 1)  # Generate graph at these values

Sample = 1

Ber = torch.zeros([Sample, User_dis.shape[0], SNR_dB.shape[0]])
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

        _, X_train, Y_train = ae.generate_transmit_data(
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
            _, X_test, Y_test = ae.generate_transmit_data(
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
