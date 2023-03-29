import random
import time

import torch
import numpy as np
import scipy.io as sio
from torch.autograd import Variable

import autoencoder as ae
import precoder
from channel_generator import nakagami_chan, rayleigh_chan

use_cuda_if_available = True
device = "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu"
torch.set_default_device(device)
print(f"Using pytorch {torch.__version__} on {device}")


class Para:
    def __init__(self, Pos_User_x: int, SNR_train: int, Chan_BS2RIS, Chan_RIS2User, Chan_BS2User):
        self.M = 16  # number of message symbols
        self.k = 4  # bits per symbol

        self.Num_train = 1000  # number of training samples
        self.Num_test = 1000  # number of test samples
        self.Num_vali = 1000
        self.Epoch_train = 60
        self.Batch_Size = 160
        self.LR_Factor = 1.2  # learning rate factor
        self.load_model = 0  # whether to load pre-trained models or not

        # Set the positions of the BS, RIS, and user in the 2D plane
        self.Pos_BS = torch.tensor([[0, -40]], dtype=torch.float)
        self.Pos_RIS = torch.tensor([[60, 10]], dtype=torch.float)
        self.Pos_User = torch.zeros([1, 2], dtype=torch.float)  # [[0, 0]]
        self.Pos_User[0, 0] = torch.tensor(Pos_User_x)  # user is placed on the x-axis, y=0; [[x, 0]]

        # Set the number of BS, RIS, and user in the simulation
        self.Num_BS = self.Pos_BS.shape[0]
        self.Num_BS_Antenna = 8
        self.Num_Subcarrier = 1
        self.Power_Max = 1  # maximum power limit of BS
        self.Num_User = 1
        self.Num_User_Antenna = 2
        self.Num_RIS = self.Pos_RIS.shape[0]  # number of RIS
        self.Num_RIS_Element = 16  # number of elements in the RIS
        self.Phase_Matrix = Variable(  # initial phase matrix
            torch.ones([1, self.Num_RIS_Element * 2]) / np.sqrt(2),
            requires_grad=False
        )
        self.RIS_radio = 1  # fraction of active RIS elements
        # self.Active_Element = np.random.randint(
        #     self.Num_RIS_Element,
        #     size=int(self.RIS_radio * self.Num_RIS_Element)
        # )
        # self.Active_Element = np.hstack([
        #     self.Active_Element,
        #     self.Active_Element + int(self.Num_RIS_Element)
        # ])  # add a phase-shift for each active element
        # self.Active_Element = np.sort(self.Active_Element)  # sort the active elements

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
            Num_N=self.Num_BS, Nt=self.Num_BS_Antenna,
            Num_M=self.Num_RIS, Mr=self.Num_RIS_Element,
            Dis=self.Dis_BS2RIS, gain=2, LOS=1, fading=2
        )
        self.Channel_RIS2User = self.Channel_Matrix(
            Channel=Chan_RIS2User[0:self.Num_User_Antenna, 0:self.Num_RIS_Element],
            Num_N=self.Num_RIS, Nt=self.Num_RIS_Element,
            Num_M=self.Num_User, Mr=self.Num_User_Antenna,
            Dis=self.Dis_RIS2User, gain=2, LOS=0, fading=2
        )
        self.Channel_BS2User = self.Channel_Matrix(
            Channel=Chan_BS2User[0:self.Num_User_Antenna, 0:self.Num_BS_Antenna],
            Num_N=self.Num_BS, Nt=self.Num_BS_Antenna,
            Num_M=self.Num_User, Mr=self.Num_User_Antenna,
            Dis=self.Dis_BS2User, gain=3, LOS=0, fading=3
        )

        self.Rece_Ampli = 10 ** (-3.5)  # amplitude of the received signal
        self.SNR_train = 10 ** (SNR_train / 10)  # convert the SNR from dB to a ratio of P_S to P_N (linear scale)
        # ensures that the received signal power is consistent across different SNR values during training.
        self.SNR_train = self.SNR_train / (self.Rece_Ampli ** 2)  # normalizing the received signal amplitude to 1

    @staticmethod
    def Distance_Matrix(A, B):
        # NumofA, NumofB = A.shape[0], B.shape[0]
        # Dis = np.zeros([NumofA, NumofB])
        # # Compute the distance between each row of matrix A and all rows of matrix B
        # # and store the result in the Dis matrix
        # for i in range(NumofA):
        #     Dis[i, :] = np.linalg.norm(A[i, :] - B, axis=1)
        NumofA, NumofB = A.shape[0], B.shape[0]
        Dis = torch.zeros((NumofA, NumofB), dtype=A.dtype, device=A.device)
        # Compute the distance between each row of matrix A and all rows of matrix B
        # and store the result in the Dis matrix
        for i in range(NumofA):
            Dis[i, :] = torch.norm(A[i, :] - B, dim=1)
        return Dis

    @staticmethod
    def Channel_Matrix(Channel, Num_N, Nt, Num_M, Mr, Dis, gain, LOS, fading):
        # # Calculate path loss based on distance and fading parameters
        # Path_Loss = np.sqrt(10 ** (-fading) * Dis ** (- gain))
        # # Repeat path loss matrix to match the size of Channel
        # Path_Loss = Path_Loss.repeat(Mr, axis=0)
        # Path_Loss = Path_Loss.repeat(Nt, axis=1)
        # # Multiply path loss with the channel matrix to obtain the final channel
        # Channel = Path_Loss * Channel

        Path_Loss = torch.sqrt(10 ** (-fading) * Dis ** (- gain))
        Path_Loss = Path_Loss.repeat(Mr, 1)
        Path_Loss = Path_Loss.repeat(1, Nt)
        Channel = Path_Loss * Channel

        return Channel


# Generate Channel Data
# ChaData_BS2User = nakagami_chan(Ta=8, Ra=2, L=1000, m=5)
ChaData_BS2User = rayleigh_chan(Ta=8, Ra=2, L=1000)
# ChaData_RIS2User = nakagami_chan(Ta=256, Ra=2, L=1000, m=5)
ChaData_RIS2User = rayleigh_chan(Ta=256, Ra=2, L=1000)
# ChaData_BS2RIS = nakagami_chan(Ta=8, Ra=256, L=1000, m=5)
ChaData_BS2RIS = rayleigh_chan(Ta=8, Ra=256, L=1000)

# ChaData_BS2RIS.dtype = 'complex128'
# ChaData_RIS2User.dtype = 'complex128'
# ChaData_BS2User.dtype = 'complex128'
print('ChaData_BS2RIS:', ChaData_BS2RIS.shape)
print('ChaData_RIS2User:', ChaData_RIS2User.shape)
print('ChaData_BS2User:', ChaData_BS2User.shape)

# X_User = np.arange(0, 101, 10)
X_User = torch.arange(0, 101, 10)
# SNR_dB_train = np.array([2, 1, 0, 0, -2, -5, -8, -6, 0, 2, 4])
SNR_dB_train = torch.tensor([2, 1, 0, 0, -2, -5, -8, -6, 0, 2, 4])
# SNR_dB_train = np.array([2,6,7,9,10,5,0,6,13,18,20])

# SNR_dB = np.arange(-5, 21, 5)  # Generate graph at these values
SNR_dB = torch.arange(-5, 21, 5)  # Generate graph at these values

Sample = 1

Ber = torch.zeros([Sample, X_User.shape[0], SNR_dB.shape[0]])
mean_rate = torch.zeros([Sample, X_User.shape[0]])
max_rate = torch.zeros([Sample, X_User.shape[0]])

for s in range(Sample):
    Channel_BS2RIS = ChaData_BS2RIS[s]
    Channel_RIS2User = ChaData_RIS2User[s]
    Channel_BS2User = ChaData_BS2User[s]

    for x in range(X_User.shape[0]):
        sys = Para(X_User[x], SNR_dB_train[x], Channel_BS2RIS, Channel_RIS2User, Channel_BS2User)
        print(
            f'\n\nSample: {s}/{Sample - 1}, x: {x}/{X_User.shape[0] - 1}, position: {X_User[x]}/{X_User[-1]}, snr {sys.SNR_train}')
        _, X_train, Y_train = ae.generate_transmit_data(
            M=sys.M, J=sys.Num_User, num=sys.Num_train,
            seed=random.randint(0, 1000)
        )

        time_start = time.time()
        precoder.train(X_train, Y_train, sys, sys.SNR_train)
        time_end = time.time()
        print(f'Time to train: {time_end - time_start}')

        sym_index_rate, X_rate, Y_rate = ae.generate_rate_data(sys.M, sys.Num_User)
        Y_pred, y_rate = precoder.test(X_rate, sys, 10 ** 1000)
        mean_rate[s, x], max_rate[s, x] = torch.tensor(ae.calcul_rate(y_rate, sys.Rece_Ampli, sys.Num_User_Antenna))
        print(f'The mean rate and max rate are {mean_rate[s, x]:0.8f}, {max_rate[s, x]:0.8f}, at x = {X_User[x]} m')

        ber_wr = torch.zeros(SNR_dB.shape)
        ber = torch.zeros(SNR_dB.shape)

        for i_snr in range(SNR_dB.shape[0]):
            SNR = 10 ** (SNR_dB[i_snr] / 10) / sys.Rece_Ampli ** 2
            sym_index_test, X_test, Y_test = ae.generate_transmit_data(
                M=sys.M, J=sys.Num_User, num=sys.Num_test,
                seed=random.randint(0, 1000)
            )
            Y_pred, y_receiver = precoder.test(X_test, sys, SNR)
            ber[i_snr] = ae.BER(X_test, sys, Y_pred, sys.Num_test)
            print(f'The BER at SNR={SNR_dB[i_snr]} is {ber_wr[i_snr]:0.8f},  {ber[i_snr]:0.8f}')

        Ber[s, x, :] = ber

    sio.savemat(f'figure/E2E_Ber_{sys.Num_RIS_Element}_{s}.mat', mdict={'Ber': Ber})
    print("Saved temp ber!")
