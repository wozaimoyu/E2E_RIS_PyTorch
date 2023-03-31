import random
from pathlib import Path

import numpy as np
# import scipy.io as sio
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm.auto import tqdm

import autoencoder as ae
import sys_model

t_learning_rate = 0.001
r_learning_rate = 0.001
ris_learning_rate = 0.001

print_interval = 30
# g_weight_gain = 0.1
# g_bias_gain = 0.1

optim_betas = (0.9, 0.999)
weight_gain = 1
bias_gain = 0.1
weight_decay = 0.0001


class BS(nn.Module):
    """
    Base station (BS) class, which defines the neural network
    architecture of the BS.

    Args:
        input_size (int): Size of input layer
        hidden_size (int): Size of hidden layer
        output_size (int): Size of output layer
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(BS, self).__init__()

        # Define layers in the neural network
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

        # Initialize weights and biases using Xavier initialization and constant initialization, respectively
        nn.init.xavier_normal_(self.map1.weight, weight_gain)
        nn.init.constant_(self.map1.bias, bias_gain)
        nn.init.xavier_normal_(self.map2.weight, weight_gain)
        nn.init.constant_(self.map2.bias, bias_gain)

    def forward(self, x):
        return self.map3(F.relu(self.map2(F.relu(self.map1(x)))))


class RIS(nn.Module):
    """
    Reconfigurable intelligent surface (RIS) class, which defines the
    phase shifting operation of the RIS.

    Args:
        output_size (int): Size of output tensor
    """

    def __init__(self, output_size: int):
        super(RIS, self).__init__()

        # Define phase shifting operation using an embedding layer
        # nn.Embedding: A simple lookup table that stores embeddings of a fixed dictionary and size. This module is
        # often used to store word embeddings and retrieve them using indices. The input to the module is a list of
        # indices, and the output is the corresponding word embeddings.
        self.phase = nn.Embedding(num_embeddings=1, embedding_dim=output_size)

    def forward(self, x):
        """
        Define forward pass of the RIS phase shifting operation.
        Apply a phase shift to the input signal in a way that optimizes the received signal quality at the receiver.
        The calculation in the forward method uses the embedding layer `self.phase` to obtain a phase shift vector p1,
        which is then normalized and reshaped to create a complex coefficient vector x3. The input signal x is then
        multiplied with x3 to apply the phase shift, and the resulting signal x4 is returned. The purpose of the
        multiplication is to steer the signal towards the receiver in a way that maximizes the received signal
        strength while minimizing the interference caused by the reflected signals.

        Args:
            x (tensor): Input tensor

        Returns:
            tensor: Output tensor after phase shifting
        """
        m, n = x.shape
        norm = torch.empty(1, int(n / 2))  # empty tensor with shape (1, int(n / 2)), store the L2 norm
        x4 = torch.empty(m, n)  # store the transformed input tensor

        p1 = self.phase(torch.tensor(0))  # embedding for the value 0
        p1 = torch.reshape(p1, [2, int(n / 2)])  # reshape the embedding tensor

        norm[0, :] = torch.norm(p1, 2, 0)  # L2 norm of the reshaped embedding tensor along the 0th dimension
        p_max = torch.max(norm[0, :])  # maximum value of norm

        p2 = p1 / p_max  # normalizes the embedding tensor
        x3 = torch.reshape(p2, [1, n])  # reshapes p2 back to a 1D tensor of length n
        x3 = x3.expand(m, n)  # expands x3 along the 0th dimension

        # compute the transformed input tensor
        x4[:, 0:int(n / 2)] = x3[:, 0:int(n / 2)] * x[:, 0:int(n / 2)] - x3[:, int(n / 2):n] * x[:, int(n / 2):n]
        x4[:, int(n / 2):n] = x3[:, 0:int(n / 2)] * x[:, int(n / 2):n] + x3[:, int(n / 2):n] * x[:, 0:int(n / 2)]
        return x4


class Receiver(nn.Module):
    """
    Receiver class, which defines the neural network architecture of the receiver.

    Args:
        input_size (int): Size of input layer
        hidden_size (int): Size of hidden layer
        output_size (int): Size of output layer
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(Receiver, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bn2 = nn.BatchNorm1d(output_size)

    def forward(self, x):
        return torch.sigmoid(self.bn2(self.map2(self.bn1(F.relu(self.map1(x))))))


def ini_weights(sys):
    B = BS(
        input_size=sys.k * sys.Num_User,
        hidden_size=4 * sys.M,
        output_size=sys.Num_BS_Antenna * 2
    )
    Ris = RIS(
        # input_size=int(sys.Num_RIS_Element * sys.RIS_radio) * 2,
        # hidden_size=4 * sys.M,
        output_size=sys.Num_RIS_Element * 2
    )
    R = []
    for i in range(sys.Num_User):
        R.append(
            Receiver(
                input_size=sys.Num_User_Antenna * 2,
                hidden_size=2 * sys.M,
                output_size=sys.k
            )
        )
    return B, Ris, R


def Channel(t_data, channel):
    """
    this code performs a channel operation on a given input data tensor "t_data" using a complex-valued channel
    "channel". The operation involves transposing the data tensor, extracting the real and imaginary components of
    the channel, concatenating them to form two new tensors "h1" and "h2", concatenating those tensors along a new
    axis, performing matrix multiplication between the resulting channel tensor and the transposed data tensor,
    and finally transposing the output back to its original shape.
    """
    cat = torch.cat
    t_data = torch.transpose(t_data, 1, 0)
    hr, hi = channel.real, channel.imag
    h1, h2 = cat([hr, -hi], 1), cat([hi, hr], 1)
    channel = cat([h1, h2], 0).float()
    r_data = torch.matmul(channel, t_data)
    return torch.transpose(r_data, 1, 0)


# c1 = nn.MSELoss()
criterion = nn.BCELoss()


def train(X, Y, sys: sys_model.Para, SNR_train: torch.Tensor):
    X_train = X
    Y_train = Y
    k = sys.k
    Num_User = sys.Num_User
    lr_factor = sys.LR_Factor
    total_batch = int(X.shape[0] / sys.Batch_Size)  # num_total/batch_size
    print(
        f"Training Started..\n"
        f"\tk: {sys.k}, M: {sys.M} Num User: {sys.Num_User}, LR Factor: {sys.LR_Factor}, SNR: {sys.SNR_train_db:.3f}\n"
        f"\tBS: {sys.Pos_BS.tolist()}, RIS: {sys.Pos_RIS.tolist()}, User: {sys.Pos_User.tolist()}\n"
        f"\tEpoch: {sys.Epoch_train}, Batch Size: {sys.Batch_Size}, Total Batch: {total_batch}\n"
        f"\tBS Antenna: {sys.Num_BS_Antenna}, RIS Element: {sys.Num_RIS_Element}, User Antenna: {sys.Num_BS_Antenna}\n"
        f"\tBS2RIS Dis: {sys.Dis_BS2RIS.squeeze().tolist()}\n"
        f"\tBS2User Dis: {sys.Dis_BS2User.squeeze().tolist()}\n"
        f"\tRIS2User Dis: {sys.Dis_RIS2User.squeeze().tolist()}\n"
    )

    if total_batch <= 0:
        raise ValueError(f"Number of Batch can not be 0 or less")

    B, Ris, R = ini_weights(sys=sys)
    if sys.load_model == 1:
        B, Ris, R = Load_Model(B, Ris, R, Num_User)

    b_optimizer = optim.Adam(
        params=B.parameters(),
        lr=t_learning_rate,
        betas=optim_betas
    )
    ris_optimizer = optim.Adam(
        params=Ris.parameters(),
        lr=ris_learning_rate,
        betas=optim_betas
    )
    r_optimizer = []
    for i in range(Num_User):
        r_optimizer.append(optim.Adam(
            params=R[i].parameters(),
            lr=r_learning_rate,
            betas=optim_betas
        ))

    if weight_decay > 0:
        # todo: What is regularization?
        print("Regularization skipped")
        # b_reg = Regularization(B, weight_decay, p=2)
        # ris_reg = Regularization(Ris, weight_decay, p=2)
        # r_reg = []
        # for i in range(Num_User):
        #     r_reg.append(Regularization(R[i], weight_decay, p=2))
    else:
        print("no regularization")

    R_error, Acc, best_ber = [], [], 1
    with tqdm(
            total=sys.Epoch_train, unit='epoch',
            bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
            # "ETA: {eta}"
    ) as pbar:  # progress bar
        for epoch in range(sys.Epoch_train):
            error_epoch = 0
            pbar.set_description(f"Epoch {epoch}")
            for index in range(total_batch):
                if type(SNR_train) is not torch.Tensor:
                    # SNR_train = torch.tensor(SNR_train)
                    raise TypeError("Not Tensor!")
                noise_train = torch.randn(
                    sys.Batch_Size,
                    sys.Num_User_Antenna * 2
                ) / torch.sqrt(SNR_train) / np.sqrt(2)

                idx = np.random.randint(X.shape[0], size=sys.Batch_Size)

                B.zero_grad()
                Ris.zero_grad()
                for i in range(Num_User):
                    R[i].zero_grad()

                x_data = B(ae.onehot2bit(X_train[idx, :]))
                target = ae.onehot2bit(Y_train[idx, :])
                norm = torch.empty(1, sys.Batch_Size)
                norm[0, :] = torch.norm(x_data, 2, 1)

                x_data = x_data / torch.t(norm)
                ri_data = Channel(x_data, sys.Channel_BS2RIS)
                ro_data = Ris(ri_data)
                y_data = Channel(
                    ro_data, sys.Channel_RIS2User
                ) + Channel(
                    x_data, sys.Channel_BS2User
                ) + noise_train

                error_user = 0
                for i in range(Num_User):
                    r_error = criterion(
                        R[i](y_data),
                        target[:, i * k:(i + 1) * k]
                    )
                    r_error.backward(retain_graph=True)

                    r_optimizer[i].step()
                    ris_optimizer.step()
                    b_optimizer.step()
                    error_user += r_error
                error_epoch = error_epoch + error_user / Num_User

                pbar.update(1 / total_batch)
                # pbar.set_description(f"Epoch {epoch} ({index + 1:3d}/{total_batch:3d})")
            R_error.append((error_user / Num_User).detach())

            # ---------------------------------------------------------------------------------------------------------

            if epoch % print_interval == 0:
                print(
                    f"\r \nEpoch: {epoch}, Loss: {(error_user / Num_User).data}, "
                    f"LR: {b_optimizer.param_groups[0]['lr']:0.5f}"
                )
                for i in range(Num_User):
                    r_optimizer[i].param_groups[0]['lr'] /= lr_factor
                ris_optimizer.param_groups[0]['lr'] /= lr_factor
                b_optimizer.param_groups[0]['lr'] /= lr_factor

                SNR_vali = torch.tensor([-5, 0, 5, 10, 15, 20])
                ber = torch.zeros(SNR_vali.shape)
                for i_snr in range(SNR_vali.shape[0]):
                    SNR = 10 ** (SNR_vali[i_snr] / 10) / sys.Rece_Ampli ** 2
                    X_test, Y_test = ae.generate_transmit_data(
                        sys.M, Num_User, sys.Num_vali, seed=random.randint(0, 1000)
                    )
                    Y_pred, y_receiver = test(X_test, sys, SNR, B, Ris, R)
                    ber[i_snr] = ae.BER(X_test, sys, Y_pred, sys.Num_vali)
                    # print(f'The BER at SNR={SNR_vali[i_snr]} is {ber[i_snr]:0.8f}')
                # print('-----------------------------------------------------------------------------')
                print(f'SNR | {"| ".join([f"{x:^10d}" for x in SNR_vali])}|')
                print(f'BER | {"| ".join([f"{x:0.8f}" for x in ber])}|')
                # print('-----------------------------------------------------------------------------')
                Acc.append(ber[3])  # BER at 10dB
                if ber[1] < best_ber:  # BER at 0dB
                    Save_Model(B, Ris, R, Num_User)
                    best_ber = ber[1]  # BER at 0dB
                power = torch.mean(torch.sum(
                    (Channel(ro_data, sys.Channel_RIS2User) + Channel(x_data, sys.Channel_BS2User)) ** 2, 1
                ))
                # SNR_train = ((2.5 * 1e-6) / power / (10 ** 0.5 * 1e-7))
                SNR_train = 10 ** 0.5 * 2.5 / power.data
                print(f"\rSNR_train changed to {SNR_train} or "
                      f"{10 * torch.log10(SNR_train * ((10 ** (-3.5)) ** 2))} db?"
                      )
            # pbar.update(1)
    print("")
    return R_error


def test(X, sys, SNR_test: torch.Tensor, B=None, Ris=None, R=None):
    """
    Simulates the transmission of a given signal over a wireless communication system and returns the predicted received signal and the actual received signal.

    Args:
        X (numpy.ndarray): The transmitted signal, represented as a one-hot-encoded array of integers.
        sys (): The communication system object that contains the channel models and other system parameters.
        SNR_test (float): The signal-to-noise ratio used to add Gaussian noise to the received signal.
        B (torch.nn.Module, optional): The preprocessing function used to transform the input signal. If None, it will be initialized using the "ini_weights" function. Default is None.
        Ris (torch.nn.Module, optional): The function that models the RIS channel. If None, it will be initialized using the "ini_weights" function. Default is None.
        R (list of torch.nn.Module, optional): The list of functions that model the user channels. If None, it will be initialized using the "ini_weights" function. Default is None.
        device (str, optional): The device used for computing. Default is "cpu".

    Returns:
        tuple: A tuple containing the predicted rate for each user based on the received signal and the "R" matrices,
               and the actual received signal.

    """
    k = sys.k
    Num_User = sys.Num_User

    if B is None:
        B, Ris, R = ini_weights(sys)
        B, Ris, R = Load_Model(B, Ris, R, Num_User)

    X_test = X
    num_test = X_test.shape[0]
    if type(SNR_test) is not torch.Tensor:
        raise TypeError
    noise_test = torch.randn(num_test, sys.Num_User_Antenna * 2) * torch.sqrt(1 / (2 * SNR_test))

    x_data = B(ae.onehot2bit(X_test))
    norm = torch.empty(1, num_test)
    norm[0, :] = torch.norm(x_data, 2, 1)
    x_data = x_data / torch.t(norm)

    # print("\tSaving x_data")
    # sio.savemat(f'x_data_{sys.Num_RIS_Element}.mat', mdict={'x_data': x_data.cpu().detach().numpy()})

    ri_data = Channel(x_data, sys.Channel_BS2RIS)
    ro_data = Ris(ri_data)
    y_data = Channel(
        ro_data, sys.Channel_RIS2User
    ) + Channel(
        x_data, sys.Channel_BS2User
    ) + noise_test

    r_decision = []
    for i in range(Num_User):
        r_decision.append(R[i](y_data))
    r_decision = torch.reshape(
        torch.stack(r_decision, 1),
        [num_test, k * Num_User]
    )

    pred = r_decision.detach()
    y_rate = y_data.detach()

    return pred, y_rate


def Save_Model(B, Ris, R, J):
    print("Saving models..")
    Path("outputs/model").mkdir(parents=True, exist_ok=True)
    torch.save(B.state_dict(), "outputs/model/b1.pkl")
    torch.save(Ris.state_dict(), "outputs/model/ris1.pkl")
    for i in range(J):
        torch.save(R[i].state_dict(), f"outputs/model/r1_{i}.pkl")


def Load_Model(B, Ris, R, J):
    # todo: test parameter removed,,, is it needed?
    B.load_state_dict(torch.load('outputs/model/b1.pkl'))
    Ris.load_state_dict(torch.load('outputs/model/ris1.pkl'))
    for i in range(J):
        R[i].load_state_dict(torch.load(f'outputs/model/r1_{i}.pkl'))
    return B, Ris, R


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    @staticmethod
    def get_weight(model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    @staticmethod
    def regularization_loss(weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
