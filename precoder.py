import torch
from torch import nn
import torch.nn.functional as F

weight_gain = 1
bias_gain = 0.1


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
        nn.init.constant(self.map1.bias, bias_gain)
        nn.init.xavier_normal_(self.map2.weight, weight_gain)
        nn.init.constant(self.map2.bias, bias_gain)

    def forward(self, x):
        """
        Define forward pass of the neural network

        Args:
            x (tensor): Input tensor

        Returns:
            tensor: Output tensor
        """
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

