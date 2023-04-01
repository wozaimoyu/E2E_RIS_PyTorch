import torch


class Para:
    def __init__(
            self, Pos_User_x: torch.Tensor, SNR_train: torch.Tensor,
            Chan_BS2RIS: torch.Tensor, Chan_RIS2User: torch.Tensor, Chan_BS2User: torch.Tensor
    ):
        self.M = 16  # number of message symbols
        self.k = 4  # bits per symbol

        self.Batch_Size = 128  # 1024 * 8
        self.Num_train = max(self.Batch_Size, 10000)  # number of training samples
        self.Num_test = max(self.Batch_Size, 100000)  # number of test samples
        self.Num_vali = max(self.Batch_Size, 10000)
        self.Epoch_train = 60
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
        # self.Num_Subcarrier = 1
        # self.Power_Max = 1  # maximum power limit of BS
        self.Num_User = 1
        self.Num_User_Antenna = 2
        self.Num_RIS = self.Pos_RIS.shape[0]  # number of RIS
        self.Num_RIS_Element = 256  # number of elements in the RIS

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

    # todo: unused term LOS
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
