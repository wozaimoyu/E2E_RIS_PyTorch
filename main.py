import math
import random
import shutil
import time
from pathlib import Path

import torch
# import numpy as np
import scipy.io as sio
# from torch.autograd import Variable

import autoencoder as ae
import precoder
import channel_generator as cg
from logging_config import get_logger
from plot_test import ber_plot
from sys_model import Para

try:
    from google.colab import files

    COLAB = True
except ModuleNotFoundError:
    COLAB = False

logger = get_logger(__name__)

use_cuda_if_available = True
device = "cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu"
try:
    torch.set_default_device(device)
except AttributeError:
    logger.debug(f"Unable to set default device to {device}")
    device = "cpu"
logger.info(f"Using pytorch {torch.__version__} on {device}")

# Generate Channel Data
channel_fn = cg.nakagami_chan
ChaData_BS2User = channel_fn(Ta=8, Ra=2, L=1000)
ChaData_RIS2User = channel_fn(Ta=1024, Ra=2, L=1000)
ChaData_BS2RIS = channel_fn(Ta=8, Ra=1024, L=1000)

# ChaData_BS2User = torch.tensor(sio.loadmat('channel/ChaData_BS2User.mat')['Channel_BS2User'])
# ChaData_RIS2User = torch.tensor(sio.loadmat('channel/ChaData_RIS2User.mat')['Channel_RIS2User'])
# ChaData_BS2RIS = torch.tensor(sio.loadmat('channel/ChaData_BS2RIS.mat')['Channel_BS2RIS'])

# ChaData_BS2RIS.dtype = 'complex128'
# ChaData_RIS2User.dtype = 'complex128'
# ChaData_BS2User.dtype = 'complex128'

logger.info(
    f'ChaData_BS2RIS: {tuple(ChaData_BS2RIS.shape)}'
    f'ChaData_RIS2User: {tuple(ChaData_RIS2User.shape)}'
    f'ChaData_BS2User: {tuple(ChaData_BS2User.shape)}\n'
)

SNR_dB = torch.arange(-5, 21, 1)  # Generate graph at these values
User_dis = torch.arange(0, 101, 10)
SNR_dB_train = torch.tensor([2, 1, 0, 0, -2, -5, -8, -6, 0, 2, 4])
# SNR_dB_train = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# SNR_dB_train = torch.tensor([2, 6, 7, 9, 10, 5, 0, 6, 13, 18, 20])

logger.debug(f"User Distances: {User_dis.tolist()}")
logger.debug(f"Training SNRs : {SNR_dB_train.tolist()}")

# How many training would be done, the output curve is mean of all training
Sample = 1

Ber = torch.zeros([Sample, User_dis.shape[0], SNR_dB.shape[0]])

# todo: figure out mean_rate and max_rate
mean_rate = torch.zeros([Sample, User_dis.shape[0]])
max_rate = torch.zeros([Sample, User_dis.shape[0]])
Path(f'outputs/tmp').mkdir(parents=True, exist_ok=True)
sys = None
for s in range(Sample):
    Channel_BS2RIS = ChaData_BS2RIS[s]
    Channel_RIS2User = ChaData_RIS2User[s]
    Channel_BS2User = ChaData_BS2User[s]

    for x in range(User_dis.shape[0]):
        sys = Para(User_dis[x], SNR_dB_train[x], Channel_BS2RIS, Channel_RIS2User, Channel_BS2User)
        logger.info(
            f'\n\nSample: {s}/{Sample - 1}, x: {x}/{User_dis.shape[0] - 1}, '
            f'position: {User_dis[x]}/{User_dis[-1]}, SNR {sys.SNR_train_db}'
        )

        X_train, Y_train = ae.generate_transmit_data(
            M=sys.M, J=sys.Num_User, num=sys.Num_train,
            seed=random.randint(0, 1000)
        )

        if x == 0:
            logger.debug("Changing Model Data")
            sys.Epoch_train = 200
            sys.LR_Factor = 1.05
            sys.load_model = 0

        time_start = time.time()
        precoder.train(X_train, Y_train, sys, sys.SNR_train)
        time_end = time.time()
        logger.info(f'Time to train: {time_end - time_start}')

        # _, X_rate, Y_rate = ae.generate_rate_data(sys.M, sys.Num_User)
        # Y_pred, y_rate = precoder.test(X_rate, sys, torch.tensor(torch.inf))
        # mean_rate[s, x], max_rate[s, x] = torch.tensor(ae.calcul_rate(y_rate, sys.Rece_Ampli, sys.Num_User_Antenna))
        # print(f'The mean rate and max rate are {mean_rate[s, x]:0.8f}, {max_rate[s, x]:0.8f}, at x = {User_dis[x]} m')

        ber = torch.zeros(SNR_dB.shape)

        logger.info("Calculating BER")
        B, Ris, R = precoder.ini_weights(sys)
        B, Ris, R = precoder.Load_Model(B, Ris, R, sys.Num_User)
        for i_snr in range(SNR_dB.shape[0]):
            print(f"\r{SNR_dB[i_snr]}", end="")
            SNR = 10 ** (SNR_dB[i_snr] / 10) / sys.Rece_Ampli ** 2
            X_test, Y_test = ae.generate_transmit_data(
                M=sys.M, J=sys.Num_User, num=sys.Num_test,
                seed=random.randint(0, 1000)
            )
            Y_pred, y_receiver = precoder.test(X_test, sys, SNR, B, Ris, R)
            ber[i_snr] = ae.BER(X_test, sys, Y_pred, sys.Num_test)
            # print(f'The BER at SNR={SNR_dB[i_snr]} is {ber[i_snr]:0.8f}')
        print('\r', end="")
        line = ""
        print_divider = '-' * (5 + 6 * 12) + '\n'
        line += print_divider
        for i in range(math.ceil(len(SNR_dB) / 6)):
            line += f'SNR | {"| ".join([f"{x:^10d}" for x in SNR_dB[i * 6:(i + 1) * 6]])}|\n'
            line += f'BER | {"| ".join([f"{x:0.8f}" for x in ber[i * 6:(i + 1) * 6]])}|\n'
            line += print_divider
        logger.info(line)

        Ber[s, x, :] = ber
        ber_plot(
            Ber=Ber[:s + 1, :, :],
            fname=Path(f"outputs/tmp/{time.strftime('%Y-%m-%d %H.%M.%S')}.png"),
            x1=User_dis,
            x2=SNR_dB
        )
        sio.savemat(
            f'outputs/tmp/E2E_Ber_{sys.Num_RIS_Element}_{s}.mat',
            mdict={'Ber': Ber[:s + 1, :, :].cpu().numpy()}
        )

ber_plot(
    Ber=Ber,
    fname=Path(f"outputs/E2E_Ber_{sys.Num_RIS_Element}.png"),
    x1=User_dis,
    x2=SNR_dB
)
sio.savemat(f'outputs/E2E_Ber_{sys.Num_RIS_Element}.mat', mdict={'Ber': Ber.cpu().numpy()})
print("Saved ber!")

Path("zip").mkdir(parents=True, exist_ok=True)
zip_name = f"zip/outputs_{time.strftime('%Y-%m-%d %H.%M.%S')}"
shutil.make_archive(
    base_name=zip_name,  # zip file name
    format="zip",
    root_dir="outputs"  # folder to zip
)
if COLAB:
    files.download(f"{zip_name}.zip")
