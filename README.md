# E2E_RIS_PyTorch
 End-to-End Reconfigurable Intelligent Surface implemented in PyTorch.

This project is a PyTorch implementation of a MIMO Communication System that involves multiple-input multiple-output (MIMO) communication channels with multiple antennas at the transmitter and receiver side. The system can be trained and evaluated to predict the optimal way of transmitting data over a noisy wireless channel in a multi-user scenario.

The code simulates communication between a base station (BS) and multiple users over a noisy channel. The system employs a radio frequency (RF) phase-shifting technique using a reconfigurable intelligent surface (RIS) to mitigate the channel's multipath effect.

The system's training process involves generating training data and using it to train the system to predict the optimal transmission and reception of data at different signal-to-noise ratio (SNR) levels. The output is a graph showing the bit-error-rate (BER) for different SNR values.

Libraries Used:

- PyTorch
- NumPy
- SciPy
