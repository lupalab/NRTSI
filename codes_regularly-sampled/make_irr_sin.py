import numpy as np
import pdb
amplitude = 1.0
freq_range = (0.5, 1.0)
time_range = (0, 5)
time_multiplier = 20
seq_len = 100
N_train = 1000
N_test = 1000
train_data = np.zeros((N_train, seq_len, 2))
test_data = np.zeros((N_test, seq_len, 2))
for i in range(N_train):
    time_steps = time_range[1] * np.random.rand(seq_len)
    time_steps.sort()
    data = []
    t_prev = time_steps[0]
    freq = np.random.uniform(low=freq_range[0], high=freq_range[1])
    phi = 0.0
    for t in time_steps:
        dt = t - t_prev
        phi = phi + 2 * np.pi * freq * dt
        y = amplitude * np.sin(phi)
        t_prev = t
        data.append([y, t])
    data = np.array(data)
    train_data[i] = data

for i in range(N_test):
    time_steps = time_range[1] * np.random.rand(seq_len)
    time_steps.sort()
    data = []
    t_prev = time_steps[0]
    freq = np.random.uniform(low=freq_range[0], high=freq_range[1])
    phi = 0.0
    for t in time_steps:
        dt = t - t_prev
        phi = phi + 2 * np.pi * freq * dt
        y = amplitude * np.sin(phi)
        t_prev = t
        data.append([y, t])
    data = np.array(data)
    test_data[i] = data
train_data[:,:,1] = time_multiplier * train_data[:,:,1]
test_data[:,:,1] = time_multiplier * test_data[:,:,1]
np.save('./data/irr_sin_train.npy', train_data)
np.save('./data/irr_sin_test.npy', test_data)
