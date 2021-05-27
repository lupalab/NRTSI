import numpy as np
import torch
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import get_dataset
train_data, test_data = get_dataset('./data', 'traffic', 0)
train_data = train_data.transpose(0,1)
test_data = test_data.transpose(0,1)
for i in range(test_data.shape[1]):
    num_missing = np.random.randint(122, 141)
    missing_list_np = np.random.choice(np.arange(test_data.shape[0]), num_missing, replace=False)
    obs_list_np = sorted(list(set(np.arange(test_data.shape[0])) - set(missing_list_np)))
    obs_list = torch.from_numpy(np.array(obs_list_np)).long().cuda()
    obs_test_data = test_data[:, i, :][obs_list]
    obs_train_data = train_data[obs_list]
    dist = (obs_test_data[:, None, :] - obs_train_data).abs()
    dist = torch.sum(dist, [0,2])
    min_idx = torch.argmin(dist)
    nearest_neighbors = train_data[:, min_idx, :]
    plt.scatter(obs_list_np, obs_test_data[:, 100])
    plt.plot(np.arange(train_data.shape[0]), nearest_neighbors[:, 100])
    plt.savefig('%d.png' % i)
    plt.close()
