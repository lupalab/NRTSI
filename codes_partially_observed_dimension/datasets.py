import torch
import os
import pickle
import pdb
import numpy as np
def get_dataset(root_dir, name, normalize=True):
    if name == 'basketball':
        test_data = torch.Tensor(pickle.load(open(os.path.join(root_dir, 'basketball_eval.p'), 'rb'))).transpose(0, 1)[:, :-1, :]
        train_data = torch.Tensor(pickle.load(open(os.path.join(root_dir, 'basketball_train.p'), 'rb'))).transpose(0, 1)[:, :-1, :]
    elif name == 'billiard':
        test_data = torch.Tensor(pickle.load(open(os.path.join(root_dir, 'billiard_eval.p'), 'rb'), encoding='latin1'))[:, :, :]
        train_data = torch.Tensor(pickle.load(open(os.path.join(root_dir, 'billiard_train.p'), 'rb'), encoding='latin1'))[:, :, :]
    elif name == 'traffic':
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'pems', 'pems_test.npy')))
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'pems', 'pems_train.npy')))
    elif name == 'mujoco':
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'mujoco_test.npy')))
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'mujoco_train.npy')))
    elif name == 'nfl':
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'nfl_train.npy')))
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'nfl_test.npy')))
    elif name == 'gas':
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'gas_train.npy')))
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'gas_test.npy')))
        val_data = torch.Tensor(np.load(os.path.join(root_dir, 'gas_val.npy')))
    elif name == 'air_quality':
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'air_quality_train.npy')))
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'air_quality_test.npy')))
        val_data = torch.Tensor(np.load(os.path.join(root_dir, 'air_quality_val.npy')))
    else:
        print('no such task')
        exit()
    if normalize:
        test_data -= torch.min(test_data, dim=1, keepdim=True)[0]
        test_data /= torch.max(test_data, dim=1, keepdim=True)[0]
        test_data = 2.0 * (test_data - 0.5)
        train_data -= torch.min(train_data, dim=1, keepdim=True)[0]
        train_data /= torch.max(train_data, dim=1, keepdim=True)[0]
        train_data = 2.0 * (train_data - 0.5)
    print(test_data.shape, train_data.shape)
    return train_data, val_data, test_data