import torch
import os
import pickle
import pdb
import numpy as np
def get_dataset(root_dir, name, normalize=True):
    if name == 'billiard_irr':
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'irr_billiard_test.npy')))
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'irr_billiard_train_large.npy')))
    elif name == 'sin_irr':
        test_data = torch.Tensor(np.load(os.path.join(root_dir, 'irr_sin_test.npy')))
        train_data = torch.Tensor(np.load(os.path.join(root_dir, 'irr_sin_train.npy')))
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
    return train_data, test_data
