import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import random
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

def gap_to_max_gap(gap):
    i = 0
    while(gap > 2 ** i):
        i += 1
    return 2 ** i

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = np.linalg.inv(K)
    
    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)



def compute_gp_uncertainty(obs_list, next_list, l=20):
    X_train = np.array(obs_list).reshape(-1, 1)
    Y_train = np.zeros_like(X_train)
    X = np.array(next_list).reshape(-1, 1)
    mu_s, cov_s = posterior(X, X_train, Y_train, l, 1)
    uncertainty = np.diag(np.abs(cov_s))
    min_uncertainty = np.min(uncertainty)
    idx = np.where(uncertainty < 2 * min_uncertainty)[0].tolist()
    return [next_list[each] for each in idx]

def get_next_to_impute(data, mask, mode, certain=None):
    target_data = data.clone()
    data_masked = data * torch.Tensor(mask[:, None, :]).cuda()
    seq_length = mask.shape[0]
    bs = data.shape[1]
    if mode in ['min', 'max', 'certain']:
        num_obs_per_t = mask.sum(1)
        if mode == 'min':
            next_list = np.argwhere(num_obs_per_t == np.amin(num_obs_per_t))[:,0].tolist()
        elif mode == 'certain':
            next_list = np.argwhere(certain == np.amax(certain))[:,0].tolist()
            certain[next_list] = -1
        else:
            num_obs_per_t[num_obs_per_t==mask.shape[1]] = -1
            next_list = np.argwhere(num_obs_per_t == np.amax(num_obs_per_t))[:,0].tolist()
        obs_list = list(set([i for i in range(seq_length)]) - set(next_list))
        obs_mask = np.tile(np.expand_dims(mask[obs_list, :], 1), (1, data.shape[1], 1))
        obs_data = torch.cat([data_masked[obs_list, :, :], torch.Tensor(obs_mask).cuda()], -1)
        next_mask = np.tile(np.expand_dims(mask[next_list, :], 1), (1, data.shape[1], 1))
        next_data = torch.cat([data_masked[next_list, :, :], torch.Tensor(next_mask).cuda()], -1)
        target_data = target_data[next_list, :, :]
        mask[next_list] = np.ones_like(mask[next_list])

        next_list = torch.Tensor(next_list).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
        obs_list = torch.Tensor(obs_list).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
        obs_data = obs_data.transpose(0,1)
        next_data = next_data.transpose(0, 1)
        target_data = target_data.transpose(0, 1)
    else:
        obs_data = None
        obs_list = None
        next_data = torch.cat([data_masked, torch.Tensor(mask).unsqueeze(1).repeat(1, bs, 1).cuda()], -1)
        next_list = torch.arange(seq_length).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
        next_data = next_data.transpose(0, 1)
        target_data = target_data.transpose(0, 1)
        mask = np.ones_like(mask)
    return obs_data, obs_list, next_data, next_list, target_data, mask

def run_imputation(main_model, certainty_model, mode, exp_data, num_missing, max_level, confidence, train_mean, test_mean, batch_size=128, fig_path=None, save_all_imgs=False, dataset='billiard', gp=0):
    main_model.eval()
    certainty_model.eval()
    #inds = np.random.permutation(exp_data.shape[0])
    inds = np.arange(exp_data.shape[0])
    i = 0
    loss = 0
    d_data = exp_data.shape[-1]
    len_data = exp_data.shape[1]

    if dataset == 'billiard':
        total_change_of_step_size = 0
        gt_total_change_of_step_size = 0
    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    while i + batch_size <= exp_data.shape[0]:
        print(i)
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
        if dataset == 'nfl': # randomly permute player order for nfl
            num_players = int(data.shape[-1] / 2)
            rand_idx = np.arange(num_players)
            random.shuffle(rand_idx)
            rand_idx = np.concatenate([rand_idx, rand_idx+num_players])
            data = data[:,:,rand_idx]
        data = data.cuda()
        ground_truth = data.clone()
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        ground_truth = ground_truth.transpose(0, 1)
        imputation = ground_truth.clone()
        if dataset == 'billiard':
            num_missing = np.random.randint(180, 196)
        elif dataset == 'traffic':
            num_missing = np.random.randint(122, 141)
        elif dataset == 'mujoco':
            num_missing = 90
        elif dataset == 'nfl':
            num_missing = 45
        elif dataset == 'gas':
            num_missing = 730
        elif dataset == 'air_quality':
            num_missing = 53
        #num_missing = get_num_missing(train, epoch, total_epoch, data.shape[0], dataset)
        missing_idx = np.random.choice(np.arange(d_data*len_data), num_missing, replace=False)
        mask = np.ones(d_data*len_data)
        mask[missing_idx] = 0
        mask = mask.reshape(len_data, d_data)
        init_mask = mask

        obs_data, obs_list, next_data, next_list, target_data, mask = get_next_to_impute(data, mask, "all")
        certainty = certainty_model(obs_data, obs_list, next_data, next_list)
        
        certainty = F.softplus(certainty[:,:,data.shape[-1]:]).detach()
        certainty = certainty * (1 - torch.Tensor(init_mask[None,:,:]).cuda())
        certainty = torch.sum(certainty,[0,2]) / (torch.Tensor(1 - init_mask).cuda().sum(1) + 1e-6)
        certainty = certainty.cpu().numpy()
        
        mask = init_mask
        while mask.sum() < mask.shape[0] * mask.shape[1]:
            obs_data, obs_list, next_data, next_list, target_data, mask = get_next_to_impute(data, mask, "certain", certainty)
            prediction = main_model(obs_data, obs_list, next_data, next_list)
            prediction = prediction.detach()
            pred_mask = next_data[:,:,int(next_data.shape[-1]/2):]
            masked_pred = pred_mask * target_data + (1 - pred_mask) * prediction
            next_idx = next_list[0,:,0].long()
            data[next_idx, :, :] = masked_pred.transpose(0,1)
        loss += torch.sum((data - ground_truth).pow(2)) / num_missing
        #print(loss / i)
        if save_all_imgs:
            for j in range(batch_size):
                if dataset == 'billiard':
                    plot(0, fig_path, init_obs_data, imputation, ground_truth, gap, i, j)
                elif dataset == 'traffic':
                    plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap, train_mean, test_mean, i, j)
                elif dataset == 'nfl':
                    plot_nfl(0, fig_path, init_obs_data, imputation, ground_truth, gap, i, j)
        else:
            if dataset == 'billiard':
                plot(0, fig_path, init_obs_data, imputation, ground_truth, gap)
            elif dataset == 'traffic':
                plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap, train_mean, test_mean)
            elif dataset == 'nfl':
                    plot_nfl(0, fig_path, init_obs_data, imputation, ground_truth, gap)
    return loss / i

def plot(epoch, fig_path, obs_data, imputation, ground_truth, gap, i=0, j=0):
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.plot(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[3])
    plt.scatter(imputation[:,j,0], imputation[:,j,1], color=colormap[1])
    plt.scatter(obs_data[:,j,0], obs_data[:,j,1], color=colormap[2])
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def plot_nfl(epoch, fig_path, obs_data, imputation, ground_truth, gap, i=0, j=0):
    imputation = imputation.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    obs_data = obs_data.detach().cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    for k in range(6):
        plt.scatter(imputation[:,j,k], imputation[:,j,k+6], color=colormap[0])
        plt.scatter(obs_data[:,j,k], obs_data[:,j,k+6], color=colormap[2])

    for k in range(6):
        plt.plot(ground_truth[:,j, k], ground_truth[:,j,k+6], color=colormap[3])
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def plot_traffic(epoch, fig_path, obs_data, imputation, ground_truth, init_obs_list, gap, train_mean, test_mean, i=0, j=0):
    init_obs_list = init_obs_list.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.plot(np.arange(ground_truth.shape[0]), ground_truth[:,j,100], color=colormap[3])
    plt.plot(np.arange(ground_truth.shape[0]), imputation[:,j,100], color=colormap[1])
    plt.scatter(init_obs_list, obs_data[:,j,100], color=colormap[2])
    plt.plot(np.arange(ground_truth.shape[0]), train_mean[:, 100], color=colormap[0])
    plt.plot(np.arange(ground_truth.shape[0]), test_mean[:, 100], color=colormap[-1])
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()


def identity_loss_schedule(epoch, start_weight=100.0, gamma=0.1, decay_epoch=50):
    if epoch < decay_epoch:
        return start_weight
    else:
        return gamma * start_weight
