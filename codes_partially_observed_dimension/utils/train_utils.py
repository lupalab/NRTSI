import torch 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import random

def get_gap_lr_bs(dataset, epoch, init_lr, use_ta):
    if dataset == 'billiard':
        reset_best_loss = False
        save_ckpt =False
        #reset_best_loss_epoch = [600, 1200, 1800, 2650, 6700]
        reset_best_loss_epoch = [250, 450, 750, 1350, 6700]
        save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
        teacher_forcing = True
        if epoch < 250:
            if epoch < 200:
                min_gap, max_gap, lr, bs = 0, 1, init_lr, 64
            else:
                min_gap, max_gap, lr, bs = 0, 1, 0.1 * init_lr, 64
        elif epoch < 450:
            if epoch < 400:
                min_gap, max_gap, lr, bs = 1, 2, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 1, 2, 0.1 * init_lr, 128
        elif epoch < 750:
            if epoch < 700:
                min_gap, max_gap, lr, bs = 2, 4, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 2, 4, 0.1 * init_lr, 128
        elif epoch < 1350:
            if epoch < 1250:
                min_gap, max_gap, lr, bs = 4, 8, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 4, 8, 0.1 * init_lr, 128
        elif epoch < 6700:
            if epoch < 6300:
                min_gap, max_gap, lr, bs = 8, 16, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 8, 16, 0.1 * init_lr, 128
                teacher_forcing = use_ta
        #elif epoch < 8000:
        #    if epoch < 7900:
        #        min_gap, max_gap, lr, bs = 16, 32, init_lr, 256
        #    else:
        #        min_gap, max_gap, lr, bs = 16, 32, 0.1 * init_lr, 256
        if epoch in reset_best_loss_epoch:
            reset_best_loss = True
        if epoch in save_ckpt_epoch:
            save_ckpt = True
        if epoch < 6300:
            test_bs = bs
        else:
            test_bs = 10
    elif dataset == 'traffic':
        reset_best_loss = False
        save_ckpt =False
        reset_best_loss_epoch = [1000, 1800, 2200, 2700, 5200]
        save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
        teacher_forcing = True
        if epoch < 1000:
            if epoch < 900:
                min_gap, max_gap, lr, bs = 0, 1, init_lr, 16
            else:
                min_gap, max_gap, lr, bs = 0, 1, 0.1 * init_lr, 16
        elif epoch < 1800:
            if epoch < 1700:
                min_gap, max_gap, lr, bs = 1, 2, init_lr, 16
            else:
                min_gap, max_gap, lr, bs = 1, 2, 0.1 * init_lr, 16
        elif epoch < 2200:
            if epoch < 2100:
                min_gap, max_gap, lr, bs = 2, 4, init_lr, 16
            else:
                min_gap, max_gap, lr, bs = 2, 4, 0.1 * init_lr, 16
        elif epoch < 2700:
            if epoch < 2600:
                min_gap, max_gap, lr, bs = 4, 8, init_lr, 16
            else:
                min_gap, max_gap, lr, bs = 4, 8, 0.1 * init_lr, 16
        elif epoch < 5200:
            if epoch < 5100:
                min_gap, max_gap, lr, bs = 8, 16, init_lr, 16
            else:
                min_gap, max_gap, lr, bs = 8, 16, 0.1 * init_lr, 16
                teacher_forcing = use_ta
        #elif epoch < 8000:
        #    if epoch < 7900:
        #        min_gap, max_gap, lr, bs = 16, 32, init_lr, 256
        #    else:
        #        min_gap, max_gap, lr, bs = 16, 32, 0.1 * init_lr, 256
        if epoch in reset_best_loss_epoch:
            reset_best_loss = True
        if epoch in save_ckpt_epoch:
            save_ckpt = True
        if epoch < 5100:
            test_bs = bs
        else:
            test_bs = 10
    elif dataset == 'mujoco':
        reset_best_loss = False
        save_ckpt =False
        reset_best_loss_epoch = [700, 1400, 1700, 2700, 4500]
        save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
        teacher_forcing = True
        if epoch < 700:
            if epoch < 650:
                min_gap, max_gap, lr, bs = 0, 1, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 0, 1, 0.1 * init_lr, 128
        elif epoch < 1400:
            if epoch < 1350:
                min_gap, max_gap, lr, bs = 1, 2, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 1, 2, 0.1 * init_lr, 128
        elif epoch < 1700:
            if epoch < 1650:
                min_gap, max_gap, lr, bs = 2, 4, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 2, 4, 0.1 * init_lr, 128
        elif epoch < 2700:
            if epoch < 2650:
                min_gap, max_gap, lr, bs = 4, 8, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 4, 8, 0.1 * init_lr, 128
        elif epoch < 4500:
            if epoch < 4400:
                min_gap, max_gap, lr, bs = 8, 16, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 8, 16, 0.1 * init_lr, 128
                teacher_forcing = use_ta
        #elif epoch < 8000:
        #    if epoch < 7900:
        #        min_gap, max_gap, lr, bs = 16, 32, init_lr, 256
        #    else:
        #        min_gap, max_gap, lr, bs = 16, 32, 0.1 * init_lr, 256
        if epoch in reset_best_loss_epoch:
            reset_best_loss = True
        if epoch in save_ckpt_epoch:
            save_ckpt = True
        if epoch < 3100:
            test_bs = bs
        else:
            test_bs = 10
    elif dataset == 'gas':
        reset_best_loss = False
        save_ckpt =False
        reset_best_loss_epoch = [700, 1400, 1700, 2700, 4500]
        save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
        teacher_forcing = True
        if epoch < 700:
            if epoch < 650:
                min_gap, max_gap, lr, bs = 0, 1, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 0, 1, 0.1 * init_lr, 128
        elif epoch < 1400:
            if epoch < 1350:
                min_gap, max_gap, lr, bs = 1, 2, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 1, 2, 0.1 * init_lr, 128
        elif epoch < 1700:
            if epoch < 1650:
                min_gap, max_gap, lr, bs = 2, 4, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 2, 4, 0.1 * init_lr, 128
        elif epoch < 2700:
            if epoch < 2650:
                min_gap, max_gap, lr, bs = 4, 8, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 4, 8, 0.1 * init_lr, 128
        elif epoch < 4500:
            if epoch < 4400:
                min_gap, max_gap, lr, bs = 8, 16, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 8, 16, 0.1 * init_lr, 128
                teacher_forcing = use_ta
        #elif epoch < 8000:
        #    if epoch < 7900:
        #        min_gap, max_gap, lr, bs = 16, 32, init_lr, 256
        #    else:
        #        min_gap, max_gap, lr, bs = 16, 32, 0.1 * init_lr, 256
        if epoch in reset_best_loss_epoch:
            reset_best_loss = True
        if epoch in save_ckpt_epoch:
            save_ckpt = True
        if epoch < 3100:
            test_bs = bs
        else:
            test_bs = 10
    elif dataset == 'nfl':
        reset_best_loss = False
        save_ckpt =False
        reset_best_loss_epoch = [700, 1400, 1700, 2700, 3100]
        save_ckpt_epoch = [each - 1 for each in reset_best_loss_epoch]
        teacher_forcing = True
        if epoch < 700:
            if epoch < 650:
                min_gap, max_gap, lr, bs = 0, 1, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 0, 1, 0.1 * init_lr, 128
        elif epoch < 1400:
            if epoch < 1350:
                min_gap, max_gap, lr, bs = 1, 2, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 1, 2, 0.1 * init_lr, 128
        elif epoch < 1700:
            if epoch < 1650:
                min_gap, max_gap, lr, bs = 2, 4, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 2, 4, 0.1 * init_lr, 128
        elif epoch < 2700:
            if epoch < 2650:
                min_gap, max_gap, lr, bs = 4, 8, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 4, 8, 0.1 * init_lr, 128
        elif epoch < 3100:
            if epoch < 3050:
                min_gap, max_gap, lr, bs = 8, 16, init_lr, 128
            else:
                min_gap, max_gap, lr, bs = 8, 16, 0.1 * init_lr, 128
                teacher_forcing = use_ta
        #elif epoch < 8000:
        #    if epoch < 7900:
        #        min_gap, max_gap, lr, bs = 16, 32, init_lr, 256
        #    else:
        #        min_gap, max_gap, lr, bs = 16, 32, 0.1 * init_lr, 256
        if epoch in reset_best_loss_epoch:
            reset_best_loss = True
        if epoch in save_ckpt_epoch:
            save_ckpt = True
        if epoch < 3100:
            test_bs = bs
        else:
            test_bs = 10
    return max_gap, min_gap, lr, bs, reset_best_loss, save_ckpt, teacher_forcing, test_bs
    #gap = [0] +  [2 ** (i) for i in range(num_levels)]
    #epoch_per_level = total_epoch / num_levels
    #cur_level = int(epoch * num_levels / total_epoch)
    #max_gap = gap[cur_level+1]
    #min_gap = gap[cur_level]
    #lr = init_lr if epoch %  epoch_per_level < epoch_per_level * 9 / 10 else init_lr / 10
    #return max_gap, min_gap, lr


def get_num_missing(train, epoch, total_epoch, seq_len, dataset):
    if dataset == 'billiard':
        num_missing = 195
    elif dataset == 'traffic':
        num_missing = 140
    elif dataset == 'mujoco':
        num_missing = 90
    elif dataset == 'nfl':
        num_missing = random.randint(40, 49)
    elif dataset == 'gas':
        #num_missing = random.randint(91, 730)
        num_missing = 730
    elif dataset == 'air_quality':
        #num_missing = random.randint(53, 422)
        num_missing = 422
    return num_missing

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

def get_next_to_impute(data, mask, mode):
    target_data = data.clone()
    data_masked = data * torch.Tensor(mask[:, None, :]).cuda()
    seq_length = mask.shape[0]
    bs = data.shape[1]
    if mode in ['min', 'max']:
        num_obs_per_t = mask.sum(1)
        if mode == 'min':
            next_list = np.argwhere(num_obs_per_t == np.amin(num_obs_per_t))[:,0].tolist()
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

def run_epoch(epoch, total_epoch, train, model, mode, exp_data, clip, confidence, optimizer=None, 
            batch_size=64, fig_path=None, save_all_imgs=False, dataset='billiard'):
    if train:
        model.train()
    else:
        model.eval()
        batch_size = batch_size // 2
    losses = []
    inds = np.random.permutation(exp_data.shape[0])
    level_losses = np.zeros(exp_data.shape[1])
    level_losses_count = np.zeros(exp_data.shape[1]) + 1e-5
    i = 0
    d_data = exp_data.shape[-1]
    len_data = exp_data.shape[1]
    
    #max_gap, min_gap, cur_lr, batch_size = get_gap_lr_bs(epoch=epoch, total_epoch=total_epoch, num_levels=num_levels, init_lr=init_lr)
    #optimizer.param_groups[0]['lr'] = cur_lr
    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    while i + batch_size <= exp_data.shape[0]:
        
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
        data = data.cuda()
        ground_truth = data.clone()
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))
        ground_truth = ground_truth.squeeze().transpose(0, 1)
        num_missing = get_num_missing(train, epoch, total_epoch, data.shape[0], dataset)
        missing_idx = np.random.choice(np.arange(d_data*len_data), num_missing, replace=False)
        mask = np.ones(d_data*len_data)
        mask[missing_idx] = 0
        mask = mask.reshape(len_data, d_data)
        batch_loss = Variable(torch.tensor(0.0), requires_grad = True).cuda()
        batch_mse_loss = Variable(torch.tensor(0.0), requires_grad = True).cuda()
        num_levels = 1e-6

        while mask.sum() < mask.shape[0] * mask.shape[1]:
            obs_data, obs_list, next_data, next_list, target_data, mask = get_next_to_impute(data, mask, mode)
            prediction = model(obs_data, obs_list, next_data, next_list)
            if not confidence:
                level_loss = torch.mean((prediction - target_data).pow(2))
            else:
                level_loss = nll_gauss(target_data, prediction)
            batch_loss += level_loss
        if train:
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        losses.append(batch_loss.data.cpu().numpy())
      
    return np.mean(losses)

def nll_gauss(gt, pred, eps=1e-6):
    pred_mean = pred[:, :, :gt.shape[-1]]
    pred_std = F.softplus(pred[:, :, gt.shape[-1]:]) + eps
    normal_distri = torch.distributions.Normal(pred_mean,pred_std)
    LL = normal_distri.log_prob(gt)
    #nll = np.log(2*3.14) + 2 * torch.log(pred_std) + 0.5 * ((gt - pred_mean) / pred_std) ** 2
    NLL = - LL.sum(-1).mean()
    return NLL


def plot(epoch, fig_path, obs_data, target_data, ground_truth, prediction, gap, i=0, j=0):
    target_data = target_data.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    if prediction.shape[-1] == 3:
        prediction[:,:,-1] = F.softplus(prediction[:,:,-1])
    prediction = prediction.detach().cpu().numpy()
    obs_data = obs_data.detach().cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.scatter(target_data[:,j,0], target_data[:,j,1], color=colormap[0])
    plt.scatter(obs_data[j,:,0], obs_data[j,:,1], color=colormap[2])
    plt.scatter(prediction[j,:,0], prediction[j,:,1], color=colormap[1])
    if prediction.shape[-1] == 3:
        plt.scatter(prediction[j,:,0], prediction[j,:,1], marker='o', c='', s= 10**2 * prediction[j,:,2], edgecolor=colormap[1])
    plt.plot(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[3])
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def plot_nfl(epoch, fig_path, obs_data, target_data, ground_truth, prediction, gap, i=0, j=0):
    target_data = target_data.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    if prediction.shape[-1] == 3:
        prediction[:,:,-1] = F.softplus(prediction[:,:,-1])
    prediction = prediction.detach().cpu().numpy()
    obs_data = obs_data.detach().cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    for k in range(6):
        plt.scatter(target_data[:,j,k], target_data[:,j,k+6], color=colormap[0])
        plt.scatter(obs_data[j,:,k], obs_data[j,:,k+6], color=colormap[2])
        plt.scatter(prediction[j,:,k], prediction[j,:,k+6], color=colormap[1])
    if prediction.shape[-1] == 3:
        plt.scatter(prediction[j,:,0], prediction[j,:,1], marker='o', c='', s= 10**2 * prediction[j,:,2], edgecolor=colormap[1])
    for k in range(6):
        plt.plot(ground_truth[:,j, k], ground_truth[:,j,k+6], color=colormap[3])
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def plot_traffic(epoch, fig_path, obs_data, target_data, ground_truth, prediction, obs_list, next_list, gap, i=0, j=0):
    obs_list = obs_list.cpu().numpy()
    next_list = next_list.cpu().numpy()
    target_data = target_data.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    obs_data = obs_data.detach().cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.scatter(next_list[0,:,0], target_data[:,j,0], color=colormap[0])
    plt.scatter(obs_list[0,:,0], obs_data[j,:,0], color=colormap[2])
    plt.scatter(next_list[0,:,0], prediction[j,:,0], color=colormap[1])
    plt.plot(np.arange(ground_truth.shape[0]), ground_truth[:,j,0], color=colormap[3])
    #plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def identity_loss_schedule(epoch, start_weight=100.0, gamma=0.1, decay_epoch=50):
    if epoch < decay_epoch:
        return start_weight
    else:
        return gamma * start_weight
