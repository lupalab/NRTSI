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

def get_next_to_impute(seq_len, obs_list, max_level, gp_uncertrain):
    
    min_dist_to_obs = np.zeros(seq_len)
    for i in range(seq_len):
        if i not in obs_list:
            min_dist = np.abs((np.array(obs_list) - i)).min()
            if min_dist <= 2 ** max_level:
                min_dist_to_obs[i] = min_dist
    next_idx = np.argwhere(min_dist_to_obs == np.amax(min_dist_to_obs))[:,0].tolist()
    
    gap = np.amax(min_dist_to_obs)
    if gp_uncertrain and gap == 2 ** max_level:
        next_idx = compute_gp_uncertainty(obs_list, next_idx)
    return next_idx, gap

def run_imputation(model, exp_data, num_missing, ckpt_dict, max_level, confidence, train_mean, test_mean, n_sample=10, batch_size=64, n_mix=1, fig_path=None, save_all_imgs=False, dataset='billiard', gp=0):
    model.eval()
    #inds = np.random.permutation(exp_data.shape[0])
    inds = np.arange(exp_data.shape[0])
    i = 0
    loss = 0
    avg_loss = 0
    count = 0
    if dataset == 'billiard' or 'nfl':
        total_change_of_step_size = 0
        gt_total_change_of_step_size = 0
        path_length = 0
        gt_path_length = 0

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
        
        if dataset == 'billiard':
            num_missing = np.random.randint(180, 196)
        elif dataset == 'traffic':
            num_missing = np.random.randint(122, 141)
        elif dataset == 'mujoco':
            num_missing = 90
        elif dataset == 'nfl':
            num_missing = np.random.randint(40, 49)
        missing_list_np = np.random.choice(np.arange(data.shape[0]), num_missing, replace=False)
        min_mse = 1e5 * np.ones(batch_size)
        avg_mse = np.zeros(batch_size)
        for d in range(n_sample):

            imputation = ground_truth.clone()
            obs_list_np = sorted(list(set(np.arange(data.shape[0])) - set(missing_list_np)))
            init_obs_list = obs_list = torch.from_numpy(np.array(obs_list_np)).long().cuda()
            init_obs_data = obs_data = data[obs_list]
            while len(obs_list_np) < data.shape[0]:
                next_list_np, gap = get_next_to_impute(data.shape[0], obs_list_np, max_level, gp)
                if gap > 2 ** 2:
                    next_list_np = [next_list_np[0]]
                #if confidence and gap ==2**max_level:
                #    next_list_np = next_list_np[:1]
                max_gap = gap_to_max_gap(gap)
                model.load_state_dict(ckpt_dict[max_gap])
                obs_list = torch.from_numpy(np.array(obs_list_np)).long().cuda()
                obs_list_np += next_list_np
                
                obs_list = obs_list[None, :, None].repeat(batch_size,1,1) # [batch_size, seq_len, 1]
                next_list = torch.from_numpy(np.array(next_list_np)).long().cuda()
                target_data = ground_truth[next_list].transpose(0,1)
                next_list = next_list[None, :, None].repeat(batch_size,1,1) # [batch_size, seq_len, 1]
                # can our model impute irregularly sampeld points?
                #next_list_irr = next_list.double() + 0.5
                prediction = model(obs_data.transpose(0,1), obs_list, next_list, gap).detach()
                if n_mix == 1:
                    samples = sample_gauss(prediction, ground_truth, n_mix, gap)
                else:
                    samples = sample_gmm(prediction, ground_truth, n_mix, gap)
                imputation[next_list_np] = samples.transpose(0,1)
                obs_data = torch.cat([obs_data, samples.transpose(0,1)], 0)
            #loss += torch.sum((imputation - ground_truth).pow(2)) / num_missing
            if dataset == 'billiard':
                step_size = np.sqrt(np.square(imputation[:, :, ::2].cpu().numpy()) + np.square(imputation[:, :, 1::2].cpu().numpy()))
                change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
                total_change_of_step_size += change_of_step_size.std()
                step_size = np.sqrt(np.square(ground_truth[:, :, ::2].cpu().numpy()) + np.square(ground_truth[:, :, 1::2].cpu().numpy()))
                change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
                gt_total_change_of_step_size += change_of_step_size.std()
            if dataset == 'nfl':
                step_size = (imputation[1:, :, :].cpu().numpy() - imputation[:-1, :, :].cpu().numpy()) ** 2
                step_size = np.sqrt(step_size[:,:,:6] + step_size[:,:,6:])
                total_change_of_step_size += step_size.std(0).mean()
                path_length += step_size.sum(0).mean()
                step_size = (ground_truth[1:, :, :].cpu().numpy() - ground_truth[:-1, :, :].cpu().numpy()) ** 2
                step_size = np.sqrt(step_size[:,:,:6] + step_size[:,:,6:])
                gt_total_change_of_step_size += step_size.std(0).mean()
                gt_path_length += step_size.sum(0).mean()
                
                mse = (torch.sum((imputation - ground_truth).pow(2), [0,2]) / num_missing).cpu().numpy()
                avg_mse += mse
                min_mse[mse < min_mse] = mse[mse < min_mse]
            if save_all_imgs:
                for j in range(batch_size):
                    if dataset == 'billiard':
                        plot(0, fig_path, init_obs_data, imputation, ground_truth, gap, i, j)
                    elif dataset == 'traffic':
                        plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap, train_mean, test_mean, i, j)
                    elif dataset == 'nfl':
                        plot_nfl(0, fig_path, init_obs_data, imputation, ground_truth, gap, d, n_sample, i, j)
            else:
                if dataset == 'billiard':
                    plot(0, fig_path, init_obs_data, imputation, ground_truth, gap)
                elif dataset == 'traffic':
                    plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap, train_mean, test_mean)
                elif dataset == 'nfl':
                    plot_nfl(0, fig_path, init_obs_data, imputation, ground_truth, gap, d, n_sample)
        count += 1
        loss += min_mse.mean()
        avg_loss += avg_mse.mean() / n_sample
    loss /= count
    avg_loss /= count
    gt_total_change_of_step_size /= n_sample*count
    total_change_of_step_size /= n_sample*count
    gt_path_length /= n_sample*count
    path_length /= n_sample*count


    return loss, avg_loss, gt_total_change_of_step_size, total_change_of_step_size, gt_path_length, path_length

def sample_gauss(pred, gt, n_mix, gap, eps=1e-6):
    pred_mean = pred[:, :, :gt.shape[-1]]
    
    pred_std = F.softplus(pred[:, :, gt.shape[-1]:]) + eps
    if gap <= 2 ** 2:
        pred_std = 1e-5 * pred_std
    normal_distri = torch.distributions.Normal(pred_mean,pred_std)
    return normal_distri.sample()

def sample_gmm(pred, gt, n_mix, gap, eps=1e-6):
    mix = torch.distributions.Categorical(F.softplus(pred[:,:,-n_mix:]))
    pred_mean = pred[:, :, :gt.shape[-1]*n_mix]
    pred_mean = pred_mean.view(pred.shape[0], pred.shape[1], n_mix, gt.shape[-1])
    pred_std = F.softplus(pred[:, :, n_mix*gt.shape[-1]:2*n_mix*gt.shape[-1]]) + eps
    pred_std = pred_std.view(pred.shape[0], pred.shape[1], n_mix, gt.shape[-1])
    normal_distri = torch.distributions.Independent(torch.distributions.Normal(pred_mean,pred_std), 1)
    gmm = torch.distributions.MixtureSameFamily(mix, normal_distri)
    return gmm.sample()

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

def plot_nfl(epoch, fig_path, obs_data, imputation, ground_truth, gap, d, n_sample, i=0, j=0):
    imputation = imputation.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    obs_data = obs_data.detach().cpu().numpy()
    plt.figure(j, figsize=(4,4))
    plt.xticks([])
    plt.yticks([])
    
    colormap = ['b', 'r' , 'm', 'brown', 'lime', 'orage', 'gold', 'indigo', 'slategrey', 'y', 'g']
    for k in range(6):
        plt.plot(imputation[:,j,k], imputation[:,j,k+6], color=colormap[0], alpha=0.5, label='imputation')
        plt.scatter(obs_data[:,j,k], obs_data[:,j,k+6], color=colormap[-1], label='observation')

    for k in range(6):
        plt.plot(ground_truth[:,j, k], ground_truth[:,j,k+6], color=colormap[1], label='ground truth')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    if d == (n_sample - 1):
        plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.pdf' % (epoch, gap, i, j)))
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
