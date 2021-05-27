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

def run_imputation(model, exp_data, num_missing, ckpt_dict, max_level, confidence, train_mean, test_mean, batch_size=32, fig_path=None, save_all_imgs=False, dataset='billiard', gp=0):
    model.eval()
    #inds = np.random.permutation(exp_data.shape[0])
    inds = np.arange(exp_data.shape[0])
    i = 0
    loss = 0

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
        imputation = 10 * torch.ones_like(ground_truth)
        if dataset == 'billiard':
            num_missing = np.random.randint(180, 196)
        elif dataset == 'traffic':
            num_missing = np.random.randint(122, 141)
        elif dataset == 'mujoco':
            num_missing = 90
        elif dataset == 'nfl':
            num_missing = 45
        missing_list_np = np.random.choice(np.arange(data.shape[0]), num_missing, replace=False)
        obs_list_np = sorted(list(set(np.arange(data.shape[0])) - set(missing_list_np)))
        init_obs_list = obs_list = torch.from_numpy(np.array(obs_list_np)).long().cuda()
        init_obs_data = obs_data = data[obs_list]
        imputation[init_obs_list, :, :] = ground_truth[init_obs_list, :, :]
        #for j in range(batch_size):
        #    if dataset == 'billiard':
        #        plot(0, fig_path, init_obs_data, imputation, ground_truth, -1, i, j)
        while len(obs_list_np) < data.shape[0]:
            next_list_np, gap = get_next_to_impute(data.shape[0], obs_list_np, max_level, gp)
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
            prediction, att = model(obs_data.transpose(0,1), obs_list, next_list, gap, return_attns=True)
            prediction = prediction.detach()
            att = [each.detach() for each in att]
            
            
            if confidence:
                pred_mean = prediction[:, :, :ground_truth.shape[-1]]
                pred_std = F.softplus(prediction[:, :, ground_truth.shape[-1]:]) + 1e-6
                noise = torch.zeros_like(pred_mean)
                prediction = noise * pred_std + pred_mean
            imputation[next_list_np] = prediction.transpose(0,1)
            #for j in range(batch_size):
            #    if dataset == 'billiard':
            #        plot(0, fig_path, obs_data, imputation, ground_truth, gap, i, j, att, next_list, obs_list)
            obs_data = torch.cat([obs_data, prediction.transpose(0,1)], 0)
        loss += torch.sum((imputation - ground_truth).pow(2)) / num_missing
        if dataset == 'billiard':
            step_size = np.sqrt(np.square(imputation[:, :, ::2].cpu().numpy()) + np.square(imputation[:, :, 1::2].cpu().numpy()))
            change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
            total_change_of_step_size += change_of_step_size.std()
            step_size = np.sqrt(np.square(ground_truth[:, :, ::2].cpu().numpy()) + np.square(ground_truth[:, :, 1::2].cpu().numpy()))
            change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
            gt_total_change_of_step_size += change_of_step_size.std()
        if save_all_imgs:
            for j in range(batch_size):
                if dataset == 'billiard':
                    plot(0, fig_path, init_obs_data, imputation, ground_truth, 32, i, j)
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
    if dataset == 'billiard':
        print('change of step size gt: %f, change of step size ours: %f' % (gt_total_change_of_step_size/exp_data.shape[0], total_change_of_step_size/exp_data.shape[0]))
    return loss / i

def plot(epoch, fig_path, obs_data, imputation, ground_truth, gap, i=0, j=0, att=None, next_list=None, obs_list=None):
    
        
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    #plt.figure(figsize=(4,4))
    #plt.plot(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[3])
    #plt.scatter(imputation[:,j,0], imputation[:,j,1], color=colormap[1])
    #plt.scatter(obs_data[:,j,0], obs_data[:,j,1], color=colormap[2])
    if att is not None and gap == 11:
        for k in reversed(range(att[0].shape[-1]-1)):
            if att[0][0,0,-1,k] != 0:
                k += 1
                break
        G_2_S_att = [each[:,:,k:,:k].detach().cpu().numpy() for each in att]
        obs_list = obs_list[0,:,0]
        next_list = next_list[0,:,0]
        for layer in range(8):
            for head in range(12):
                plt.figure(figsize=(4,4))
                candidates = []
                for g in range(G_2_S_att[0].shape[2]):
                    for s in range(G_2_S_att[0].shape[3]):
                        candidates.append(G_2_S_att[layer][j, head, g, s])
                candidates_min = min(candidates)
                candidates_max = max(candidates - candidates_min)
                for g in range(G_2_S_att[0].shape[2]):
                    for s in range(G_2_S_att[0].shape[3]):
                        gx, sx, gy, sy =  imputation[next_list[g], j, 0], imputation[obs_list[s], j, 0] , imputation[next_list[g], j, 1], imputation[obs_list[s], j, 1]
                        print(gx, sx, gy, sy, G_2_S_att[layer][j, head, g, s] )
                        att_weight = G_2_S_att[layer][j, head, g, s]
                        rescaled_att_weight = (att_weight - candidates_min) / candidates_max
                        plt.plot([gx, sx], [gy, sy], linewidth=3 * rescaled_att_weight, alpha=min(0.5*rescaled_att_weight, 0.9), color='b')
                plt.plot(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[3])
                plt.scatter(imputation[:,j,0], imputation[:,j,1], color=colormap[1])
                plt.scatter(obs_data[:,j,0], obs_data[:,j,1], color=colormap[2]) 
                plt.xlim(-0.9, 0.9)
                plt.ylim(-0.9, 0.9)
                plt.xticks([])
                plt.yticks([])

                plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
                plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j, layer, head)), dpi=300)
                plt.close()
        return
    
    plt.figure(figsize=(4,4))
    plt.plot(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[3])
    plt.scatter(imputation[:,j,0], imputation[:,j,1], color=colormap[1])
    plt.scatter(obs_data[:,j,0], obs_data[:,j,1], color=colormap[2])
                
    plt.xlim(-0.9, 0.9)
    plt.ylim(-0.9, 0.9)
    plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)                

    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.pdf' % (epoch, gap, i, j)))
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
