import torch 
import numpy as np
from torch.autograd import Variable
from torch import nn

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

def get_next_to_impute(data, target_mask, obs_mask, max_level):
    data_time = data[:,:,-1]
    #obs_time = obs_mask * data_time
    #target_time = target_mask * data_time
    min_dist_to_obs = (data_time[None,:, :] - data_time[:,None, :]).abs()
    
    min_dist_to_obs[:, torch.where(target_mask>0)[0], torch.where(target_mask>0)[1]] = 1e7
    min_dist_to_obs = torch.min(min_dist_to_obs, 1)[0]
    min_dist_to_obs[torch.where(target_mask == 0)[0], torch.where(target_mask == 0)[1]] = -1

    # not consider gap size greater than max size
    min_dist_to_obs[min_dist_to_obs > 2 ** max_level] = -1
    # not consider masked values
    
    level = max_level
    
    while(level >= 0):
        if level == 0:
            next_idx = torch.where((min_dist_to_obs >=0) * (min_dist_to_obs <= 2 ** level))
            break
        else:
            gap = 2 ** level
            while(gap > 2 ** (level -1)):
                next_idx = torch.where((min_dist_to_obs <= gap) * (min_dist_to_obs > (gap - 1)))
                if next_idx[0].shape[0] != 0:
                    break
                else:
                    gap -= 1
            if next_idx[0].shape[0] == 0:
                level -= 1
            else:
                break
    gap = 2 ** level
    
    mask = torch.zeros_like(min_dist_to_obs)
    mask[next_idx] = 1.0
    max_next_len = mask.sum(0).max()
    max_obs_len = obs_mask.sum(0).max()
    
    obs_data = torch.zeros(data.shape[1], max_obs_len.int(), data.shape[-1])
    obs_data_mask = torch.zeros(data.shape[1], max_obs_len.int(), 1)
    next_data = torch.zeros(data.shape[1], max_next_len.int(), data.shape[-1])
    next_data_mask = torch.zeros(data.shape[1], max_next_len.int(), 1)

    obs_idx = torch.where(obs_mask > 0)
    valid_count = 0
    for i in range(data.shape[1]):
        batch_i_next_data = data[next_idx[0][torch.where(next_idx[1] == i)], i, :]
        if batch_i_next_data.shape[0] == 0:
            continue
        next_data[valid_count, :batch_i_next_data.shape[0], :] = batch_i_next_data
        next_data_mask[valid_count, :batch_i_next_data.shape[0], :] = 1.0
        batch_i_obs_data = data[obs_idx[0][torch.where(obs_idx[1] == i)], i, :]
        obs_data[valid_count, :batch_i_obs_data.shape[0], :] = batch_i_obs_data
        obs_data_mask[valid_count, :batch_i_obs_data.shape[0], :] = 1.0
        valid_count += 1
    next_data = next_data[:valid_count]
    next_data_mask = next_data_mask[:valid_count]
    obs_data = obs_data[:valid_count]
    obs_data_mask = obs_data_mask[:valid_count]

    next_data.to(data)
    obs_data.to(data)
    next_data_mask.to(data)
    obs_data_mask.to(data)
    
    # update observed and missing mask
    target_mask[next_idx] = 0
    obs_mask[next_idx] = 1
    
    return next_data, next_data_mask, obs_data, obs_data_mask, target_mask, obs_mask, gap, next_idx

def run_imputation(model, exp_data, num_missing, ckpt_dict, max_level, confidence, train_mean, test_mean, batch_size=1, fig_path=None, save_all_imgs=False, dataset='billiard'):
    model.eval()
    inds = np.arange(exp_data.shape[0])
    loss = []
    i = 0
    
    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    while i + batch_size <= exp_data.shape[0]:
        print(i)
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
        data = data.cuda()
        ground_truth = data.clone()
        imputation = data.clone()
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))
        ground_truth = ground_truth.squeeze().transpose(0, 1)
        imputation = imputation.squeeze().transpose(0, 1)
        if dataset == 'billiard_irr':
            num_missing = np.random.randint(180, 196)
        elif dataset == 'sin_irr':
            num_missing = 90
        missing_list_np = np.random.choice(np.arange(data.shape[0]), num_missing, replace=False)
        obs_list_np = sorted(list(set(np.arange(data.shape[0])) - set(missing_list_np)))
        init_obs_data = data[obs_list_np, :, :2]
        obs_mask = torch.zeros(data.shape[0], data.shape[1]).to(data)
        obs_mask[obs_list_np] = 1

        target_mask = 1.0 - obs_mask
        while obs_mask.sum() < torch.ones_like(obs_mask).sum():
            
            next_data, next_data_mask, obs_data, obs_data_mask, target_mask, obs_mask, gap, next_idx = get_next_to_impute(imputation, target_mask, obs_mask, max_level)
            #print(obs_mask.sum(), target_mask.sum(), gap)
            model.load_state_dict(ckpt_dict[gap])
            
            prediction = model(obs_data.cuda(), obs_data_mask.cuda(), next_data.cuda(), next_data_mask.cuda(), gap)
            
            target_data = next_data[:,:,:-1]
            # replace gt data in imputation with pred 
            valid_count = 0
            for j in range(data.shape[1]):
                if torch.where(next_idx[1] == j)[0].shape[0] == 0:
                    continue
                imputation[next_idx[0][torch.where(next_idx[1] == j)], j, :data.shape[-1]-1] = prediction[valid_count, :int(next_data_mask[valid_count].sum())]
                valid_count += 1
            
        loss.append((torch.sum((imputation - ground_truth).pow(2)) / (batch_size * num_missing)).item())
        #print(loss)
        if save_all_imgs:
            for j in range(batch_size):
                if dataset == 'billiard_irr':
                    plot(0, fig_path, init_obs_data, imputation, ground_truth, gap, i=i, j=j)
                elif dataset == 'sin_irr':
                    plot_sin(0, fig_path, init_obs_data, imputation, ground_truth, gap, i=i, j=j)
        else:
            if dataset == 'billiard_irr':
                plot(0, fig_path, init_obs_data, imputation, ground_truth, gap, i=i, j=0)
            elif dataset == 'sin_irr':
                plot_sin(0, fig_path, init_obs_data, imputation, ground_truth, gap, i=i, j=0)
    return np.mean(loss)
                
          
       

def plot(epoch, fig_path, obs_data, imputation, ground_truth, gap, i=0, j=0):
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.scatter(ground_truth[:,j, 0], ground_truth[:,j,1], color=colormap[0])
    plt.scatter(imputation[:,j,0], imputation[:,j,1], color=colormap[1])
    plt.scatter(obs_data[:,j,0], obs_data[:,j,1], color=colormap[2])
    plt.xlim(-1.05, 1.05)
    plt.ylim(-1.05, 1.05)
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()

def plot_sin(epoch, fig_path, obs_data, imputation, ground_truth, gap, i=0, j=0):
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.scatter(ground_truth[:,j, 1], ground_truth[:,j,0], color=colormap[0])
    plt.scatter(imputation[:,j,1], imputation[:,j,0], color=colormap[1])
    plt.scatter(obs_data[:,j,1], obs_data[:,j,0], color=colormap[2])
    plt.xlim(-0.05, 100.05)
    plt.ylim(-0.05, 1.05)
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
