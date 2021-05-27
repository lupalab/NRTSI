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

def get_next_to_impute(irr_target_list, obs_list, max_level):
    min_dist_to_obs = torch.min((irr_target_list[None,:] - obs_list[:,None]).abs(), 0)[0]
    min_dist_to_obs[min_dist_to_obs > 2 ** max_level] = -1
    level = max_level
    while(level >= 0):
        if level == 0:
            next_idx = torch.where(min_dist_to_obs <= 2 ** level)
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
    next_list = irr_target_list[next_idx]
    print(min_dist_to_obs[next_idx])
    irr_target_list[next_idx] = -1
    irr_target_list = irr_target_list[irr_target_list != -1]
    return irr_target_list, next_list, gap

def run_imputation(model, exp_data, num_missing, ckpt_dict, max_level, confidence, batch_size=1, fig_path=None, save_all_imgs=False, dataset='billiard'):
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
        ind = torch.from_numpy(inds[i:i+batch_size]).long()
        i += batch_size
        data = exp_data[ind]
        data = data.cuda()
        ground_truth = data.clone()
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        ground_truth = ground_truth.transpose(0, 1)
        
        if dataset == 'billiard':
            num_missing = np.random.randint(180, 196)
        elif dataset == 'traffic':
            num_missing = np.random.randint(122, 141)
        missing_list_np = np.random.choice(np.arange(data.shape[0]), num_missing, replace=False)
        obs_list_np = sorted(list(set(np.arange(data.shape[0])) - set(missing_list_np)))
        init_obs_list = obs_list = torch.from_numpy(np.array(obs_list_np)).long().cuda()
        init_obs_data = obs_data = data[obs_list]
        imputation = None
        irr_target_list = (data.shape[0] - 1) * torch.rand(200).cuda()
        #irr_target_list = torch.tensor(missing_list_np).cuda()
        while irr_target_list.shape[0] > 0:
            
            irr_target_list, next_list, gap = get_next_to_impute(irr_target_list, obs_list, max_level)
            print(gap)
            model.load_state_dict(ckpt_dict[gap])
            
            obs_list_tiled = obs_list[None, :, None].repeat(batch_size,1,1) # [batch_size, seq_len, 1]
            obs_list = torch.cat([obs_list, next_list])
            next_list_tiled = next_list[None, :, None].repeat(batch_size,1,1) # [batch_size, seq_len, 1]
            # can our model impute irregularly sampeld points?
            #next_list_irr = next_list.double() + 0.5
            prediction = model(obs_data.transpose(0,1), obs_list_tiled, next_list_tiled, gap).detach()
            if confidence:
                prediction = prediction[:,:,:exp_data.shape[-1]]
            imputation = prediction.transpose(0,1) if imputation is None else torch.cat([imputation, prediction.transpose(0,1)], 0)
            obs_data = torch.cat([obs_data, prediction.transpose(0,1)], 0)
            
            
        if save_all_imgs:
            for j in range(batch_size):
                if dataset == 'billiard':
                    plot(0, fig_path, init_obs_data, imputation, ground_truth, gap, i, j)
                elif dataset == 'traffic':
                    plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap, i, j)
        else:
            if dataset == 'billiard':
                plot(0, fig_path, init_obs_data, imputation, ground_truth, gap)
            elif dataset == 'traffic':
                plot_traffic(0, fig_path, init_obs_data, imputation, ground_truth, init_obs_list, gap)
    if dataset == 'billiard':
        print('change of step size gt: %f, change of step size ours: %f' % (gt_total_change_of_step_size/exp_data.shape[0], total_change_of_step_size/exp_data.shape[0]))
    return loss / (exp_data.shape[0])

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

def plot_traffic(epoch, fig_path, obs_data, imputation, ground_truth, init_obs_list, gap, i=0, j=0):
    init_obs_list = init_obs_list.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    imputation = imputation.detach().cpu().numpy()
    obs_data = obs_data.cpu().numpy()
    colormap = ['b', 'r', 'g', 'm', 'y']
    plt.plot(np.arange(ground_truth.shape[0]), ground_truth[:,j,100], color=colormap[3])
    plt.plot(np.arange(ground_truth.shape[0]), imputation[:,j,100], color=colormap[1])
    plt.scatter(init_obs_list, obs_data[:,j,100], color=colormap[2])
    plt.savefig(os.path.join(fig_path, 'test_epoch_{%d}_{%d}_{%d}_{%d}.png' % (epoch, gap, i, j)))
    plt.close()


def identity_loss_schedule(epoch, start_weight=100.0, gamma=0.1, decay_epoch=50):
    if epoch < decay_epoch:
        return start_weight
    else:
        return gamma * start_weight
