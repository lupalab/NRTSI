import os
import sys
import pdb
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import torch
import numpy as np
from pprint import pformat, pprint

from datasets import get_dataset
from utils.hparams import HParams
from utils.test_utils import run_imputation
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformers import Encoder
#from exnode import ExnodeEncoder

import torch.nn as nn
import torch.optim as optim
import time

ckpt_path_dict = dict()

ckpt_root_dir = './log/per_gap_8_layer_12_heads_no_dropout_unnormalize_max_level_4_mse_0.975_miss_att_d_128_model_d_1024/ckpt'
ckpt_path_dict[1] = os.path.join(ckpt_root_dir, 'best_model_gap_1.pt')
ckpt_path_dict[2] = os.path.join(ckpt_root_dir, 'best_model_gap_2.pt')
ckpt_path_dict[4] = os.path.join(ckpt_root_dir, 'best_model_gap_4.pt')
ckpt_path_dict[8] = os.path.join(ckpt_root_dir, 'best_model_gap_8.pt')
ckpt_path_dict[16] = os.path.join(ckpt_root_dir, 'best_model_gap_16.pt')
ckpt_dict =dict()
for key in ckpt_path_dict:
    ckpt_dict[key] = torch.load(ckpt_path_dict[key])


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
parser.add_argument('--num_missing', type=int)
parser.add_argument('--save_fig', type=int)
args = parser.parse_args()
params = HParams(args.cfg_file)
pprint(params.dict)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


# creat exp dir
if not os.path.exists(params.exp_dir):
    os.mkdir(params.exp_dir)
if not os.path.exists(os.path.join(params.exp_dir, 'gen')):
    os.mkdir(os.path.join(params.exp_dir, 'gen'))
if not os.path.exists(os.path.join(params.exp_dir, 'ckpt')):
    os.mkdir(os.path.join(params.exp_dir, 'ckpt'))
if not os.path.exists(os.path.join(params.exp_dir, 'impute')):
    os.mkdir(os.path.join(params.exp_dir, 'impute'))


train_data, test_data = get_dataset(params.data_root, params.dataset, False)

train_mean = torch.mean(train_data, 0)
test_mean = torch.mean(test_data, 0)
model = eval(params.model_name)(
    max_time_scale=params.max_time_scale,
    time_enc_dim=params.time_enc_dim,
    time_dim=params.time_dim,
    expand_dim=params.expand_dim,
    mercer=params.mercer,
    n_layers=params.n_layers,
    n_head=params.n_heads,
    d_k=params.att_dims,
    d_v=params.att_dims,
    d_model=params.model_dims,
    d_inner=params.inner_dims,
    d_data=train_data.shape[-1],
    dropout=params.dropout,
    use_layer_norm=params.layer_norm,
    use_gap_encoding=params.use_gap_encoding,
    adapter=params.adapter,
    use_mask=params.att_mask,
    confidence=params.confidence
)
model = nn.DataParallel(model).to(device)
print(model)
print("Start Imputation")
start_time = time.time()
loss = run_imputation(model, test_data.repeat(10,1,1), args.num_missing, ckpt_dict, confidence=params.confidence , max_level=params.max_level, fig_path = os.path.join(params.exp_dir, 'impute'), 
                    save_all_imgs=args.save_fig, dataset=params.dataset, train_mean=train_mean, test_mean=test_mean, gp=params.gp)
elase_time = time.time() - start_time
output_str = 'Testing_Loss: %4f' % (loss)
print(output_str, elase_time)
