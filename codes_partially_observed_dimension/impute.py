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

ckpt_path_dict = dict()
ckpt_root_dir = './log/air_quality_min_0.8_miss/ckpt/'
ckpt_dir = os.path.join(ckpt_root_dir, 'best_model.pt')

ckpt = torch.load(ckpt_dir)


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


train_data, val_data, test_data = get_dataset(params.data_root, params.dataset, False)

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
model.load_state_dict(ckpt)
loss = run_imputation(model, params.mode, test_data.repeat(10,1,1), args.num_missing, confidence=params.confidence , max_level=params.max_level, fig_path = os.path.join(params.exp_dir, 'impute'), 
                    save_all_imgs=args.save_fig, dataset=params.dataset, train_mean=train_mean, test_mean=test_mean, gp=params.gp)

output_str = 'Testing_Loss: %4f' % (loss)
print(output_str)
