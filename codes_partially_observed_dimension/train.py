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
from utils.train_utils import run_epoch, get_gap_lr_bs
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from transformers import Encoder
#from exnode import ExnodeEncoder

import torch.nn as nn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', type=str)
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

############################################################
logging.basicConfig(filename=params.exp_dir + '/train.log',
                    filemode='w',
                    level=logging.INFO,
                    format='%(message)s')
logging.info(pformat(params.dict))
############################################################


train_data, val_data, test_data = get_dataset(params.data_root, params.dataset, params.normalize)
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
optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
print("Start Training")
writer = SummaryWriter(params.exp_dir)

best_test_loss = 1e6
best_test_mse_loss = 1e6

optimizer.param_groups[0]['lr'] = params.lr
start_epoch = 0
if params.pretrained:
    checkpoint = torch.load(params.pretrained)
    model.load_state_dict(checkpoint, strict=False)
if params.resume_from_epoch > 0:
    start_epoch = params.resume_from_epoch
    checkpoint = torch.load(os.path.join(params.exp_dir, 'ckpt', "epoch_%d.pt" % params.resume_from_epoch))
    model.load_state_dict(checkpoint)

for epoch in range(start_epoch, params.epochs):
    #max_gap, min_gap, cur_lr, bs, reset_best_loss, save_ckpt, teacher_forcing, test_bs = get_gap_lr_bs(params.dataset, epoch, params.lr, params.use_ta)
    epoch_mean_train_loss = run_epoch(epoch, params.epochs, True, model, params.mode,
                                     train_data, params.clip, params.confidence, optimizer=optimizer, 
                                    batch_size=params.batch_size, fig_path=os.path.join(params.exp_dir, 'gen'), 
                                    dataset=params.dataset)
    epoch_mean_test_loss = run_epoch(epoch, params.epochs, False, model, params.mode,
                                     val_data, params.clip, params.confidence, optimizer=optimizer, 
                                     batch_size=params.batch_size, fig_path=os.path.join(params.exp_dir, 'gen'), 
                                     dataset=params.dataset)
    if epoch_mean_test_loss < best_test_loss:
        best_test_loss = epoch_mean_test_loss
        torch.save(model.state_dict(), params.exp_dir + '/ckpt/epoch_%d.pt' % epoch)
        torch.save(model.state_dict(), params.exp_dir + '/ckpt/best_model.pt')
    output_str = '[%d]/[%d] Training_Loss: %4f Testing_Loss: %4f Best Testing Loss: %4f' % (epoch, params.epochs, epoch_mean_train_loss, epoch_mean_test_loss, best_test_loss)
    
    print(output_str)
    logging.info(output_str)
    writer.add_scalar('Loss/train', epoch_mean_train_loss, epoch)
    writer.add_scalar('Loss/test', epoch_mean_test_loss, epoch)
   
