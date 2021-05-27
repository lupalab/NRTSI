import os
import math
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from torchdiffeq import odeint as odeint_normal
from torchdiffeq import odeint_adjoint as odeint

import pdb


class DeepsetBlock(nn.Module):
    def __init__(self, i_dim, h_dims):
        super(DeepsetBlock, self).__init__()

        self.encode = FCnet(i_dim, h_dims)
        self.max = Max(1)
        self.fc = nn.Linear(h_dims[-1], i_dim)

    def forward(self, t, x):
        x = self.encode(x)
        x = x - self.max(x)
        x = self.fc(x)
        return x


class Deepset(nn.Module):
    def __init__(self, args):
        super(Deepset, self).__init__()

        class dsetblock(nn.Module):
            def __init__(self, i_dim, h_dims):
                super(dsetblock, self).__init__()

                self.encode = FCnet(i_dim, h_dims)
                self.max = Max(1)
                self.fc = nn.Linear(h_dims[-1], i_dim)

            def forward(self, x):
                x = self.encode(x)
                x = x - self.max(x)
                x = self.fc(x)
                return x

        self.feature_extractor = FCnet(args.pts_dim, args.dims)
        self.deepset = nn.Sequential(
            *[dsetblock(args.dims[-1], args.set_hdims) for _ in range(args.num_blocks)])
        self.fc = FCnet(args.dims[-1], args.fc_dims)
        self.logit = nn.Linear(args.fc_dims[-1], 40)
        self.max = Max(1)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.deepset(x)
        x = self.max(x)
        x = x.view(x.shape[0], -1)
        x = self.logit(self.fc(x))
        return x


def FCnet(in_dim, h_dims):
    net = []
    net.append(nn.Linear(in_dim, h_dims))
    net.append(nn.Tanh())
    net.append(nn.Linear(h_dims, h_dims))

    return nn.Sequential(*net)


class SmallTransformer(nn.Module):
    
    def __init__(self, i_dim, h_dims, num_head=4):
        super(SmallTransformer, self).__init__()
        self.dim = h_dims
        self.num_head = num_head
        self.K = FCnet(i_dim, h_dims)
        self.Q = FCnet(i_dim, h_dims)
        self.V = FCnet(i_dim, h_dims)
        self.M = nn.Linear(h_dims, i_dim)
    
    def encode(self, x):
        batch_size = x.shape[0]
        k = self.K(x)
        q = self.Q(x)
        v = self.V(x)
        head_size = self.dim // self.num_head
        k = torch.cat(k.split(head_size, dim=2), dim=0)
        q = torch.cat(q.split(head_size, dim=2), dim=0)
        v = torch.cat(v.split(head_size, dim=2), dim=0)
        A = q.bmm(k.transpose(1,2)) / math.sqrt(head_size)
        A = torch.softmax(A, dim=2)
        r = torch.cat((q + A.bmm(v)).split(batch_size, dim=0), dim=2)
        r = self.M(torch.tanh(r))
        return r
    
    def forward(self, t, x):
        x = self.encode(x)
        return x


class ODEBlock(nn.Module):

    def __init__(self, odefunc, T, steps, rtol, atol, solver):
        super(ODEBlock, self).__init__()
        
        self.odefunc = odefunc
        self.integration_time = torch.linspace(0.0, T, steps).float()

        self.rtol = rtol
        self.atol = atol
        self.solver = solver

    def forward(self, x):
        self.integration_time = self.integration_time.to(x)
        if self.solver != 'dopri5':
            out = odeint_normal(self.odefunc, x, self.integration_time, self.rtol, self.atol, self.solver)
        else:
            out = odeint(self.odefunc, x, self.integration_time, self.rtol, self.atol, self.solver)
        return out[-1]
    
    def logits(self, x):
        return self.forward(x)
    
    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nef(self, value):
        self.odefunc.nfe = value


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.max(x, self.dim, keepdim=True)[0]


class Flatten(nn.Module):
    def __init__(self, ):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Transpose(nn.Module):
    def __init__(self, i_dim, j_dim):
        super(Transpose, self).__init__()

        self.i_dim = i_dim
        self.j_dim = j_dim
    
    def forward(self, x):
        return x.transpose(self.i_dim, self.j_dim)


class ODEModel(nn.Module):
    def __init__(self, sub_model, num_head, d_model, T_end=1, steps=2, tol=1e-5, solver='rk4', num_blocks=1):
        super(ODEModel, self).__init__()
        self.sub_model = sub_model
        self.num_head = num_head
        self.d_model = d_model
        self.T_end= T_end
        self.steps = steps
        self.tol = tol
        self.solver = solver
        self.num_blocks = num_blocks
        
        if self.sub_model == 'odedset':
            feature_layers = [ODEBlock(DeepsetBlock(self.d_model, self.d_model), self.T_end, self.steps, self.tol, self.tol, self.solver)
                for _ in range(self.num_blocks)]
        elif self.sub_model == 'odetrans':
            feature_layers = [ODEBlock(SmallTransformer(self.d_model, self.d_model, num_head=self.num_head), self.T_end, self.steps, self.tol, self.tol, self.solver)
                for _ in range(self.num_blocks)]
        else:
            raise NotImplementedError('the input diffeq model is not understood')
        self.model = nn.Sequential(*feature_layers)

    def forward(self, x):
        x = self.model(x)
        return x

class TimeEncoding(nn.Module):
    ''' time encoding from paper Set Functions for Time Series'''
    def __init__(self, max_time_scale, time_enc_dim):
        super(TimeEncoding, self).__init__()
        self.max_time = max_time_scale
        self.n_dim = time_enc_dim
        self._num_timescales = self.n_dim // 2
    
    def get_timescales(self):
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales[None, None, :]

    def forward(self, times):
        ''' times has shape [bs, T, 1] '''
        timescales = torch.tensor(self.get_timescales()).to(times)
        scaled_time = times.float() / timescales
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal

class ExnodeEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, max_time_scale, time_enc_dim, time_dim, expand_dim, mercer, n_layers, n_head, d_k, d_v,
            d_model, d_inner, d_data, dropout=0.1, use_layer_norm=0, use_gap_encoding=0, adapter=0, use_mask=0):

        super().__init__()
        self.mercer = mercer
        self.use_mask = use_mask
        if not self.mercer:
            self.position_enc = TimeEncoding(max_time_scale, time_enc_dim)
            td = time_enc_dim
        else:
            self.position_enc = MercerTimeEncoding(time_dim=time_dim, expand_dim=expand_dim)
            td = time_dim
        self.dropout = nn.Dropout(p=dropout)
        self.exnode = ODEModel(sub_model='odetrans', num_head=n_head, d_model=d_model)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_proj = nn.Linear(d_model, d_data)
        self.use_gap_encoding = use_gap_encoding
        if use_gap_encoding:
            self.input_proj = nn.Linear(d_data + 2 * td, d_model)
        else:
            self.input_proj = nn.Linear(d_data + td + 1, d_model)
        

    def forward(self, obs_data, obs_time, imput_time, gap, src_mask=None, return_attns=False):
        ''' 
            obs_data has shape [batch_size, obs_len, dim_in]
            obs_time has shape [batch_size, obs_len, 1]
            imput_time has shape [batch_size, imput_len, 1]
            gap is a scalar
        '''
        num_obs = obs_data.shape[1]
        num_imp = imput_time.shape[1]
        if self.use_mask:
            mask = torch.cat([torch.ones(num_obs, num_obs + num_imp), torch.cat([torch.ones(num_imp, num_obs), torch.eye(num_imp)], dim=1)], dim=0).unsqueeze(0).cuda()
        else:
            mask = None
        if self.use_gap_encoding:
            obs_time_encoding = self.position_enc(obs_time).float()
            obs = torch.cat([obs_data, obs_time_encoding, torch.zeros_like(obs_time_encoding).float()], dim=-1)
            missing_data = torch.zeros(size=(imput_time.shape[0], imput_time.shape[1], obs_data.shape[-1])).cuda()
            gap_embedding = torch.tensor([gap])[None, None, :].repeat(imput_time.shape[0], imput_time.shape[1], 1).cuda()
            imput = torch.cat([missing_data.float(), self.position_enc(imput_time).float(), self.position_enc(gap_embedding).float()], dim=-1)
        else:
            obs = torch.cat([obs_data, self.position_enc(obs_time).float(), torch.ones_like(obs_time).float()], dim=-1)
            missing_data = torch.zeros(size=(imput_time.shape[0], imput_time.shape[1], obs_data.shape[-1])).cuda()
            imput = torch.cat([missing_data.float(), self.position_enc(imput_time).float(), torch.zeros_like(imput_time).float()], dim=-1)
        combined = torch.cat([obs, imput], dim=1)
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.input_proj(combined)
        
        enc_output = self.exnode(enc_output)
        if self.use_layer_norm:
            enc_output = self.layer_norm(enc_output)
        output = self.output_proj(enc_output[:, num_obs:, :])
        if return_attns:
            return output, enc_slf_attn_list
        return output

