''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=0.0)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        return x

class Adapter(nn.Module):
    ''' Adapter Network '''
    def __init__(self, d_model):
        super(Adapter, self).__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_model // 8)
        self.fc2 = nn.Linear(d_model // 8, d_model)
        self.weight_init()
    
    def weight_init(self):
        nn.init.xavier_uniform_(self.fc1.weight, 1e-4)
        nn.init.xavier_uniform_(self.fc2.weight, 1e-4)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        residual = x
        x = self.fc2(F.relu(self.fc1(x)))
        x += residual
        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, use_layer_norm=0, adapter=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.use_layer_norm = use_layer_norm
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.adapter = adapter
        if adapter:
            multi_res_adapter_1 = dict()
            multi_res_adapter_2 = dict()
            i = 1
            while i < 256:
                multi_res_adapter_1[str(int(math.log(i+1,2)))] = Adapter(d_model)
                multi_res_adapter_2[str(int(math.log(i+1,2)))] = Adapter(d_model)
                i *= 2
            self.multi_res_adapter_1 = nn.ModuleDict(multi_res_adapter_1)
            self.multi_res_adapter_2 = nn.ModuleDict(multi_res_adapter_2)

    def forward(self, enc_input, gap, slf_attn_mask=None):
        residual = enc_input
        if self.use_layer_norm:
            enc_input = self.layer_norm_1(enc_input)
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if self.adapter:
            enc_output = self.multi_res_adapter_1[str(math.floor(math.log(gap+1, 2)))](enc_output)
        enc_output += residual
        residual = enc_output
        if self.use_layer_norm:
            enc_output = self.layer_norm_2(enc_output)
        enc_output = self.pos_ffn(enc_output)
        if self.adapter:
            enc_output = self.multi_res_adapter_2[str(math.floor(math.log(gap+1, 2)))](enc_output)
        enc_output += residual
        return enc_output, enc_slf_attn

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
        timescales = torch.tensor(self.get_timescales()).to(times.device)
        scaled_time = times.float() / timescales
        
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return signal

class MercerTimeEncoding(nn.Module):
    '''Self-attention with Functional Time Representation Learning'''
    def __init__(self, time_dim, expand_dim):
        super().__init__()
        self.time_dim = time_dim
        self.expand_dim = expand_dim
        self.init_period_base = nn.Parameter(torch.linspace(0, 8, time_dim))
        self.basis_expan_var = torch.empty(time_dim, 2*expand_dim)
        nn.init.xavier_uniform_(self.basis_expan_var)
        self.basis_expan_var = nn.Parameter(self.basis_expan_var)
        self.basis_expan_var_bias = nn.Parameter(torch.zeros([time_dim]))
    
    def forward(self, t):
        ''' t has shape [batch size, seq_len, 1]'''
        expand_t = t.repeat(1,1,self.time_dim)
        period_var = 10 ** self.init_period_base
        period_var = period_var[:, None].repeat(1, self.expand_dim) # [time_dim, expand_dim]
        expand_coef = torch.range(1, self.expand_dim)[None, :].float().cuda() # [1, expand_dim]
        freq_var = 1 / period_var
        freq_var = freq_var * expand_coef
        sin_enc = torch.sin(expand_t[:,:,:,None] * freq_var[None, None, :, :])
        cos_enc = torch.cos(expand_t[:,:,:,None] * freq_var[None, None, :, :])
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1) * self.basis_expan_var[None, None, :, :]
        time_enc = time_enc.sum(-1) + self.basis_expan_var_bias[None, None, :]
        return time_enc

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, max_time_scale, time_enc_dim, time_dim, expand_dim, mercer, n_layers, n_head, d_k, d_v,
            d_model, d_inner, d_data, confidence, dropout=0.1, use_layer_norm=0, use_gap_encoding=0, adapter=0, use_mask=0):

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
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, use_layer_norm=use_layer_norm, adapter=adapter)
            for _ in range(n_layers)])
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_proj = nn.Linear(d_model, d_data + 1) if confidence else nn.Linear(d_model, d_data) 
        self.use_gap_encoding = use_gap_encoding
        if use_gap_encoding:
            self.input_proj = nn.Linear(d_data + 2 * td, d_model)
        else:
            self.input_proj = nn.Linear(d_data + td + 1, d_model)
        

    def forward(self, obs_data, obs_data_mask, next_data, next_data_mask, gap, return_attns=False):
        ''' 
            obs_data has shape [batch_size, obs_len, dim_in]
            obs_time has shape [batch_size, obs_len, 1]
            imput_time has shape [batch_size, imput_len, 1]
            gap is a scalar
        '''
        num_obs = obs_data.shape[1]
        num_imp = next_data.shape[1]
        if self.use_mask:
            o2o_mask = obs_data_mask * obs_data_mask.transpose(1,2)
            o2o_mask_full = torch.zeros_like(obs_data_mask)
            o2o_mask_full = o2o_mask_full * o2o_mask_full.transpose(1,2)
            o2o_mask_full += o2o_mask


            o2u_mask = obs_data_mask * next_data_mask.transpose(1,2)
            o2u_mask_full = torch.zeros_like(obs_data_mask) * torch.zeros_like(next_data_mask).transpose(1,2)
            o2u_mask_full += o2u_mask

            u2u_mask = torch.diag_embed(next_data_mask[:,:,0])
            mask = torch.cat([torch.cat([o2o_mask, o2u_mask], 2), torch.cat([o2u_mask.transpose(1,2), u2u_mask], 2)], 1).to(obs_data)

        else:
            mask = None
        obs_time = obs_data[:,:,-1].unsqueeze(-1)
        obs_data = obs_data[:,:,:-1]
        next_time = next_data[:,:,-1].unsqueeze(-1)
        next_data = next_data[:,:,:-1]
        if self.use_gap_encoding:
            obs_time_encoding = self.position_enc(obs_time).float()
            obs = torch.cat([obs_data, obs_time_encoding, torch.zeros_like(obs_time_encoding).float()], dim=-1)
            missing_data = torch.zeros(size=(imput_time.shape[0], imput_time.shape[1], obs_data.shape[-1])).cuda()
            gap_embedding = torch.tensor([gap])[None, None, :].repeat(imput_time.shape[0], imput_time.shape[1], 1).cuda()
            imput = torch.cat([missing_data.float(), self.position_enc(imput_time).float(), self.position_enc(gap_embedding).float()], dim=-1)
        else:
            obs = torch.cat([obs_data, self.position_enc(obs_time).float(), torch.ones_like(obs_time).float()], dim=-1)
            missing_data = torch.zeros(size=(next_time.shape[0], next_time.shape[1], obs_data.shape[-1])).cuda()
            imput = torch.cat([missing_data.float(), self.position_enc(next_time).float(), torch.zeros_like(next_time).float()], dim=-1)
        combined = torch.cat([obs, imput], dim=1)
        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.input_proj(combined)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, gap, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if self.use_layer_norm:
            enc_output = self.layer_norm(enc_output)
        output = self.output_proj(enc_output[:, num_obs:, :])
        if return_attns:
            return output, enc_slf_attn_list
        return output