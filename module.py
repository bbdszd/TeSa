import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from trend_intro import *
# from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import random

from easy_tpp.preprocess.event_tokenizer import EventTokenizer
from easy_tpp.config_factory import DataSpecConfig

from attnhp import AttNHP


def assert_monotonic(tensor):
    # 判断张量的维度是否正确
    assert len(tensor.shape) == 2, "Input tensor must be 2D"

    # 遍历每一行
    for row in tensor:
        # 使用 torch.all 判断是否单调递增
        assert torch.all(row[:-1] <= row[1:]), "Tensor rows must be monotonic increasing"



class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        # x = self.layer_norm(x)
        h = self.act(self.fc1(x))
        return self.fc2(h)
    

# Integrate type information bot not based on concat
class MergeLayerEvent(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4, num_edge_type):
        super().__init__()
        # self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
        self.fc1 = torch.nn.Linear(dim1 + dim2 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, num_edge_type)
        self.fc3 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=1)

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x1, x2, type):
        # x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x1, x2, type], dim=1)
        x = self.act(self.fc1(x))
        # for class
        h1 = self.softmax(self.fc2(x))
        h2 = self.fc3(x)
        return h1, h2


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_v, d]

        output = torch.bmm(attn, v)

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
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = self.layer_norm(output)

        return output, attn


class MapBasedMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.wq_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wk_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)
        self.wv_node_transform = nn.Linear(d_model, n_head * d_k, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.weight_map = nn.Linear(2 * d_k, 1, bias=False)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()

        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.wq_node_transform(q).view(sz_b, len_q, n_head, d_k)

        k = self.wk_node_transform(k).view(sz_b, len_k, n_head, d_k)

        v = self.wv_node_transform(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        q = torch.unsqueeze(q, dim=2)  # [(n*b), lq, 1, dk]
        q = q.expand(q.shape[0], q.shape[1], len_k, q.shape[3])  # [(n*b), lq, lk, dk]

        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        k = torch.unsqueeze(k, dim=1)  # [(n*b), 1, lk, dk]
        k = k.expand(k.shape[0], len_q, k.shape[2], k.shape[3])  # [(n*b), lq, lk, dk]

        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x lq x lk

        ## Map based Attention
        # output, attn = self.attention(q, k, v, mask=mask)
        q_k = torch.cat([q, k], dim=3)  # [(n*b), lq, lk, dk * 2]
        attn = self.weight_map(q_k).squeeze(dim=3)  # [(n*b), lq, lk]

        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        attn = self.softmax(attn)  # [n * b, l_q, l_k]
        attn = self.dropout(attn)  # [n * b, l_q, l_k]

        # [n * b, l_q, l_k] * [n * b, l_v, d_v] >> [n * b, l_q, d_v]
        output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)

        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.act(self.fc(output)))
        output = self.layer_norm(output + residual)

        return output, attn


def expand_last_dim(x, num):
    view_size = list(x.size()) + [1]
    expand_size = list(x.size()) + [num]
    return x.view(view_size).expand(expand_size)


class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


class PosEncode(torch.nn.Module):
    def __init__(self, expand_dim, seq_len):
        super().__init__()

        self.pos_embeddings = nn.Embedding(num_embeddings=seq_len, embedding_dim=expand_dim)

    def forward(self, ts):
        # ts: [N, L]
        order = ts.argsort()
        ts_emb = self.pos_embeddings(order)
        return ts_emb


class EmptyEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super().__init__()
        self.expand_dim = expand_dim

    def forward(self, ts):
        out = torch.zeros_like(ts).float()
        out = torch.unsqueeze(out, dim=-1)
        out = out.expand(out.shape[0], out.shape[1], self.expand_dim)
        return out


class LSTMPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim, time_dim):
        super(LSTMPool, self).__init__()
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim

        self.att_dim = feat_dim + edge_dim + time_dim

        self.act = torch.nn.ReLU()

        self.lstm = torch.nn.LSTM(input_size=self.att_dim,
                                  hidden_size=self.feat_dim,
                                  num_layers=1,
                                  batch_first=True)
        self.merger = MergeLayer(feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        seq_x = torch.cat([seq, seq_e, seq_t], dim=2)

        _, (hn, _) = self.lstm(seq_x)

        hn = hn[-1, :, :]  # hn.squeeze(dim=0)

        out = self.merger.forward(hn, src)
        return out, None


class MeanPool(torch.nn.Module):
    def __init__(self, feat_dim, edge_dim):
        super(MeanPool, self).__init__()
        self.edge_dim = edge_dim
        self.feat_dim = feat_dim
        self.act = torch.nn.ReLU()
        self.merger = MergeLayer(edge_dim + feat_dim, feat_dim, feat_dim, feat_dim)

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        # seq [B, N, D]
        # mask [B, N]
        src_x = src
        seq_x = torch.cat([seq, seq_e], dim=2)  # [B, N, De + D]
        hn = seq_x.mean(dim=1)  # [B, De + D]
        output = self.merger(hn, src_x)
        return output, None


class AttnModel(torch.nn.Module):
    """Attention based temporal layers
    """

    def __init__(self, feat_dim, edge_dim, time_dim,
                 attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(AttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim

        self.edge_in_dim = (feat_dim + edge_dim + time_dim)
        self.model_dim = self.edge_in_dim
        # self.edge_fc = torch.nn.Linear(self.edge_in_dim, self.feat_dim, bias=False)

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        # self.act = torch.nn.ReLU()

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(n_head,
                                                        d_model=self.model_dim,
                                                        d_k=self.model_dim // n_head,
                                                        d_v=self.model_dim // n_head,
                                                        dropout=drop_out)
            self.logger.info('Using scaled prod attention')

        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(n_head,
                                                                d_model=self.model_dim,
                                                                d_k=self.model_dim // n_head,
                                                                d_v=self.model_dim // n_head,
                                                                dropout=drop_out)
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, src_t, seq, seq_t, seq_e, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        # src_e_ph = torch.zeros_like(src_ext)
        src_e_ph = torch.zeros(src_ext.shape[0], src_ext.shape[1], seq_e.shape[2]).to(src.device)
        q = torch.cat([src_ext, src_e_ph, src_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t], dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]
#        print(mask[0])
        # print('mask.shape', mask.shape)

        # # target-attention
        output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask)  # output: [B, 1, D + Dt], attn: [B, 1, N]
        output = output.squeeze()
        if output.dim() == 1:
            output = torch.unsqueeze(output, dim=0)
        attn = attn.squeeze()

        output = self.merger(output, src)

        return output, attn


class TGAN(torch.nn.Module):
    def __init__(self, ngh_finder, n_feat, e_feat, n_degree,
                 attn_mode='prod', use_time='time', agg_method='attn', node_dim=None, time_dim=None,
                 num_layers=3, n_head=4, null_idx=0, num_heads=1, drop_out=0.1, seq_len=None, num_edge_type=3, num_node_type=2):
        super(TGAN, self).__init__()

        # self.n_degree = n_degree
        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.num_edge_type = num_edge_type
        self.num_node_type = num_node_type
        self.logger = logging.getLogger(__name__)
        self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)

        self.feat_dim = self.n_feat_th.shape[1]
        self.e_feat_dim = self.e_feat_th.shape[1]

        self.edge_dim_convertor = torch.nn.Linear(self.e_feat_dim, self.feat_dim)
        torch.nn.init.xavier_normal_(self.edge_dim_convertor.weight)
        
        # +1 for not dealing with edge type with 0, equally work
        self.edge_type_embed = nn.Parameter(torch.FloatTensor(num_edge_type + 1, self.feat_dim))
        nn.init.uniform_(self.edge_type_embed, -np.sqrt(3.0 / (self.feat_dim + self.feat_dim)),
                         np.sqrt(3.0 / (self.feat_dim + self.feat_dim)))

        # node type 0,1
        self.embed_typed = nn.ModuleList([nn.Linear(self.feat_dim, self.feat_dim) for _ in range(num_node_type)])
        self.n_feat_dim = self.feat_dim
        # self.e_feat_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.use_time = use_time
        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, self.feat_dim)
        # todo: args configuration
        self.attnhp = AttNHP(self.feat_dim, hidden_size=16, use_ln=False, time_emb_size=16, num_layers=2, num_heads=2,\
                             dropout_rate=0.0, num_edge_type=self.num_edge_type+2)
        self.nbr_embed = nn.ModuleList([nn.Linear(self.feat_dim, self.feat_dim) for _ in range(num_edge_type)])
        self.hete_att = nn.Linear(self.feat_dim, 1)

        if agg_method == 'attn':
            self.logger.info('Aggregation uses attention model')
            self.attn_model_list = torch.nn.ModuleList([AttnModel(self.feat_dim,
                                                                  self.e_feat_dim,
                                                                  self.feat_dim,
                                                                  attn_mode=attn_mode,
                                                                  n_head=n_head,
                                                                  drop_out=drop_out) 
                                                        for _ in range(num_edge_type)
                                                        for _ in range(num_layers)])
        elif agg_method == 'lstm':
            self.logger.info('Aggregation uses LSTM model')
            self.attn_model_list = torch.nn.ModuleList([LSTMPool(self.feat_dim,
                                                                 self.feat_dim,
                                                                 self.feat_dim) 
                                                        for _ in range(num_edge_type)
                                                        for _ in range(num_layers)])
        elif agg_method == 'mean':
            self.logger.info('Aggregation uses constant mean model')
            self.attn_model_list = torch.nn.ModuleList([MeanPool(self.feat_dim,
                                                                 self.feat_dim) 
                                                        for _ in range(num_edge_type)
                                                        for _ in range(num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')

        if use_time == 'time':
            self.logger.info('Using time encoding')
            self.time_encoder = TimeEncode(expand_dim=self.n_feat_th.shape[1])
        elif use_time == 'pos':
            assert (seq_len is not None)
            self.logger.info('Using positional encoding')
            self.time_encoder = PosEncode(expand_dim=self.n_feat_th.shape[1], seq_len=seq_len)
        elif use_time == 'empty':
            self.logger.info('Using empty encoding')
            self.time_encoder = EmptyEncode(expand_dim=self.n_feat_th.shape[1])
        else:
            raise ValueError('invalid time option!')

        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1)
        self.event_score = MergeLayerEvent(self.feat_dim, self.feat_dim, self.feat_dim, 1, self.num_edge_type)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers, num_neighbors)

        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score
        
    def contrast(self, batch_data, num_neighbors=20, tra_len=20):
        e_type_l = batch_data[0]
        src_idx_l, src_type_l, _, _ = batch_data[1]
        target_idx_l, target_type_l, background_idx_l, background_type_l = batch_data[2]
        cut_time_l = batch_data[3]
        
        src_embed = self.tem_conv(src_idx_l, src_type_l, cut_time_l, self.num_layers, num_neighbors)
        target_embed = self.tem_conv(target_idx_l, target_type_l, cut_time_l, self.num_layers, num_neighbors)
        background_embed = self.tem_conv(background_idx_l, background_type_l, cut_time_l, self.num_layers, num_neighbors)
        
        e_embed = self.edge_type_embed[e_type_l]

        pos_score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed, background_embed).squeeze(dim=-1)
        
        pos_class, pos_event_score = self.event_score(src_embed, target_embed, e_embed)
        neg_class, neg_event_score = self.event_score(src_embed, background_embed, e_embed)
        
        pos_event_score = pos_event_score.squeeze(dim=-1)
        neg_event_score = neg_event_score.squeeze(dim=-1)
        # trend implementation
        event_intensity = torch.sigmoid(pos_event_score) + 1e-6
        log_event_intensity = torch.mean(-torch.log(event_intensity))
        neg_event_intensity = torch.sigmoid(- neg_event_score) + 1e-6
        neg_mean_intensity = torch.mean(-torch.log(neg_event_intensity))

        l2_loss = torch.sum(torch.pow(self.edge_type_embed, 2))
        l2_loss = l2_loss * 0.001

        L = log_event_intensity + neg_mean_intensity + l2_loss
        
        return pos_score.sigmoid(), neg_score.sigmoid(), pos_event_score.sigmoid(), neg_event_score.sigmoid(), L
    
    def tem_conv(self, src_idx_l, src_type_l, cut_time_l, curr_layers, num_neighbors=20, tra_len=20):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            type_spe_src_node_feat = torch.zeros_like(src_node_feat)
            
            for i in range(self.num_node_type):
                mask = (src_type_l == i)
                type_spe_src_node_feat[mask] = self.embed_typed[i](src_node_feat[mask])
            return type_spe_src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(src_idx_l,
                                               src_type_l,
                                               cut_time_l,
                                               curr_layers=curr_layers - 1,
                                               num_neighbors=num_neighbors)

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch, _, src_ngh_dst_type_batch, src_ngh_edge_type_batch \
                = self.ngh_finder.get_temporal_neighbor(
                    src_idx_l,
                    cut_time_l,
                    num_neighbors=num_neighbors)

            # src_ngh_node_batch_th = torch.from_numpy(src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(src_ngh_eidx_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(src_ngh_t_batch_delta).float().to(device)
            src_ngh_edge_type_batch_th = torch.from_numpy(src_ngh_edge_type_batch).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_type_batch_flat = src_ngh_dst_type_batch.flatten()
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten()  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_node_type_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors)
            try:
                src_ngh_feat = src_ngh_node_conv_feat.view(batch_size, num_neighbors, -1)
            except:
                print(src_ngh_feat.shape)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)
            
            node_fea_tpp = self.tpp_evolution(src_ngh_t_batch_th, src_ngh_edge_type_batch_th, src_ngh_feat, tra_len)
            ngh_size = src_ngh_feat.shape[1]
            local_list = []
            # attention aggregation
            for i in range(self.num_edge_type):
                attn_m = self.attn_model_list[(curr_layers - 1) * self.num_edge_type + i]
                # mask = (type_seqs != 0) | (src_ngh_edge_type_batch_th != i + 1)  # edge type 1,2,3...
                mask = (src_ngh_edge_type_batch_th != i + 1)
                '''
                local, weight = attn_m(src_node_conv_feat,
                                   src_node_t_embed,
                                   src_ngh_feat,
                                   src_ngh_t_embed,
                                   src_ngn_edge_feat,
                                   mask)
                '''
                local, weight = attn_m(src_node_conv_feat,
                                       src_node_t_embed,
                                       # node_fea_tpp.unsqueeze(1).repeat(1, ngh_size, 1),
                                       node_fea_tpp,
                                       src_ngh_t_embed[:, 0:1, :].repeat(1, ngh_size, 1),
                                       # src_ngh_t_embed,
                                       src_ngn_edge_feat,
                                       mask)
                local_list.append(local)
            local = sum(local_list) / len(local_list)
            
            # xin v2
            return F.softplus(local)
            # return local
            # return torch.tanh(local)
            
    def tpp_evolution(self, t_batch, type_batch, event_fea_batch, tra_len=None):
        if torch.any(t_batch != 0) and any(torch.unique(row).numel() > 1 for row in t_batch):
            a = 1
        t_batch = t_batch[:, -tra_len:]
        type_batch = type_batch[:, -tra_len:]
        event_fea_batch = event_fea_batch[:, -tra_len:]
        # TPP processing
        # 1. Data preparation
        # t_batch 本质上是\delat time, 所以t本来是从小到大，后来变成从大到小，因此倒一下就可以了，但type无需翻转
        time_seqs = torch.flip(t_batch, dims=[1])
        # row_mins, _ = torch.min(time_seqs, dim=1, keepdim=True)
        time_seqs = time_seqs - torch.min(time_seqs)
        # assert_monotonic(time_seqs)
        
        diff = time_seqs[:, 1:] - time_seqs[:, :-1]
        time_seqs_list = time_seqs.tolist()
        time_seqs_list = [row for row in time_seqs_list]
        
        # edge type 1,2,3   so if 0 invalid
        type_seqs = type_batch
        # if not torch.all(type_seqs.eq(0)):
        #     a = 1
        type_seqs_list = type_seqs.tolist()
        type_seqs_list = [row for row in type_seqs_list]
        
        zeros_column = torch.zeros(diff.size(0), 1, dtype=diff.dtype, device=diff.device)
        time_delta_seqs = torch.cat((zeros_column, diff), dim=1)
        time_delta_seqs_list = time_delta_seqs.tolist()
        time_delta_seqs_list = [row for row in time_delta_seqs_list]
        
        input_data = {'time_seqs': time_seqs_list,
                      'type_seqs': type_seqs_list,
                      'time_delta_seqs': time_delta_seqs_list}
        
        # edge_type 1,2,3 so if 0 invalid, +1 for processing this situation
        config = DataSpecConfig.parse_from_yaml_config({'num_event_types': self.num_edge_type + 2,  'pad_token_id': self.num_edge_type + 2})
        tokenizer = EventTokenizer(config)
        output = tokenizer.pad(input_data, return_tensors='pt').to(diff.device)
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask, type_mask = list(output.values())
        batch_non_pad_mask = batch_non_pad_mask & (type_seqs != 0)
        node_fea_tpp = self.attnhp.loglike_loss(time_seqs, time_delta_seqs, type_seqs + 1, batch_non_pad_mask,\
                                                attention_mask, type_mask, event_fea_batch)
        return node_fea_tpp

