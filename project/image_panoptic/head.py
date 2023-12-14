import math
import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple
import pdb

class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2):
        super().__init__()
        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                              nn.ReLU(True),
                              nn.Dropout(0.0)
                              ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(0.0))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(0.0)

        # pdb.set_trace()

    def forward(self, x):
        out = self.layers(x)
        return x + self.dropout_layer(out)


class DySepConvAtten(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 256
        self.num_proposals = 100
        self.kernel_size = 3

        # self.depth_weight_linear = nn.Linear(hidden_dim, kernel_size)
        # self.point_weigth_linear = nn.Linear(hidden_dim, num_proposals)
        self.weight_linear = nn.Linear(self.hidden_dim, self.num_proposals + self.kernel_size)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, query, value):
        assert query.shape == value.shape
        B, N, C = query.shape
        
        dy_conv_weight = self.weight_linear(query)
        dy_depth_conv_weight = dy_conv_weight[:, :, :self.kernel_size].view(B,self.num_proposals,1,self.kernel_size)
        dy_point_conv_weight = dy_conv_weight[:, :, self.kernel_size:].view(B,self.num_proposals,self.num_proposals,1)

        res = []
        value = value.unsqueeze(1)
        for i in range(B):
            # input: [1, N, C]
            # weight: [N, 1, K]
            # output: [1, N, C]
            out = F.relu(F.conv1d(input=value[i], weight=dy_depth_conv_weight[i], groups=N, padding="same"))
            # input: [1, N, C]
            # weight: [N, N, 1]
            # output: [1, N, C]
            out = F.conv1d(input=out, weight=dy_point_conv_weight[i], padding='same')

            res.append(out)
        point_out = torch.cat(res, dim=0)
        point_out = self.norm(point_out)
        return point_out


class CrossAttenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cls_fcs = 1
        self.num_mask_fcs = 1
        self.num_classes = 150
        self.conv_kernel_size_2d = 1

        self.hidden_dim = 256
        self.num_proposals = 100
        self.hard_mask_thr = 0.5

        self.f_atten = DySepConvAtten()
        self.f_dropout = nn.Dropout(0.0)
        self.f_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.k_atten = DySepConvAtten() 
        self.k_dropout = nn.Dropout(0.0)
        self.k_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2) 
        
        self.s_atten = nn.MultiheadAttention(embed_dim=self.hidden_dim * self.conv_kernel_size_2d**2,
                                             num_heads=8,
                                             dropout=0.0)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.ffn = FFN(self.hidden_dim, feedforward_channels=2048, num_fcs=2)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.cls_fcs = nn.ModuleList()
        for _ in range(self.num_cls_fcs):
            self.cls_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.cls_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.cls_fcs.append(nn.ReLU(True))
        self.fc_cls = nn.Linear(self.hidden_dim, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(self.num_mask_fcs):
            self.mask_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.mask_fcs.append(nn.ReLU(True))
        self.fc_mask = nn.Linear(self.hidden_dim, self.hidden_dim)

        # prior_prob = 0.01
        # self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        # pdb.set_trace()


    def forward(self, features, proposal_kernels, mask_preds, train_flag: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, C, H, W = features.shape

        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()

        # [B, N, C]
        f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, features)
        # [B, N, C, K, K] -> [B, N, C * K * K]
        k = proposal_kernels.view(B, self.num_proposals, -1)

        # ----
        f_tmp = self.f_atten(k, f)
        f = f + self.f_dropout(f_tmp)
        f = self.f_atten_norm(f)

        f_tmp = self.k_atten(k, f)
        f = f + self.k_dropout(f_tmp)
        k = self.k_atten_norm(f)
        # ----

        # [N, B, C]
        k = k.permute(1, 0, 2)

        k_tmp = self.s_atten(query = k, key = k, value = k )[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))

        # [B, N, C * K * K] -> [B, N, C, K * K] -> [B, N, K * K, C]
        obj_feat = k.reshape(B, self.num_proposals, self.hidden_dim, -1).permute(0, 1, 3, 2)

        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat

        if train_flag:
            for cls_layer in self.cls_fcs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.fc_cls(cls_feat).view(B, self.num_proposals, -1) # size() -- [1, 100, 151]
        else:
            cls_score = torch.zeros(B, self.num_proposals, 151).to(features.device) # None

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        # [B, N, K * K, C] -> [B, N, C]
        mask_kernels = self.fc_mask(mask_feat).squeeze(2)
        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)

        return cls_score, new_mask_preds, \
            obj_feat.permute(0, 1, 3, 2).reshape(B, self.num_proposals, self.hidden_dim, 
                self.conv_kernel_size_2d, self.conv_kernel_size_2d)


class YOSOHead(nn.Module):
    def __init__(self, num_stages):
        super().__init__()
        self.num_stages = num_stages # 2
        self.temperature = 0.5

        self.kernels = nn.Conv2d(in_channels=256, out_channels=100, kernel_size=1)

        self.mask_heads = nn.ModuleList()
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead())
        # pdb.set_trace()

    def forward(self, features)->Tuple[torch.Tensor, torch.Tensor]:
        # torch.jit.script not support !!!
        # for stage in range(self.num_stages + 1):
        #     if stage == 0:
        #         mask_preds = self.kernels(features)
        #         cls_scores = None
        #         proposal_kernels = self.kernels.weight.clone() # size() -- [100, 256, 1, 1]
        #         object_kernels = proposal_kernels[None].expand(features.shape[0], *proposal_kernels.size())
        #     elif stage == self.num_stages: # stage -- 2
        #         mask_head = self.mask_heads[stage - 1]
        #         cls_scores, mask_preds, proposal_kernels = mask_head(features, object_kernels, mask_preds, True)
        #     else: # stage -- 1
        #         mask_head = self.mask_heads[stage - 1]
        #         cls_scores, mask_preds, proposal_kernels = mask_head(features, object_kernels, mask_preds, False)
        #         object_kernels = proposal_kernels

        #     if cls_scores is not None:
        #         cls_scores = cls_scores / self.temperature

        # stage == 0
        mask_preds = self.kernels(features)
        proposal_kernels = self.kernels.weight.clone() # size() -- [100, 256, 1, 1]
        S1, S2, S3, S4 = proposal_kernels.size()
        object_kernels = proposal_kernels[None].expand(features.shape[0], S1, S2, S3, S4)

        # stage == 1
        # stage == 2
        for i, mask_head in enumerate(self.mask_heads):
            cls_scores, mask_preds, proposal_kernels = mask_head(features, object_kernels, mask_preds, i == 1)
            object_kernels = proposal_kernels

        if cls_scores is not None:
            cls_scores = cls_scores / self.temperature

        return cls_scores, mask_preds