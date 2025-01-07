import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import sys

class CFRM(nn.Module):
    """
    Feature Mutual Reconstruction Module
    """

    def __init__(self, hidden_size, inner_size=None, drop_prob=0., reconstruction_attention_dropout=0.1):
        super(CFRM, self).__init__()
        self.hidden_size = hidden_size
        self.inner_size = inner_size if inner_size is not None else hidden_size // 8
        self.num_heads = 1
        self.q_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        )
        self.k_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        )
        self.v_layer = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        )
        self.reconstruction_num_heads = 1
        head_dim = hidden_size // self.reconstruction_num_heads
        self.scale = head_dim ** -0.5
        self.scale = head_dim ** -0.5
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(reconstruction_attention_dropout)
        self.reconstruction_attn_drop = nn.Dropout(reconstruction_attention_dropout)
        self.dropout = nn.Dropout(drop_prob)
        self.flattener01 = nn.Flatten(0, 1)
        self.flattener12 = nn.Flatten(1, 2)
        self.flattener23 = nn.Flatten(2, 3)
        self.flattener34 = nn.Flatten(3, 4)
        self.flattener45 = nn.Flatten(4, 5)

    def reconstruction(self, re_support, features_a, phi):
        train_way = features_a.size(0)
        train_shot = features_a.size(2)
        re_way = phi
        roll_re_support = re_support.clone()
        for i in range(re_way - 1):
            roll_re_support = torch.roll(roll_re_support, shifts=1, dims=0)
            re_support = torch.cat((re_support, roll_re_support), dim=2)
            features_a= torch.cat((features_a, roll_re_support), dim=2)
        train_shot = features_a.size(2)
        features_a = features_a.permute(0, 2, 1, 3, 4)
        features_a = self.flattener01(features_a)
        q_a = self.q_layer(features_a)
        q_a = self.flattener23(q_a)
        q_a = q_a.permute(0, 2, 1)
        B, N, C = q_a.shape
        q_a = q_a.unsqueeze(1)
        k_a = self.k_layer(features_a)
        k_a = self.flattener23(k_a)
        k_a = k_a.permute(0, 2, 1)
        k_a = k_a.unsqueeze(1)
        v_a = self.v_layer(features_a)
        v_a = self.flattener23(v_a)
        v_a = v_a.permute(0, 2, 1)
        v_a = v_a.unsqueeze(1)

        q_a = q_a.view(q_a.size(0) // train_shot, train_shot, self.reconstruction_num_heads, q_a.size(2), q_a.size(3))
        k_a = k_a.view(k_a.size(0) // train_shot, train_shot, self.reconstruction_num_heads, k_a.size(2),
                       k_a.size(3))

        k_a = k_a.permute(0, 2, 1, 3, 4)
        k_a = self.flattener23(k_a)
        k_a = k_a.transpose(-1, -2)
        k_a = k_a.unsqueeze(1)

        v_a = v_a.view(v_a.size(0) // train_shot, train_shot, self.reconstruction_num_heads, v_a.size(2), v_a.size(3))
        v_a = v_a.permute(0, 2, 1, 3, 4)
        v_a = self.flattener23(v_a)
        v_a = v_a.unsqueeze(1)

        re_support = re_support.permute(0, 2, 1, 3, 4)
        re_support = self.flattener01(re_support)

        q_coa = self.q_layer(re_support)
        q_coa = self.flattener23(q_coa)
        q_coa = q_coa.permute(0, 2, 1)
        q_coa = q_coa.unsqueeze(1)
        k_coa = self.k_layer(re_support)
        k_coa = self.flattener23(k_coa)
        k_coa = k_coa.permute(0, 2, 1)
        k_coa = k_coa.unsqueeze(1)
        v_coa = self.v_layer(re_support)
        v_coa = self.flattener23(v_coa)
        v_coa = v_coa.permute(0, 2, 1)
        v_coa = v_coa.unsqueeze(1)

        q_coa = q_coa.view(q_coa.size(0) // train_shot, train_shot, self.reconstruction_num_heads, q_coa.size(2),
                           q_coa.size(3))
        k_coa = k_coa.view(k_coa.size(0) // train_shot, train_shot, self.reconstruction_num_heads, k_coa.size(2),
                           k_coa.size(3))

        k_coa = k_coa.permute(0, 2, 1, 3, 4)
        k_coa = self.flattener23(k_coa)
        k_coa = k_coa.transpose(-1, -2)
        k_coa = k_coa.unsqueeze(1)
        v_coa = v_coa.view(v_coa.size(0) // train_shot, train_shot, self.reconstruction_num_heads, v_coa.size(2),
                           v_coa.size(3))

        v_coa = v_coa.permute(0, 2, 1, 3, 4)
        v_coa = self.flattener23(v_coa)
        v_coa = v_coa.unsqueeze(1)

        attn = (q_a @ k_coa) * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.reconstruction_attn_drop(attn)
        re_a1 = (attn @ v_coa).reshape(B, N, C)
        re_a1 = self.proj(re_a1)
        re_a1 = self.proj_drop(re_a1)

        attn = (q_coa @ k_a) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.reconstruction_attn_drop(attn)
        re_a2 = (attn @ v_a).reshape(B, N, C)
        re_a2 = self.proj(re_a2)
        re_a2 = self.proj_drop(re_a2)
        re_a = (re_a1 + re_a2) / 2
        return re_a

    def forward(self, re_support, features_a, phi):
        '''
        (way, channel, shot, h, w)
        '''
        new_support = self.reconstruction(re_support, features_a,phi)
        return new_support

