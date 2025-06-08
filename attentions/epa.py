# ------------------------------------------------------------
# Author: Huynh Nhat Nam (nhatnamit214@gmail.com)
# GitHub: https://github.com/NhatNam0608
# Created: 2025-06-08
# ------------------------------------------------------------
import torch.nn as nn
import torch
import math
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor
class EPA(nn.Module):
    """
        Efficient Paired Attention Block, based on: "Shaker et al.,
        UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
        """
    def __init__(self, input_size, proj_size, embed_dim, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(embed_dim, embed_dim * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)
        self.gamma = nn.Parameter(1e-6 * torch.ones(embed_dim), requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum('bhdn,nk->bhdk', *args)
        k_shared_projected, v_SA_projected = map(proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF)))

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        return self.gamma*(x_CA + x_SA)
