# ------------------------------------------------------------
# Author: Huynh Nhat Nam (nhatnamit214@gmail.com)
# GitHub: https://github.com/NhatNam0608
# Created: 2025-06-08
# ------------------------------------------------------------
import math
from torch import nn
from merge.bsm import bipartite_soft_matching_random3d
from utils.utils import cube_root
class BSM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        merge_mode: dict = None
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        merge_mode: mode of merge strategy for efficient attention
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dim should be divisible by number of heads!"

        self.num_heads = num_heads
        # embedding dimesion of each attention head
        self.attention_head_dim = embed_dim // num_heads

        # The same input is used to generate the query, key, and value,
        self.query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_value = nn.Linear(embed_dim, 2 * embed_dim, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.merge_mode = merge_mode

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (batch_size, num_patches, hidden_size)
        x_q = x.clone()
        x_kv = x.clone()

        if self.sr_ratio > 1:
            B, N, C = x_kv.shape
            n = cube_root(N)
            # (batch_size, num_patches, embed_dim) -> (batch_size, embed_dim, D, H, W)
            x_kv = x_kv.permute(0, 2, 1).reshape(B, C, n, n, n)
            # (batch_size, embed_dim, D, H, W) -> (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio)
            # (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio) -> (batch_size, num_patches/sr_ratio^3, embed_dim)
            x_kv = self.sr(x_kv).reshape(B, C, -1).permute(0, 2, 1)
            # normalizing the layer
            x_kv = self.sr_norm(x_kv)
        if self.merge_mode["q_mode"] == "bsm":
            B, N, C = x_q.shape
            n = cube_root(N)
            merge, unmerge = bipartite_soft_matching_random3d(metric=x_q, d=n, w=n, h=n,
                                                              r=int(x_q.size()[1] * self.merge_mode['q_r']),
                                                              sx=self.merge_mode['q_sx'], sy=self.merge_mode['q_sy'],
                                                              sz=self.merge_mode['q_sz'], rand=False)
            x_q = merge(x_q)
        if self.merge_mode["kv_mode"] == "bsm":
            B, N, C = x_kv.shape
            n = cube_root(N)
            merge, unmerge = bipartite_soft_matching_random3d(metric=x_kv, d=n, w=n, h=n,
                                                              r=int(x_kv.size()[1] * self.merge_mode['kv_r']),
                                                              sx=self.merge_mode['kv_sx'], sy=self.merge_mode['kv_sy'], 
                                                              sz=self.merge_mode['kv_sz'], rand=False)
            x_kv = merge(x_kv)
        
        B, _, C = x_q.shape
        # (batch_size, num_head, num_patches, attention_head_dim)
        q = (
            self.query(x_q)
            .reshape(B, -1, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )
        # (2, batch_size, num_head, num_patches, attention_head_dim)
        kv = (
            self.key_value(x_kv)
            .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_head_dim)
        attention_prob = attention_score.softmax(dim=-1)
        attention_prob = self.attn_dropout(attention_prob)
        out = (attention_prob @ v).transpose(1, 2).reshape(B, -1, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        if self.merge_mode["q_mode"] == "bsm":
            out = unmerge(out)
        return out
