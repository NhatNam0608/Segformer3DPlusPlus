import math
from torch import nn
from utils.utils import cube_root
class ESA(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
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

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (batch_size, num_patches, hidden_size)
        B, N, C = x.shape

        # (batch_size, num_head, num_patches, attention_head_dim)
        q = (
            self.query(x)
            .reshape(B, N, self.num_heads, self.attention_head_dim)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            n = cube_root(N)
            # (batch_size, num_patches, embed_dim) -> (batch_size, embed_dim, D, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, n, n, n)
            # (batch_size, embed_dim, D, H, W) -> (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio)
            # (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio) -> (batch_size, num_patches/sr_ratio^3, embed_dim)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            # normalizing the layer
            x_ = self.sr_norm(x_)
            # (batch_size, num_patches/sr_ratio^3, embed_dim) -> (batch_size, num_patches/sr_ratio^3, 2, num_head, attention_head_dim)
            # (batch_size, num_patches/sr_ratio^3, 2, num_head, attention_head_dim) -> (2, batch_size, num_head, num_patches/sr_ratio^3, attention_head_dim)
            kv = (
                self.key_value(x_)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )
           
        else:
            # (2, batch_size, num_head, num_patches, attention_head_dim)
            kv = (
                self.key_value(x)
                .reshape(B, -1, 2, self.num_heads, self.attention_head_dim)
                .permute(2, 0, 3, 1, 4)
            )

        k, v = kv[0], kv[1]

        attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(self.attention_head_dim)
        attnention_prob = attention_score.softmax(dim=-1)
        attnention_prob = self.attn_dropout(attnention_prob)
        out = (attnention_prob @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out