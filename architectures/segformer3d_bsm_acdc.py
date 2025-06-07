import torch
import math
from torch import nn
import os
from merge.bsm import bipartite_soft_matching_random3d
def build_segformer3d_model(config=None):
    model = SegFormer3D(
        in_channels=config["model"]["parameters"]["in_channels"],
        sr_ratios=config["model"]["parameters"]["sr_ratios"],
        embed_dims=config["model"]["parameters"]["embed_dims"],
        patch_kernel_size=config["model"]["parameters"]["patch_kernel_size"],
        patch_stride=config["model"]["parameters"]["patch_stride"],
        patch_padding=config["model"]["parameters"]["patch_padding"],
        mlp_ratios=config["model"]["parameters"]["mlp_ratios"],
        num_heads=config["model"]["parameters"]["num_heads"],
        depths=config["model"]["parameters"]["depths"],
        decoder_head_embedding_dim=config["model"]["parameters"]["decoder_head_embedding_dim"],
        num_classes=config["model"]["parameters"]["num_classes"],
        decoder_dropout=config["model"]["parameters"]["decoder_dropout"],
        merge_modes=config["model"]["parameters"]["merge_modes"]
    )
    return model

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

    def forward(self, x, spatial_dims=None):
        B, N, C = x.shape
        assert spatial_dims is not None, "spatial_dims must be provided (D, H, W)"
        d, h, w = spatial_dims
        d_kv, h_kv, w_kv = d, h, w
        # (batch_size, num_patches, hidden_size)
        x_q = x.clone()
        x_kv = x.clone()

        if self.sr_ratio > 1:
            B, _, C = x_kv.shape
            # (batch_size, num_patches, embed_dim) -> (batch_size, embed_dim, D, H, W)
            x_kv = x_kv.permute(0, 2, 1).reshape(B, C, d_kv, h_kv, w_kv)
            # (batch_size, embed_dim, D, H, W) -> (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio)
            # (batch_size, embed_dim, D/sr_ratio, H/sr_ratio, W/sr_ratio) -> (batch_size, num_patches/sr_ratio^3, embed_dim)
            x_kv = self.sr(x_kv)
            _, _, d_kv, h_kv, w_kv = x_kv.shape
            x_kv = x_kv.reshape(B, C, -1).permute(0, 2, 1)
            # normalizing the layer
            x_kv = self.sr_norm(x_kv)
        if self.merge_mode["q_mode"] == "bsm":
            merge, unmerge = bipartite_soft_matching_random3d(metric=x_q, d=d, w=w, h=h,
                                                              r=int(x_q.size()[1] * self.merge_mode['q_r']),
                                                              sx=self.merge_mode['q_sx'], sy=self.merge_mode['q_sy'],
                                                              sz=self.merge_mode['q_sz'], rand=False)
            x_q = merge(x_q)
        if self.merge_mode["kv_mode"] == "bsm":
            merge, unmerge = bipartite_soft_matching_random3d(metric=x_kv, d=d_kv, w=w_kv, h=h_kv,
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
    
class SegFormer3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        sr_ratios: list = [4, 2, 1, 1],
        embed_dims: list = [32, 64, 160, 256],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [4, 4, 4, 4],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        decoder_dropout: float = 0.0,
        merge_modes: list = []
    ):
        """
        in_channels: number of the input channels
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratios: at which rate increasse the projection dim of the hidden_state in the mlp
        num_heads: number of attention heads
        depths: number of attention layers
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channel of the network
        decoder_dropout: dropout rate of the concatenated feature maps
        merge_modes: mode of merge strategy that using for attent efficient
        """
        super().__init__()
        self.segformer_encoder = MixVisionTransformer(
            in_channels=in_channels,
            sr_ratios=sr_ratios,
            embed_dims=embed_dims,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratios=mlp_ratios,
            num_heads=num_heads,
            depths=depths,
            merge_modes=merge_modes
        )
        reversed_embed_dims = embed_dims[::-1]
        self.segformer_decoder = SegFormerDecoderHead(
            input_feature_dims=reversed_embed_dims,
            decoder_head_embedding_dim=decoder_head_embedding_dim,
            num_classes=num_classes,
            dropout=decoder_dropout,
        )
    def forward(self, x):
     
        x = self.segformer_encoder(x)
        c1 = x[0]
        c2 = x[1]
        c3 = x[2]
        c4 = x[3]
        x = self.segformer_decoder(c1, c2, c3, c4)
        return x
    
# ----------------------------------------------------- encoder -----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channel: int = 4,
        embed_dim: int = 768,
        kernel_size: int = 7,
        stride: int = 4,
        padding: int = 3,
    ):
        """
        in_channels: number of the channels in the input volume
        embed_dim: embedding dimmesion of the patch
        """
        super().__init__()
        self.patch_embeddings = nn.Conv3d(
            in_channel,
            embed_dim,
            kernel_size=(3, kernel_size, kernel_size),
            stride=(int(stride / 2), stride, stride),
            padding=(1, padding, padding)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        patches = self.patch_embeddings(x)
        _, _, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(patches)
        return patches, (D, H, W)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        mlp_ratio: int = 2,
        num_heads: int = 8,
        sr_ratio: int = 2,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        merge_mode: dict = None
    ):
        """
        embed_dim : hidden size of the PatchEmbedded input
        mlp_ratio: at which rate increasse the projection dim of the embedded patch in the _MLP component
        num_heads: number of attention heads
        sr_ratio: the rate at which to down sample the sequence length of the embedded patch
        qkv_bias: whether or not the linear projection has bias
        attn_dropout: the dropout rate of the attention component
        proj_dropout: the dropout rate of the final linear projection
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = BSM(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            merge_mode=merge_mode
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _MLP(in_feature=embed_dim, mlp_ratio=mlp_ratio, dropout=0.0)

    def forward(self, x, spatial_dims=None):
        x = x + self.attention(self.norm1(x), spatial_dims=spatial_dims)
        x = x + self.mlp(self.norm2(x), spatial_dims=spatial_dims)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        sr_ratios: list = [8, 4, 2, 1],
        embed_dims: list = [64, 128, 320, 512],
        patch_kernel_size: list = [7, 3, 3, 3],
        patch_stride: list = [4, 2, 2, 2],
        patch_padding: list = [3, 1, 1, 1],
        mlp_ratios: list = [2, 2, 2, 2],
        num_heads: list = [1, 2, 5, 8],
        depths: list = [2, 2, 2, 2],
        merge_modes: list = []
    ):
        """
        in_channels: number of the input channels
        img_volume_dim: spatial resolution of the image volume (Depth, Width, Height)
        sr_ratios: the rates at which to down sample the sequence length of the embedded patch
        embed_dims: hidden size of the PatchEmbedded input
        patch_kernel_size: kernel size for the convolution in the patch embedding module
        patch_stride: stride for the convolution in the patch embedding module
        patch_padding: padding for the convolution in the patch embedding module
        mlp_ratio: at which rate increasse the projection dim of the hidden_state in the mlp
        num_heads: number of attenion heads
        depth: number of attention layers
        """
        super().__init__()
        self.embed_1 = PatchEmbedding(
            in_channel=in_channels,
            embed_dim=embed_dims[0],
            kernel_size=patch_kernel_size[0],
            stride=patch_stride[0],
            padding=patch_padding[0],
        )
        self.embed_2 = PatchEmbedding(
            in_channel=embed_dims[0],
            embed_dim=embed_dims[1],
            kernel_size=patch_kernel_size[1],
            stride=patch_stride[1],
            padding=patch_padding[1],
        )
        self.embed_3 = PatchEmbedding(
            in_channel=embed_dims[1],
            embed_dim=embed_dims[2],
            kernel_size=patch_kernel_size[2],
            stride=patch_stride[2],
            padding=patch_padding[2],
        )
        self.embed_4 = PatchEmbedding(
            in_channel=embed_dims[2],
            embed_dim=embed_dims[3],
            kernel_size=patch_kernel_size[3],
            stride=patch_stride[3],
            padding=patch_padding[3],
        )

        # block 1
        self.tf_block1 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    sr_ratio=sr_ratios[0],
                    qkv_bias=True,
                    merge_mode = merge_modes[0]
                )
                for _ in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # block 2
        self.tf_block2 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    sr_ratio=sr_ratios[1],
                    qkv_bias=True,
                    merge_mode = merge_modes[1]
                )
                for _ in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # block 3
        self.tf_block3 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    sr_ratio=sr_ratios[2],
                    qkv_bias=True,
                    merge_mode = merge_modes[2]
                )
                for _ in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # block 4
        self.tf_block4 = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    sr_ratio=sr_ratios[3],
                    qkv_bias=True,
                    merge_mode = merge_modes[3]
                )
                for _ in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        out = []
        # at each stage these are the following mappings:
        # (batch_size, num_patches, hidden_state)
        # (num_patches,) -> (D, H, W)
        # (batch_size, num_patches, hidden_state) -> (batch_size, hidden_state, D, H, W)

        # stage 1
        x, (d1, h1, w1) = self.embed_1(x)
        B, _, _ = x.shape
        for blk in self.tf_block1:
            x = blk(x, spatial_dims=(d1, h1, w1))
        x = self.norm1(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, d1, h1, w1, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 2
        x, (d2, h2, w2) = self.embed_2(x)
        B, _, _ = x.shape
        for blk in self.tf_block2:
            x = blk(x, spatial_dims=(d2, h2, w2))
        x = self.norm2(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, d2, h2, w2, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 3
        x, (d3, h3, w3) = self.embed_3(x)
        B, _, _ = x.shape
        for blk in self.tf_block3:
            x = blk(x, spatial_dims=(d3, h3, w3))
        x = self.norm3(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, d3, h3, w3, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        # stage 4
        x, (d4, h4, w4) = self.embed_4(x)
        B, _, _ = x.shape
        for blk in self.tf_block4:
            x = blk(x, spatial_dims=(d4, h4, w4))
        x = self.norm4(x)
        # (B, N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
        x = x.reshape(B, d4, h4, w4, -1).permute(0, 4, 1, 2, 3).contiguous()
        out.append(x)

        return out


class _MLP(nn.Module):
    def __init__(self, in_feature, mlp_ratio=2, dropout=0.0):
        super().__init__()
        out_feature = mlp_ratio * in_feature
        self.fc1 = nn.Linear(in_feature, out_feature)
        self.dwconv = DWConv(dim=out_feature)
        self.fc2 = nn.Linear(out_feature, in_feature)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, spatial_dims=None):
        x = self.fc1(x)
        x = self.dwconv(x, spatial_dims=spatial_dims)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        # added batchnorm (remove it ?)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x, spatial_dims=None):
        d, h, w = spatial_dims
        B, N, C = x.shape
        # (batch, patch_cube, hidden_size) -> (batch, hidden_size, D, H, W)
        # assuming D = H = W, i.e. cube root of the patch is an integer number!
        x = x.transpose(1, 2).view(B, C, d, h, w)
        x = self.dwconv(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


###################################################################################
# ----------------------------------------------------- decoder -------------------
class MLP_(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.bn = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        # added batchnorm (remove it ?)
        x = self.bn(x)
        return x


###################################################################################
class SegFormerDecoderHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(
        self,
        input_feature_dims: list = [512, 320, 128, 64],
        decoder_head_embedding_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.0,
    ):
        """
        input_feature_dims: list of the output features channels generated by the transformer encoder
        decoder_head_embedding_dim: projection dimension of the mlp layer in the all-mlp-decoder module
        num_classes: number of the output channels
        dropout: dropout rate of the concatenated feature maps
        """
        super().__init__()
        self.linear_c4 = MLP_(
            input_dim=input_feature_dims[0],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c3 = MLP_(
            input_dim=input_feature_dims[1],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c2 = MLP_(
            input_dim=input_feature_dims[2],
            embed_dim=decoder_head_embedding_dim,
        )
        self.linear_c1 = MLP_(
            input_dim=input_feature_dims[3],
            embed_dim=decoder_head_embedding_dim,
        )
        # convolution module to combine feature maps generated by the mlps
        self.linear_fuse = nn.Sequential(
            nn.Conv3d(
                in_channels=4 * decoder_head_embedding_dim,
                out_channels=decoder_head_embedding_dim,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm3d(decoder_head_embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

        # final linear projection layer
        self.linear_pred = nn.Conv3d(
            decoder_head_embedding_dim, num_classes, kernel_size=1
        )

        # segformer decoder generates the final decoded feature map size at 1/4 of the original input volume size
        self.upsample_volume = nn.Upsample(
            scale_factor=(2.0, 4.0, 4.0), mode="trilinear", align_corners=False
        )

    def forward(self, c1, c2, c3, c4):
       ############## _MLP decoder on C1-C4 ###########
        n, _, _, _, _ = c4.shape

        _c4 = (
            self.linear_c4(c4)
            .permute(0, 2, 1)
            .reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
            .contiguous()
        )
        _c4 = torch.nn.functional.interpolate(
            _c4,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c3 = (
            self.linear_c3(c3)
            .permute(0, 2, 1)
            .reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
            .contiguous()
        )
        _c3 = torch.nn.functional.interpolate(
            _c3,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c2 = (
            self.linear_c2(c2)
            .permute(0, 2, 1)
            .reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
            .contiguous()
        )
        _c2 = torch.nn.functional.interpolate(
            _c2,
            size=c1.size()[2:],
            mode="trilinear",
            align_corners=False,
        )

        _c1 = (
            self.linear_c1(c1)
            .permute(0, 2, 1)
            .reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])
            .contiguous()
        )
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.upsample_volume(x)
        return x
