import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from xformers.ops import SwiGLU as SwiGLU_cuda
from mmcv.models.bricks.registry import (
    ATTENTION,
    FEEDFORWARD_NETWORK
)
from mmcv.utils import force_fp32, auto_fp16
from mmcv.models.backbones import BaseModule
import torch.utils.checkpoint as cp
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
# @torch.compiler.disable()
# def proxy_attn(*args, **kwargs):
#     return xops.memory_efficient_attention(*args, **kwargs)


# @torch.compiler.disable()
# def proxy_mlp(module, x):
#     return module(x)

## Naive Pytorch implement
import math
def scaled_dot_product_attention(query, key, value, attn_bias=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_bias is not None:
        attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight, attn_weight @ value


class LayerScale(nn.Module):
    def __init__(
        self,
        dim,
        init_values = 1e-5,
        inplace = True,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

@ATTENTION.register_module()
class MyAttention(nn.Module):
    def __init__(
            self, 
            embed_dims, 
            head_dim=64,
            attn_drop=0.0,
            batch_first=True,
            layer_scale=None,
            no_wq=False,
            extra_post_norm=False,
            with_cp=False,
        ):
        super().__init__()
        dim = embed_dims
        self.embed_dims = embed_dims
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.no_wq = no_wq
        if not self.no_wq:
            self.w_q = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim)
            )
        self.w_k = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.w_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.fp16_enabled = False
        self.batch_first = batch_first
        self.apply(self._init_weights)
        self.with_cp = with_cp
        # if layer_scale is not None:
        #     self.ls = LayerScale(dim=dim, init_values=layer_scale)
        # else:
        #     self.ls = nn.Identity()
        self.extra_post_norm = extra_post_norm
        if self.extra_post_norm:
            self.postnorm = nn.LayerNorm(dim)
        else:
            self.postnorm = nn.Identity()

    def _init_weights(self, m):
        is_linear_layer = any([isinstance(m, nn.Linear), isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d)])
        if is_linear_layer:
            nn.init.xavier_normal_(m.weight)#, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, query, key, value, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None and (not self.no_wq):
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        B, N_Q, C = query.shape
        B2, N_KV, C2 = key.shape
        assert (B==B2 and C==C2)
        attn_bias = None
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, N_Q). expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, 1, N_Q)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))
        
        if attn_mask is not None:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=query.dtype).to(query.device)
            attn_bias = new_attn_mask.masked_fill_(attn_mask.bool(), float("-inf")).unsqueeze(1).expand(-1, self.num_heads, -1, -1).to(dtype=query.dtype)
            del new_attn_mask
            if attn_bias.shape[-1] % 8 != 0:
                attn_bias = torch.cat(
                    [attn_bias, attn_bias.new_zeros(attn_bias.shape[0], attn_bias.shape[1], attn_bias.shape[2], int(8-attn_bias.shape[3]%8))], dim=-1).contiguous()[:, :, :, :attn_bias.shape[-1]]
        if self.no_wq:
            query = query.view(B, N_Q, self.num_heads, -1)
        else:
            query = self.w_q(query).view(B, N_Q, self.num_heads, -1)
        key = self.w_k(key).view(B, N_KV, self.num_heads, -1)
        value = self.w_v(value).view(B, N_KV, self.num_heads, -1)
        if self.with_cp:
            return self.postnorm(self.proj(cp.checkpoint(xops.memory_efficient_attention, query, key, value, attn_bias, self.attn_drop, ).view(B, N_Q, -1))) + identity
        else:
            return self.postnorm(self.proj(xops.memory_efficient_attention(query, key, value, attn_bias=attn_bias, p=self.attn_drop,).view(B, N_Q, -1))) + identity
    

@FEEDFORWARD_NETWORK.register_module()
class MySwiGLU(BaseModule):
    def __init__(self,
        # in_features,
        # hidden_features = None,
        # out_features = None,
        embed_dims = None,
        feedforward_channels = None,
        ffn_drop = 0.0,
        with_cp=False,
        extra_post_norm=False):
        super().__init__()
        in_features = embed_dims
        out_features = embed_dims
        hidden_features = feedforward_channels#hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.swiglu = SwiGLU_cuda(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
            )
        # self.swiglu = nn.Sequential(
        #     nn.Linear(in_features, hidden_features, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_features, in_features, bias=False),
        # )
        self.prenorm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(ffn_drop) if ffn_drop != 0.0 else nn.Identity()
        self.fp16_enabled = False
        self.apply(self._init_weights)
        self.with_cp = with_cp
        
        self.extra_post_norm = extra_post_norm
        if self.extra_post_norm:
            self.postnorm = nn.LayerNorm(embed_dims)
        else:
            self.postnorm = nn.Identity()

    def _init_weights(self, m):
        is_linear_layer = any([isinstance(m, nn.Linear), isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d)])
        if is_linear_layer:
            nn.init.xavier_normal_(m.weight)#, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        #return x + self.dropout(proxy_mlp(self.swiglu, self.prenorm(x)))
        if self.with_cp:
            return x + self.postnorm(self.dropout(cp.checkpoint(self.swiglu, self.prenorm(x))))
        else:
            return x + self.postnorm(self.dropout(self.swiglu(self.prenorm(x))))
