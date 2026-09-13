import warnings
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn.init import normal_
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.models import build_norm_layer, xavier_init
from mmcv.utils import ext_loader, deprecated_api_warning, ConfigDict
from mmcv.models.backbones.base_module import BaseModule
from mmcv.models.utils.builder import TRANSFORMER
from mmcv.models.bricks.drop import build_dropout
from mmcv.models.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE,
                                      FEEDFORWARD_NETWORK)
from mmcv.models.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.utils import force_fp32, auto_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
from .utils import (pos2posemb, agent_map_ego_split, map_transform_box)
import xformers.ops as xops
from xformers.ops import SwiGLU as SwiGLU_cuda


@ATTENTION.register_module()
class AttentionLayer(nn.Module):
    """
    Basic AttentionLayer in DriveTransformer
    """
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
        self.extra_post_norm = extra_post_norm
        if self.extra_post_norm:
            self.postnorm = nn.LayerNorm(dim)
        else:
            self.postnorm = nn.Identity()

    def _init_weights(self, m):
        is_linear_layer = any([isinstance(m, nn.Linear), isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d)])
        if is_linear_layer:
            nn.init.xavier_normal_(m.weight)
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
class SwiGLULayer(BaseModule):
    """
    Basic SwiGLULayer in DriveTransformer.
    """
    def __init__(self,
        embed_dims = None,
        feedforward_channels = None,
        ffn_drop = 0.0,
        with_cp=False,
        extra_post_norm=False):
        super().__init__()
        in_features = embed_dims
        out_features = embed_dims
        hidden_features = feedforward_channels
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.swiglu = SwiGLU_cuda(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
            )
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
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_cp:
            return x + self.postnorm(self.dropout(cp.checkpoint(self.swiglu, self.prenorm(x))))
        else:
            return x + self.postnorm(self.dropout(self.swiglu(self.prenorm(x))))
        
@TRANSFORMER_LAYER.register_module()
class DriveTransformerPreDecoderLayer(BaseModule):
    """ Base TransformerLayer for initial decode in DriveTransformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]
        super().__init__(init_cfg)

        self.batch_first = batch_first

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))
        self.use_checkpoint = with_cp
        self.fp16_enabled = False

    @auto_fp16()
    def _forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerPreDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
                else:
                    temp_key = temp_value = query
                if temp_pos is not None:
                    temp_pos = torch.cat([query_pos, temp_pos], dim=1)
                else:
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, None)
                ffn_index += 1
        return query

    @auto_fp16()
    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.

        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                temp_memory,
                temp_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
                query,
                key,
                value,
                query_pos,
                key_pos,
                temp_memory,
                temp_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        return x

@TRANSFORMER_LAYER.register_module()
class DriveTransformerDecoderLayer(BaseModule):
    """ Major layer of DriveTransformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 memory_len_frame=3,
                 agent_query_num=300,
                 map_query_num=50,
                 agent_num_propagated=50,
                 map_num_propagated=50,
                 map_pts_per_vec=4,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first
        self.agent_query_num = agent_query_num
        self.map_query_num = map_query_num
        self.memory_len_frame = memory_len_frame
        self.map_pts_per_vec = map_pts_per_vec
        self.agent_num_propagated = agent_num_propagated
        self.map_num_propagated = map_num_propagated

        assert set(operation_order) & {
            'task_self_attn', 'norm', 'ffn', 'temporal_cross_attn', 'sensor_cross_attn'} == \
            {'task_self_attn', 'norm', 'ffn', 'temporal_cross_attn', 'sensor_cross_attn'}, f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all five operation type ' \
            f"{'task_self_attn', 'norm', 'ffn', 'temporal_cross_attn', 'sensor_cross_attn'}"

        num_attn = operation_order.count('task_self_attn') + operation_order.count(
            'temporal_cross_attn') + operation_order.count('sensor_cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        #self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.task_self_attentions = ModuleList()
        self.temporal_cross_attention_layers = ModuleList()
        self.sensor_cross_attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            # build temporal cross attention, task self attention and sensor cross attention respectively
            if operation_name == 'temporal_cross_attn':
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                for _ in range(3):
                    attention = build_attention(attn_cfgs[index])
                    attention.operation_name = operation_name
                    self.temporal_cross_attention_layers.append(attention)
                index += 1
            if operation_name == 'task_self_attn':
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                attention.operation_name = operation_name
                self.task_self_attentions.append(attention)
                index += 1
            if operation_name in 'sensor_cross_attn':
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                attention.operation_name = operation_name
                self.sensor_cross_attentions.append(attention)
                index += 1
        self.embed_dims = self.task_self_attentions[0].embed_dims
        self.cross_w_q = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(self.embed_dims), nn.Linear(self.embed_dims, self.embed_dims)) for _ in range(3)]
            )

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                nn.ModuleList(
                [build_feedforward_network(ffn_cfgs[ffn_index], dict(type='FFN')) for _ in range(3)]
                )
                )
        self.use_checkpoint = with_cp
        self.fp16_enabled = False

    @auto_fp16()
    def temporal_cross_attentions(self,
                                  query,
                                  pos,
                                  agent_temp_memory,
                                  agent_temp_pos,
                                  map_temp_memory,
                                  map_temp_pos,
                                  ego_temp_memory,
                                  ego_temp_pos,
                                  agent_query_num,
                                  map_query_num,
                                  temp_attn_masks,):
        # split query and PE
        agent_query, map_query, ego_query = agent_map_ego_split(query, agent_query_num, map_query_num, dim=1)
        agent_pos, map_pos, ego_pos = agent_map_ego_split(pos, agent_query_num, map_query_num, dim=1)
        queries = [agent_query, map_query, ego_query]
        poses = [agent_pos, map_pos, ego_pos]
        temp_memories = [agent_temp_memory, map_temp_memory, ego_temp_memory]
        temp_poses = [agent_temp_pos, map_temp_pos, ego_temp_pos]
        out_queries = []
        # attention masks
        temp_attn_masks = ~(temp_attn_masks.bool()).squeeze(-1)
        temp_attn_masks = [
            torch.cat([torch.zeros((agent_query.shape[0], agent_query.shape[1], 1)).bool().to(agent_query.device), temp_attn_masks.unsqueeze(1).expand(-1, agent_query.shape[1], -1).repeat_interleave(repeats=self.agent_num_propagated, dim=-1)], dim=-1),
            torch.cat([torch.zeros((map_query.shape[0], map_query.shape[1], 1)).bool().to(agent_query.device), temp_attn_masks.unsqueeze(1).expand(-1, map_query.shape[1], -1).repeat_interleave(repeats=self.map_num_propagated, dim=-1)], dim=-1),
            torch.cat([torch.zeros((ego_query.shape[0], ego_query.shape[1], 1)).bool().to(agent_query.device),temp_attn_masks.unsqueeze(1).expand(-1, ego_query.shape[1], -1)], dim=-1)
        ]
        # temporal attention for each query 
        for i in range(3):
            out_queries.append(
                self.temporal_cross_attention_layers[i](
                queries[i],
                torch.cat([torch.zeros((temp_memories[i].shape[0], 1, temp_memories[i].shape[2])).to(temp_memories[i].device), temp_memories[i]], dim=1), 
                torch.cat([torch.zeros((temp_memories[i].shape[0], 1, temp_memories[i].shape[2])).to(temp_memories[i].device), temp_memories[i]], dim=1),
                None, 
                query_pos=poses[i],
                key_pos=torch.cat([torch.zeros((temp_poses[i].shape[0], 1, temp_poses[i].shape[2])).to(temp_poses[i].device), temp_poses[i]], dim=1), 
                attn_mask=temp_attn_masks[i]
                )
            )
        return torch.cat(out_queries, dim=1)
    
    
    @auto_fp16()
    def _forward(self,
                agent_query, # agent query, [B, N_agent_query, D] 
                map_query, # map query, [B, N_map_query, D]
                ego_query, # ego query, [B, N_ego_mode, D]
                img_feats, # image feature, [B, N_image_token, D]
                img_pos_embed, # image PE, [B, N_image_token, D]
                agent_temp_memory, # memorized agent query, [B, L_memory * N_memory_agent_per_frame, D]
                agent_temp_pos, # memorized agent PE, [B, L_memory * N_memory_agent_per_frame, D]
                map_temp_memory, # memorized map query, [B, L_memory * N_memory_map_per_frame, D]
                map_temp_pos, # memorized map PE, [B, L_memory * N_memory_map_per_frame, D]
                ego_temp_memory, # memorized ego query, [B, L_memory, D] 
                ego_temp_pos, # memorized ego PE, [B, N_agent_query, C_box]
                agent_pos_embed, # agent position embedding
                map_pts_pos_embed, # map points position embedding
                map_ins_pos_embed, # map instance position embedding
                ego_pos_embed, # ego position embedding
                attn_masks=None,  # attention mask 
                temp_attn_masks=None, # temproral attention mask
                **kwargs):

        norm_index = 0
        attn_index = 0
        task_self_attn_index = 0
        sensor_cross_attn_index = 0
        ffn_index = 0
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        agent_query_num = agent_query.shape[1]
        map_query_num = map_query.shape[1]

        query = torch.cat([agent_query, map_query, ego_query], dim=1) # [B, N_agent_query+N_map_query+N_ego_mode, D] 
        pos = torch.cat([agent_pos_embed, map_ins_pos_embed, ego_pos_embed], dim=1) # [B, N_agent_query+N_map_query+N_ego_mode, D] 
        for layer in self.operation_order:
            if layer == 'task_self_attn': # task self attention
                
                ##############################################################
                # project-1
                # TODO-6 attention mask, ego query 
                n_agent = agent_query_num
                n_map = map_query_num
                n_ego = 1
                total = n_agent + n_map + n_ego
                # 创建基础掩码（True表示需要mask）
                mask = torch.ones(total, total, dtype=torch.bool, device=query.device)
                # 定义任务区间
                agent_range = slice(0, n_agent)
                map_range = slice(n_agent, n_agent + n_map)
                ego_range = slice(n_agent + n_map, total)
                # 设置任务间不可见（mask=True）
                # 任务内部可见（mask=False）
                ##############################################################
                # project-1
                # 允许agent看agent
                # 替换此处代码
                pass
                # 允许map看agent, map
                # 替换此处代码
                pass
                ###############################################################
                # project-2
                # 允许agent看agent, map
                # 替换此处代码
                pass
                # 允许map看agent, map
                # 替换此处代码
                pass
                ###############################################################
                # project-3
                # 允许agent, map, ego互相看
                # 替换此处代码
                pass
                ###############################################################
                mask[ego_range, :] = False
                mask = mask.unsqueeze(0)
                attn_masks[attn_index] = mask

                query = self.task_self_attentions[task_self_attn_index](
                    query,
                    query,
                    query,
                    None,
                    query_pos=pos,
                    key_pos=pos,
                    attn_mask=attn_masks[attn_index],
                    **kwargs)
                task_self_attn_index += 1
                attn_index += 1
            
            elif layer == 'temporal_cross_attn': # temporal cross attention
                query = self.temporal_cross_attentions(
                    query,
                    pos,
                    agent_temp_memory,
                    agent_temp_pos,
                    map_temp_memory,
                    map_temp_pos,
                    ego_temp_memory,
                    ego_temp_pos,
                    agent_query_num,
                    map_query_num,
                    temp_attn_masks=temp_attn_masks,
                )
            elif layer == 'sensor_cross_attn': # sensor cross attention
                agent_query, map_query, ego_query = agent_map_ego_split(query, agent_query_num, map_query_num, dim=1)
                map_query = map_query.unsqueeze(2).expand(-1, -1, self.map_pts_per_vec, -1).flatten(1, 2) # [B, N_map_query*points_per_line, D]
                identity = torch.cat([agent_query, map_query, ego_query], dim=1)
                agent_query = self.cross_w_q[0](agent_query + agent_pos_embed)
                if map_pts_pos_embed is not None:
                    map_query = self.cross_w_q[1](map_query + map_pts_pos_embed) 
                ego_query = self.cross_w_q[2](ego_query + ego_pos_embed)
                query = self.sensor_cross_attentions[sensor_cross_attn_index](
                    torch.cat([agent_query, map_query, ego_query], dim=1),
                    img_feats,
                    img_feats,
                    identity,
                    query_pos=None,
                    key_pos=img_pos_embed,
                    **kwargs)
                sensor_cross_attn_index += 1
                attn_index += 1
            elif layer == 'ffn': # separated ffn for each part of query
                agent_query, map_query, ego_query = agent_map_ego_split(query, agent_query_num, map_query_num * self.map_pts_per_vec, dim=1)
                agent_query = self.ffns[ffn_index][0](agent_query)
                map_query = self.ffns[ffn_index][1](map_query)
                ego_query = self.ffns[ffn_index][2](ego_query)
                ffn_index += 1
                return agent_query, map_query, ego_query

    def forward(self, 
                *args,
                **kwargs
                ):
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(self._forward, *args, **kwargs)

        else:
            x = self._forward(*args, **kwargs)
        return x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DriveTransformerDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 agent_num_query,
                 map_num_query,
                 map_num_pts_per_vec,
                 fut_mode=None,
                 embed_dims=None,
                 refine=False,
                 **kwargs):
        
        super(DriveTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fut_mode = fut_mode
        self.agent_num_query = agent_num_query
        self.map_num_query = map_num_query
        self.map_num_pts_per_vec = map_num_pts_per_vec
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.refine = refine

    @auto_fp16()
    def forward(
            self,
            agent_query, # agent query, [B, N_agent_query, D] 
            map_query, # map query, [B, N_map_query, D]
            ego_query, # ego query, [B, N_ego_mode, D]
            img_feats, # image feature, [B, N_image_token, D]
            img_pos_embed, # image PE, [B, N_image_token, D]
            agent_temp_memory, # memorized agent query, [B, L_memory * N_memory_agent_per_frame, D]
            agent_temp_pos, # memorized agent PE, [B, L_memory * N_memory_agent_per_frame, D]
            map_temp_memory, # memorized map query, [B, L_memory * N_memory_map_per_frame, D]
            map_temp_pos, # memorized map PE, [B, L_memory * N_memory_map_per_frame, D]
            ego_temp_memory, # memorized ego query, [B, L_memory, D] 
            ego_temp_pos, # memorized ego PE, [B, N_agent_query, C_box]
            agent_ref, # reference points of agents, [B, N_map_query,2]
            map_ref, # reference points of maps (center of polyline), [B,N_map_query, N_pts_per_line, 2]
            map_pts_coord,  # reference points of each points in polyline, [B,N_map_query, N_pts_per_line, 2]
            ego_ref, # reference points of ego, [B, N_ego_mode,3]
            agent_traj_ref, # reference points of agent trajectory, [B ,N_agent_query, N_mode,N_future, 2]
            ego_traj_ref_fix_time, # reference points of ego with fixed time interval, [B, N_mode, N_future_ego_time, 2]
            ego_traj_ref_fix_dist, # reference points of ego with fixed distance interval, [B, N_mode, N_future_ego_dist, 2]
            mode_query=None, # mode query, [B, N_mode, D] 
            agent_cls=None, # preliminary classification prediction of agents, [B, N_agent_query, N_object_type]
            map_cls=None, # preliminary classification prediction of maps, [B, N_map_query, N_map_type]
            agent_ref_embedding=None, # agent reference points embedding layer
            agent_cls_embedding=None, # agent classfication embedding layer
            map_ref_embedding=None, # map position embedding layer
            map_cls_embedding=None, # map classfication embedding layer
            ego_pos_embedding=None, # ego position embedding layer
            ego_traj_ref_fix_time_embedding=None, # ego fixed time interval trajectory position embedding layer
            ego_traj_ref_fix_dist_embedding=None, # ego fixed distance interval trajectory position embedding layer
            reg_branches=None, # regression head for dedection
            cls_branches=None, # classification head for dedection
            traj_branches=None, # regression head for motion prediction 
            traj_cls_branches=None, # classification head for motion prediction 
            map_reg_branches=None, # regression head for online mapping
            map_cls_branches=None, # classification head for online mapping
            attn_mask=None, # attention mask
            temp_attn_masks=None, # temproral attention mask
            ego_traj_branches_fix_dist=None, # regression head for planning (fixed distance interval form)
            ego_traj_branches_fix_time=None, # regression head for planning (fixed time interval form)
            ego_traj_cls_branches=None, # classification head for planning
        ):
        bs = agent_query.shape[0]
        intermediate_agent_traj_coords = []
        intermediate_agent_traj_cls = []
        intermediate_agent_coords_bev = []
        intermediate_agent_coords = []
        intermediate_agent_class = []
        intermediate_map_coords = []
        intermediate_map_class = []
        intermediate_ego_traj_fix_dist = []
        intermediate_ego_traj_fix_time = []
        intermediate_ego_class = []
        cur_agent_pred = agent_ref.clone()
        agent_ref = agent_ref[..., [0, 1, 4]] #(x,y,z)
        for lid, layer in enumerate(self.layers):
            # position embeddings
            ego_pos_embed = ego_pos_embedding(pos2posemb(ego_ref.detach(), self.embed_dims//2))
            agent_pos_embed = agent_ref_embedding(pos2posemb(agent_ref.detach(), self.embed_dims//2))
            map_ins_pos_embed = map_ref_embedding(pos2posemb(map_ref.detach(), self.embed_dims//2))
            ego_pos_embed += ego_traj_ref_fix_time_embedding(pos2posemb(ego_traj_ref_fix_time.detach().flatten(-2, -1), 32))
            if ego_traj_ref_fix_dist is not None:
                ego_pos_embed += ego_traj_ref_fix_dist_embedding(pos2posemb(ego_traj_ref_fix_dist.detach().flatten(-2, -1), 32))
            agent_pos_cls_embed = agent_pos_embed + agent_cls_embedding(agent_cls.detach())
            map_class_embed = map_cls_embedding(map_cls.detach())
            map_ins_pos_embed = map_ins_pos_embed + map_class_embed
            map_pts_pos_embed = (map_ref_embedding(pos2posemb(map_pts_coord.detach(), self.embed_dims//2)) + map_class_embed.unsqueeze(-2)).flatten(1, 2)
            # major attention layer: task self attention, temporal cross attention, sensor cross attention
            agent_query, map_query, ego_query = layer(agent_query, map_query, ego_query, img_feats, img_pos_embed, 
                agent_temp_memory, agent_temp_pos, map_temp_memory, map_temp_pos, ego_temp_memory, ego_temp_pos,
                agent_pos_cls_embed, map_pts_pos_embed, map_ins_pos_embed, ego_pos_embed, attn_mask, temp_attn_masks=temp_attn_masks)
            # task heads
            # detection 
            agent_cls = cls_branches[lid](agent_query) 
            intermediate_agent_class.append(agent_cls)
            if self.refine:
                tmp = reg_branches[lid](agent_query + agent_pos_embed) 
                tmp[..., 0:2] = tmp[..., 0:2] + cur_agent_pred[..., 0:2]
                tmp[..., 4:5] = tmp[..., 4:5] + cur_agent_pred[..., 4:5]
                cur_agent_pred = tmp
            else:
                cur_agent_pred = reg_branches[lid](agent_query + agent_pos_embed)
            agent_ref = cur_agent_pred[..., [0,1,4]]
            intermediate_agent_coords_bev.append(agent_ref[..., :2])
            intermediate_agent_coords.append(cur_agent_pred)
            # motion prediction 
            agent_query_mode = agent_query.unsqueeze(2) + mode_query.unsqueeze(1)
            agent_traj_ref = traj_branches[lid](agent_query_mode).view(agent_query_mode.size(0), agent_query_mode.size(1), agent_query_mode.size(2), 6, 2)
            agent_traj_cls = traj_cls_branches[lid](agent_query_mode)
            intermediate_agent_traj_cls.append(agent_traj_cls)
            intermediate_agent_traj_coords.append(agent_traj_ref)
            # online mapping
            #########################################################################
            # project-2 
            if self.refine: # coarse-to-fine optimization
                # TODO-7 get map_pts_coord
                # input: map_query + map_pts_pos_embed
                # map_pts_coord_refine = map_reg_branches[lid](input)  # shape -> map_pts_coord.shape
                # map_pts_coord = map_pts_coord_refine + map_pts_coord
                pass
            else:
                # map_pts_coord = map_reg_branches[lid](input)  # shape -> map_pts_coord.shape
                pass
            #########################################################################
            _, map_ref  = map_transform_box(map_pts_coord.unsqueeze(0))
            intermediate_map_coords.append(map_pts_coord)
            map_query = map_query.view(map_pts_coord.shape[0], map_pts_coord.shape[1], map_pts_coord.shape[2], self.embed_dims)
            map_query = map_query.mean(-2)
            map_cls = map_cls_branches[lid](map_query)
            intermediate_map_class.append(map_cls)
            # planning
            if self.refine: 
                #####################################################################
                # project-3
                # TODO-8 get planning result
                # 替换此处代码
                # input = ego_query + ego_pos_embed  # [1, 1, 768]
                if ego_traj_branches_fix_dist is not None:
                    # 替换此处代码
                    # ego_traj_ref_fix_dist_refine = ego_traj_branches_fix_dist[lid](input) # shape: [1, 1, 20] -> [1, 1, 20, 1]
                    # ego_traj_ref_fix_dist = ego_traj_ref_fix_dist_refine + ego_traj_ref_fix_dist
                    # ego_traj_ref_fix_dist: [1, 1, 20, 1]
                    pass
                # 替换此处代码
                # ego_traj_ref_fix_time_refine = ego_traj_branches_fix_time[lid](input)  # [1, 1, 60] -> [bs, ego_query.shape[1], 30, 2]
                # ego_traj_ref_fix_time = ego_traj_ref_fix_time_refine + ego_traj_ref_fix_time
                # ego_traj_ref_fix_time: [1, 1, 30, 2]
            else:
                # 替换此处代码
                # ego_traj_ref_fix_dist = ego_traj_ref_fix_dist_refine
                # ego_traj_ref_fix_time = ego_traj_ref_fix_time_refine
                pass
                ######################################################################
            ego_traj_cls = ego_traj_cls_branches[lid](ego_query + ego_pos_embed) if ego_traj_cls_branches is not None else None
            intermediate_ego_traj_fix_time.append(ego_traj_ref_fix_time)
            intermediate_ego_traj_fix_dist.append(ego_traj_ref_fix_dist)
            intermediate_ego_class.append(ego_traj_cls)

        return agent_query, map_query, ego_query, \
            (intermediate_agent_traj_coords,
            intermediate_agent_traj_cls,
            intermediate_agent_coords_bev,
            intermediate_agent_coords,
            intermediate_agent_class,
            intermediate_map_coords,
            intermediate_map_class,
            intermediate_ego_traj_fix_time,
            intermediate_ego_traj_fix_dist,
            intermediate_ego_class
            )
            
@TRANSFORMER.register_module()
class DriveTransformerWrapper(BaseModule):
    def __init__(self, decoder, embed_dims, **kwargs):
        super(DriveTransformerWrapper, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        self.fp16_enabled = False
        
    def init_ln(self, m):
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            try:
                m.init_weight()
            except AttributeError:
                m.init_weights()
        self.apply(self.init_ln)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DriveTransformerPreDecoder(TransformerLayerSequence):
    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 **kwargs):
        super(DriveTransformerPreDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.fp16_enabled = False

    @auto_fp16()
    def forward(
            self, 
            query,
            img_feats,
            query_pos,
            pos_embed,
        ):
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query,
                key=img_feats,
                value=img_feats,
                query_pos=query_pos,
                key_pos=pos_embed,
            )
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)
                
        if self.return_intermediate and len(self.layers) > 1:
            return torch.stack(intermediate)
        return query
    