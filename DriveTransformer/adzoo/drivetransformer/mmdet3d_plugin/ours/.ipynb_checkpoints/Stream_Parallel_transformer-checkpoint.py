import warnings
import copy
import torch
import numpy as np
import torch.nn as nn
from torch.nn import ModuleList
from mmcv.models import build_norm_layer, xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.models.backbones.base_module import BaseModule
from mmcv.models.utils.builder import TRANSFORMER
from mmcv.models.bricks.drop import build_dropout
from mmcv.models.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.models.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.utils import deprecated_api_warning, ConfigDict


from .misc import (pos2posemb, agent_map_ego_split, map_transform_box)
import torch.utils.checkpoint as cp
from mmcv.utils import force_fp32, auto_fp16
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER.register_module()
class PETRTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

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

        # assert set(operation_order) & {
        #     'self_attn', 'norm', 'ffn', 'cross_attn'} == \
        #     set(operation_order), f'The operation_order of' \
        #     f' {self.__class__.__name__} should ' \
        #     f'contains all four operation type ' \
        #     f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

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

        # self.norms = ModuleList()
        # num_norms = operation_order.count('norm')
        # for _ in range(num_norms):
        #     self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
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
        """Forward function for `TransformerDecoderLayer`.

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
        #identity = query
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
                
            elif layer == 'norm':
                #query = self.norms[norm_index](query)
                continue
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    None,#identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                #identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, None)# identity if self.pre_norm else None)
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
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
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
class OneBigDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

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
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            {'self_attn', 'norm', 'ffn', 'cross_attn'}, f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn') + operation_order.count('temp_attn')
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
        self.attentions = ModuleList()
        self.temp_attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            # build agent, map and ego temp attention respectively
            if operation_name == 'temp_attn':
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                for _ in range(3):
                    attention = build_attention(attn_cfgs[index])
                    attention.operation_name = operation_name
                    self.temp_attentions.append(attention)
                index += 1
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
        # self.norms = ModuleList()
        # self.temp_norms = ModuleList()
        # num_norms = operation_order.count('norm')
        # for _ in range(num_norms):
        #     self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        # for _ in range(3):
        #     self.temp_norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp
        self.fp16_enabled = False

    @auto_fp16()
    def temp_attns(
            self,
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
            temp_attn_masks,
        ):
        agent_query, map_query, ego_query = agent_map_ego_split(query, agent_query_num, map_query_num, dim=1)#0)
        agent_pos, map_pos, ego_pos = agent_map_ego_split(pos, agent_query_num, map_query_num, dim=1)#0)
        queries = [agent_query, map_query, ego_query]
        #identities = queries
        poses = [agent_pos, map_pos, ego_pos]
        temp_memories = [agent_temp_memory, map_temp_memory, ego_temp_memory]
        temp_poses = [agent_temp_pos, map_temp_pos, ego_temp_pos]
        out_queries = []
        
        #temp_attn_masks = ~(torch.cat([torch.ones((temp_attn_masks.shape[0], 1, 1)).to(device=temp_attn_masks.device, dtype=temp_attn_masks.dtype), temp_attn_masks], dim=1).bool()).squeeze(-1)
        temp_attn_masks = ~(temp_attn_masks.bool()).squeeze(-1)
        temp_attn_masks = [
            torch.cat([torch.zeros((agent_query.shape[0], agent_query.shape[1], 1)).bool().to(agent_query.device), temp_attn_masks.unsqueeze(1).expand(-1, agent_query.shape[1], -1).repeat_interleave(repeats=self.agent_num_propagated, dim=-1)], dim=-1),
            torch.cat([torch.zeros((map_query.shape[0], map_query.shape[1], 1)).bool().to(agent_query.device), temp_attn_masks.unsqueeze(1).expand(-1, map_query.shape[1], -1).repeat_interleave(repeats=self.map_num_propagated, dim=-1)], dim=-1),
            torch.cat([torch.zeros((ego_query.shape[0], ego_query.shape[1], 1)).bool().to(agent_query.device),temp_attn_masks.unsqueeze(1).expand(-1, ego_query.shape[1], -1)], dim=-1)
        ]
        
        for i in range(3):
            out_queries.append(
                self.temp_attentions[i](
                queries[i],
                torch.cat([torch.zeros((temp_memories[i].shape[0], 1, temp_memories[i].shape[2])).to(temp_memories[i].device), temp_memories[i]], dim=1), #torch.cat([queries[i], temp_memories[i]], dim=1),
                torch.cat([torch.zeros((temp_memories[i].shape[0], 1, temp_memories[i].shape[2])).to(temp_memories[i].device), temp_memories[i]], dim=1), #torch.cat([queries[i], temp_memories[i]], dim=1),
                None, #identities[i] if self.pre_norm else None,
                query_pos=poses[i],
                key_pos=torch.cat([torch.zeros((temp_poses[i].shape[0], 1, temp_poses[i].shape[2])).to(temp_poses[i].device), temp_poses[i]], dim=1), #torch.cat([poses[i], temp_poses[i]], dim=1),
                attn_mask=temp_attn_masks[i]
                )
            )
        return torch.cat(out_queries, dim=1)#0)
    
    @auto_fp16()
    def _forward(self,
                agent_query,
                map_query,
                ego_query,
                img_feats,
                pos_embed,
                agent_temp_memory,
                agent_temp_pos,
                map_temp_memory,
                map_temp_pos,
                ego_temp_memory,
                ego_temp_pos,
                agent_pos_embed,
                map_pts_pos_embed,
                map_ins_pos_embed,
                ego_pos_embed,
                attn_masks=None,
                temp_attn_masks=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

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

        agent_query_num = agent_query.shape[1]
        map_query_num = map_query.shape[1]

        query = torch.cat([agent_query, map_query, ego_query], dim=1)
        #identity = query
        pos = torch.cat([agent_pos_embed, map_ins_pos_embed, ego_pos_embed], dim=1)
        for layer in self.operation_order:
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    None,#identity if self.pre_norm else None,
                    query_pos=pos,
                    key_pos=pos,
                    attn_mask=attn_masks[attn_index],
                    **kwargs)
                attn_index += 1
                #identity = query

            elif layer == 'norm':
                #query = self.norms[norm_index](query)
                continue
                norm_index += 1
            
            elif layer == 'temp_attn':
                query = self.temp_attns(
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

            elif layer == 'cross_attn':
                agent_query, map_query, ego_query = agent_map_ego_split(query, agent_query_num, map_query_num, dim=1)#0)
                #map_query = map_query.unsqueeze(1).expand(-1, self.map_pts_per_vec,-1,-1).flatten(0,1)
                map_query = map_query.unsqueeze(2).expand(-1, -1, self.map_pts_per_vec, -1).flatten(1, 2)
                identity = torch.cat([agent_query, map_query, ego_query], dim=1)
                agent_query = self.cross_w_q[0](agent_query+agent_pos_embed)
                if map_pts_pos_embed is not None:
                    map_query = self.cross_w_q[1](map_query+map_pts_pos_embed) 
                ego_query = self.cross_w_q[2](ego_query+ego_pos_embed)
                #pos = torch.cat([agent_pos_embed, map_pts_pos_embed, ego_pos_embed], dim=1)
                query = self.attentions[attn_index](
                    torch.cat([agent_query, map_query, ego_query], dim=1),
                    img_feats,
                    img_feats,
                    identity,
                    query_pos=None,#pos,
                    key_pos=pos_embed,
                    # attn_mask=attn_masks[attn_index],
                    **kwargs)
                attn_index += 1
                #identity = query
            elif layer == 'ffn':
                # query = self.ffns[ffn_index](
                #     query, None)#identity if self.pre_norm else None)
                
                # after cross attn
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
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(self._forward, *args, **kwargs)

        else:
            x = self._forward(*args, **kwargs)
        return x

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MixUpDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """
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
        
        super(MixUpDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fut_mode = fut_mode
        self.agent_num_query = agent_num_query
        self.map_num_query = map_num_query
        self.map_num_pts_per_vec = map_num_pts_per_vec
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.refine = refine

    #@force_fp32()
    @auto_fp16()
    def forward(
            self,
            agent_query,
            map_query,
            ego_query,
            img_feats,
            pos_embed,
            agent_temp_memory,
            agent_temp_pos,
            map_temp_memory,
            map_temp_pos,
            ego_temp_memory,
            ego_temp_pos,
            agent_ref,
            map_ref,
            map_pts_coord,
            ego_ref,
            agent_traj_ref,
            ego_traj_ref_fix_time,
            ego_traj_ref_fix_dist,
            mode_query=None,
            agent_cls=None,
            map_cls=None,
            agent_mode_fus=None,
            agent_traj_fus=None,
            map_query_fus=None,
            map_pts_fus=None,
            ego_mode_fus=None,
            agent_ref_embedding=None,
            agent_cls_embedding=None,
            map_ref_embedding=None,
            map_cls_embedding=None,
            ego_pos_embedding=None,
            ego_traj_ref_fix_time_embedding=None,
            ego_traj_ref_fix_dist_embedding=None,
            reg_branches=None,
            traj_branches=None,
            traj_cls_branches=None,
            cls_branches=None,
            map_reg_branches=None,
            map_cls_branches=None,
            attn_mask=None,
            temp_attn_masks=None,
            ego_traj_branches_fix_dist=None,
            ego_traj_branches_fix_time=None,
            ego_traj_cls_branches=None,
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
        agent_ref = agent_ref[..., [0, 1, 4]]
        for lid, layer in enumerate(self.layers):
            # expand agent & ego traj refs to 3d, z axis 0
            # [bs, q, fut_ts, dim] -> [bs, q, dim]
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
                
            agent_query, map_query, ego_query = layer(agent_query, map_query, ego_query, img_feats, pos_embed, 
                agent_temp_memory, agent_temp_pos, map_temp_memory, map_temp_pos, ego_temp_memory, ego_temp_pos,
                agent_pos_cls_embed, map_pts_pos_embed, map_ins_pos_embed, ego_pos_embed, attn_mask, temp_attn_masks=temp_attn_masks)

            agent_cls = cls_branches[lid](agent_query)
            intermediate_agent_class.append(agent_cls)
            if self.refine:
                tmp = reg_branches[lid](agent_query + agent_pos_embed) 
                tmp[..., 0:2] = tmp[..., 0:2] + cur_agent_pred[..., 0:2]
                tmp[..., 4:5] = tmp[..., 4:5] + cur_agent_pred[..., 2:3] if lid == 0 else tmp[..., 4:5] + cur_agent_pred[..., 4:5]
                cur_agent_pred = tmp
            else:
                cur_agent_pred = reg_branches[lid](agent_query + agent_pos_embed)
            agent_ref = cur_agent_pred[..., [0,1,4]]
            intermediate_agent_coords_bev.append(agent_ref[..., :2])
            intermediate_agent_coords.append(cur_agent_pred)

            agent_query_mode = agent_query.unsqueeze(2) + mode_query.unsqueeze(1)
            agent_traj_ref = traj_branches[lid](agent_query_mode).view(agent_query_mode.size(0), agent_query_mode.size(1), agent_query_mode.size(2), 6, 2)
            agent_traj_cls = traj_cls_branches[lid](agent_query_mode)
            intermediate_agent_traj_cls.append(agent_traj_cls)
            intermediate_agent_traj_coords.append(agent_traj_ref)
            if self.refine:
                map_pts_coord = map_reg_branches[lid](map_query + map_pts_pos_embed).view(*map_pts_coord.shape) + map_pts_coord
            else:
                map_pts_coord = map_reg_branches[lid](map_query + map_pts_pos_embed).view(*map_pts_coord.shape)
            _, map_ref  = map_transform_box(map_pts_coord.unsqueeze(0))

            intermediate_map_coords.append(map_pts_coord)
            map_query = map_query.view(map_pts_coord.shape[0], map_pts_coord.shape[1], map_pts_coord.shape[2], self.embed_dims)
            map_query = map_query.mean(-2)
            map_cls = map_cls_branches[lid](map_query)
            intermediate_map_class.append(map_cls)

            if self.refine: 
                ego_traj_ref_fix_dist = ego_traj_branches_fix_dist[lid](ego_query + ego_pos_embed).unsqueeze(-1) + ego_traj_ref_fix_dist if ego_traj_branches_fix_dist is not None else None
                ego_traj_ref_fix_time = ego_traj_branches_fix_time[lid](ego_query + ego_pos_embed).reshape(bs,ego_query.shape[1], -1, 2) + ego_traj_ref_fix_time
            else:
                ego_traj_ref_fix_dist = ego_traj_branches_fix_dist[lid](ego_query + ego_pos_embed).unsqueeze(-1) if ego_traj_branches_fix_dist is not None else None
                ego_traj_ref_fix_time = ego_traj_branches_fix_time[lid](ego_query + ego_pos_embed).reshape(bs,ego_query.shape[1], -1, 2)
                
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
class WrapperAroundMixUpDecoder(BaseModule):
    def __init__(self, decoder, embed_dims, **kwargs):
        super(WrapperAroundMixUpDecoder, self).__init__(**kwargs)
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
            # if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
            #         or isinstance(m, CustomMSDeformableAttention):
            try:
                m.init_weight()
            except AttributeError:
                m.init_weights()
        self.apply(self.init_ln)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class StreamTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=None,
                 return_intermediate=False,
                 **kwargs):

        super(StreamTransformerDecoder, self).__init__(*args, **kwargs)
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
            # temp_memory,
            # temp_pos
        ):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].,
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query,
                key=img_feats,
                value=img_feats,
                query_pos=query_pos,
                key_pos=pos_embed,
                # temp_memory=temp_memory,
                # temp_pos=temp_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)
                
        if self.return_intermediate and len(self.layers) > 1:
            return torch.stack(intermediate)
        return query
    