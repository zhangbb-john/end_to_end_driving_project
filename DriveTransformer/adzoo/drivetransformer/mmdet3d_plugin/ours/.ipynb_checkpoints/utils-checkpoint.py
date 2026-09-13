import torch
import math
import numpy as np

from torch.cuda.amp import autocast as autocast
from mmcv.models.backbones.base_module import BaseModule
from mmcv.utils import force_fp32, auto_fp16
import torch.nn as nn
from mmcv.core.bbox.transforms import bbox_xyxy_to_cxcywh

@torch.no_grad()
def locations(hw, device, stride, pad_h, pad_w):
    """
    Arguments:
        features:  (N, C, H, W)
    Return:
        locations:  (H, W, 2)
    """
    h, w = hw
    with autocast(enabled=False):
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.reshape(h, w, 2)
        return locations

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    if isinstance(memory,np.ndarray):
        prev_exist = prev_exist.clone().detach().cpu().numpy()
    return memory * prev_exist   

def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    with autocast(enabled=False):
        reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
        if reverse:
            matrix = egopose.inverse()
        else:
            matrix = egopose
        if not translation:
            matrix[..., :3, 3] = 0.0
        reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    with autocast(enabled=False):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return pos_x

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    with autocast(enabled=False):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    with autocast(enabled=False):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_z = pos[..., 2, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def pos2posemb(pos, num_pos_feats=128, temperature=10000):
    with autocast(enabled=False):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_tmp = pos[..., None] / dim_t
        posemb = torch.stack((pos_tmp[..., 0::2].sin(), pos_tmp[..., 1::2].cos()), dim=-1).flatten(-2).flatten(-2)
    return posemb

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    with autocast(enabled=False):
        encoding = [tensor] if include_input else []
        frequency_bands = None
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions,
                dtype=tensor.dtype,
                device=tensor.device,
            )

        for freq in frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat

def agent_map_split(x, num_agent, dim=1):
    """Split the input tensor into agent and map part.
    Args:
        x (Tensor): The input tensor to be split.
        num_agent (int): The number of agent part.
        num_map (int): The number of map part.
    Returns:
        tuple[Tensor]: The agent and map part of input
            tensor.
    """
    assert x.dim() == 3
    assert dim == 1 or dim == 0
    if dim == 1:
        agent = x[:, :num_agent]
        map = x[:, num_agent:]
    elif dim == 0:
        agent = x[:num_agent]
        map = x[num_agent:]
    return agent, map

def agent_map_ego_split(x, num_agent, num_map, dim=1):
    assert x.dim() == 3
    assert dim == 1 or dim == 0;
    if dim == 1:
        agent = x[:, :num_agent]
        map = x[:, num_agent:num_map+num_agent]
        ego = x[:, num_map+num_agent:]
    elif dim == 0:
        agent = x[:num_agent]
        map = x[num_agent:num_map+num_agent]
        ego = x[num_map+num_agent:]
    return agent, map, ego

@torch.no_grad()
def agent_self_to_ego(coords, dir, trans):
    # rotation angle from y+ to dir
    with autocast(enabled=False):
        theta = (torch.atan2(dir[..., 1], dir[..., 0]) + math.pi * 1.5).unsqueeze(-1)
        rot_matrix = torch.cat([torch.cos(theta), torch.sin(theta), -torch.sin(theta), torch.cos(theta)],dim=-1).view(theta.size(0), theta.size(1),theta.size(2), 2, 2)
        return (rot_matrix @ coords.unsqueeze(-1)).squeeze(-1) + trans
    



def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)
    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.SiLU(inplace=True)]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.LayerNorm(mlp_channels[k]), nn.SiLU(inplace=True)])
            c_in = mlp_channels[k]
    return nn.Sequential(*layers)


def map_transform_box(pts, y_first=False):
    """
    Converting the points set into bounding box.

    Args:
        pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
    Returns:
        The bbox [cx, cy, w, h] transformed from points.
    """
    with torch.no_grad():
        pts_y = pts[:, :, :, :, 0] if y_first else pts[:, :, :, :, 1]
        pts_x = pts[:, :, :, :, 1] if y_first else pts[:, :, :, :, 0]
        xmin = pts_x.min(dim=3, keepdim=True)[0]
        xmax = pts_x.max(dim=3, keepdim=True)[0]
        ymin = pts_y.min(dim=3, keepdim=True)[0]
        ymax = pts_y.max(dim=3, keepdim=True)[0]
        bbox = torch.cat([xmin, ymin, xmax, ymax], dim=3)
        bbox = bbox_xyxy_to_cxcywh(bbox)
        mid_index = pts.shape[-2]//2
        if pts.shape[-2] % 2 == 0:
            ref_points = (pts[..., mid_index, :] + pts[..., mid_index-1, :])/2.0
        else:
            ref_points = pts[..., mid_index, :]
    return bbox.squeeze(0), ref_points.squeeze(0)


class PointNetPolylineEncoder(BaseModule):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

        self.fp16_enabled = False
        self.apply(self._init_weights)

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
        expand_shape = list(x.shape)
        for i in range(len(expand_shape)):
            if i != (len(expand_shape)-2):
                expand_shape[i] = -1

        x = self.pre_mlps(x)
        # max-pooling
        pooled_feature = x.max(dim=-2)[0]
        x = torch.cat((x, pooled_feature[..., None, :].expand(*expand_shape)), dim=-1)

        # mlp
        x = self.mlps(x)

        # max-pooling
        x = x.max(dim=-2)[0]
        
        # out-mlp 
        if self.out_mlps is not None:
            x = self.out_mlps(x)
        return x