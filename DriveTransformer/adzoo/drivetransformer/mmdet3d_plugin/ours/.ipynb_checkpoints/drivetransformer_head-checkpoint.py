import copy
from math import pi, cos, sin, tan
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from mmcv.models import HEADS, build_loss 
from mmcv.utils import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.core import build_assigner, build_sampler
from mmcv.core.bbox import build_bbox_coder
from mmcv.models.utils.transformer import inverse_sigmoid
from mmcv.models import Linear, bias_init_with_prob, xavier_init
from mmcv.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.models.bricks.transformer import build_transformer_layer_sequence
from mmcv.models.utils import build_transformer
from mmcv.core.bbox.util import normalize_bbox
from adzoo.drivetransformer.mmdet3d_plugin.datasets.map_utils.normalization import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)

from .utils import (
    locations, memory_refresh, transform_reference_points, pos2posemb, pos2posemb1d, pos2posemb2d, pos2posemb3d, nerf_positional_encoding, topk_gather, agent_map_split, PointNetPolylineEncoder, map_transform_box

)
from mmcv.models.backbones.base_module import BaseModule
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes



    

@HEADS.register_module()
class DriveTransformerlHead(BaseModule):
    """
    Head of DriveTransformer model. 
    """    
    def __init__(
        self,
        # Hyper-Parameter
        ## Model Size
        ego_lcf_feat_idx=None,
        img_stride=16,
        embed_dims=256,
        num_reg_fcs=2,
        num_cls_fcs=2,
        memory_len_frame=8,
        agent_num_propagated=50,
        map_num_propagated=50,
        ## Det & Motion
        agent_num_query=100,
        agent_num_query_sifted=100,
        fut_ts=6, ## Number of Points
        fut_mode=6,
        fut_ego_mode=6,
        num_classes=10,
        code_size=10,
        ## Online Mapping
        map_num_query=100,
        map_num_query_sifted=100,
        map_num_classes=3,
        map_num_pts_per_vec=4,
        map_num_pts_per_gt_vec=4,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        ## Optimization
        sync_cls_avg_factor=True,
        with_box_refine=True,
        ## 3D PE
        LID=True,
        with_ego_pos=True,
        position_range=[-65, -65, -8.0, 65, 65, 8.0],
        depth_start=1,
        depth_step=0.8,
        depth_num=64,
        # Layers
        ## InitLayer
        agent_prep_decoder=None,
        map_prep_decoder=None,
        ## Major Layer
        transformer=None,
        # Losses
        ## Det Loss
        loss_cls=None,
        loss_bbox=None,
        bbox_coder=None,
        code_weights=None,
        ## Motion Loss
        loss_traj=None,
        loss_traj_cls=None,
        ## Online Mapping Loss
        map_bbox_coder=None,
        loss_map_cls=None,
        loss_map_pts=None,
        loss_map_dir=None,
        ## Planning Loss
        loss_plan_reg_fix_time=None,
        loss_plan_reg_fix_dist=None,
        loss_plan_cls=None,
        ## Cfg
        train_cfg=None,
        test_cfg=None,
        ## Bench2Drive
        ego_command_dim=3,
        fut_ts_ego_fix_dist=None,
        fut_ts_ego_fix_time=None,
        fut_ego_fix_dist=False,
    ):
    
        super().__init__()
        self.fp16_enabled = False
        
        # Hyper-Parameter
        ## Model Size
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.img_stride = img_stride
        self.embed_dims = embed_dims
        self.memory_len_frame = memory_len_frame
        self.agent_num_propagated = agent_num_propagated
        self.map_num_propagated = map_num_propagated
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        ## Det & Motion
        self.agent_num_query = agent_num_query
        self.agent_num_query_sifted = agent_num_query_sifted
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.fut_ego_mode = fut_ego_mode
        self.ego_multi_modal = (fut_ego_mode > 1)
        self.num_classes = num_classes
        self.code_size = code_size
        self.fut_ts_ego_fix_dist = fut_ts_ego_fix_dist
        self.fut_ts_ego_fix_time = fut_ts_ego_fix_time
        self.fut_ego_fix_dist = fut_ego_fix_dist
        
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.traj_bg_cls_weight = 0
        if train_cfg is not None:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        
        ## Online Mapping
        self.map_num_query = map_num_query
        self.map_num_query_sifted = map_num_query_sifted
        self.map_num_classes = map_num_classes
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
               
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        self.map_dir_interval = map_dir_interval
        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1
        self.map_code_size = map_code_size
        self.map_code_weights = map_code_weights
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        if train_cfg is not None:
            map_assigner = train_cfg['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'
            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
            
        
        ## 3D PE
        self.LID = LID
        self.depth_start = depth_start
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.position_range = nn.Parameter(torch.tensor(position_range), requires_grad=False)
        self.with_ego_pos = with_ego_pos
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        self.register_buffer('coords_d', coords_d)
        ## Optimization
        self.with_box_refine = with_box_refine
        
        # Losses
        ## Det Loss
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.traj_num_cls = 1 ## Different Query
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)
        
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight
        
        ## Online Mapping Loss
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None:
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        ## Planning Loss
        self.loss_plan_reg_fix_dist = build_loss(loss_plan_reg_fix_dist)
        self.loss_plan_reg_fix_time = build_loss(loss_plan_reg_fix_time)
        self.loss_plan_cls = build_loss(loss_plan_cls)
        ## Bench2Drive
        self.ego_command_dim = ego_command_dim
        self.num_query = self.agent_num_query + self.map_num_query
        # Module
        ## Det
        self.agent_query = nn.Embedding(self.agent_num_query, self.embed_dims) # Init Feature
        self.agent_query.requires_grad_(False)
        self.agent_reference_points = nn.Embedding(self.agent_num_query, 3) # Init Ref Point
        self.agent_reference_points.requires_grad_(False)
        
        self.agent_cls_embedding = nn.Sequential(
            nn.Linear(self.num_classes, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        ) # Det Cls PE
        self.agent_ref_embedding = nn.Sequential(
            nn.Linear(int(self.embed_dims*3.0/2.0), self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        ) # Det Reg PE
        ## Planning
        if self.ego_multi_modal:
            self.register_buffer('anchor_ref', self.prepare_anchor_ref()) # 6 mode anchor
            self.ego_mode_embedding = nn.Embedding(self.fut_ego_mode, self.embed_dims)
            self.mode_mlp = PointNetPolylineEncoder(in_channels=self.embed_dims, hidden_dim=self.embed_dims, num_layers=3, out_channels=self.embed_dims) # mode anchor to embedding
            
        self.mode_embedding = nn.Embedding(self.fut_mode, self.embed_dims)
        ## Online Mapping
        self.map_query = nn.Embedding(self.map_num_query, self.embed_dims)
        self.map_query.requires_grad_(False)       
        self.map_reference_points = nn.Embedding(self.map_num_query, 2)
        self.map_reference_points.requires_grad_(False)
        self.map_cls_embedding = nn.Sequential(
            nn.Linear(self.map_num_classes, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        ) # Map Cls PE
        self.map_ref_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        ) # Map Reg PE
        ## Planning
        self.ego_lcf_encoder = nn.Sequential(
            nn.Linear(len(self.ego_lcf_feat_idx) + 2*2 + self.ego_command_dim, self.embed_dims, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.ego_traj_ref_fix_time_embedding = nn.Sequential(
            nn.Linear(64 * self.fut_ts_ego_fix_time, self.embed_dims, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.ego_traj_ref_fix_dist_embedding = nn.Sequential(
            nn.Linear(32 * self.fut_ts_ego_fix_dist, self.embed_dims, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        ) if self.fut_ego_fix_dist else None
        
        ## 3D PE
        self.featurized_pe = MLN(self.embed_dims, self.embed_dims)
        self.spatial_alignment = MLN(8, self.embed_dims)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims)
        )
        self.img_position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims),
                nn.SiLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.SiLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180, self.embed_dims)
            self.ego_pose_memory = MLN(180, self.embed_dims)

        # Layers
        ## InitLayer
        self.agent_prep_decoder = agent_prep_decoder
        self.map_prep_decoder = map_prep_decoder
        self.agent_prep_decoder = build_transformer_layer_sequence(agent_prep_decoder)
        self.map_prep_decoder = build_transformer_layer_sequence(map_prep_decoder)

        ## Major Layer
        self.transformer = build_transformer(transformer)
        self.init_output_head()
        self.reset_memory()
        self.pseudo_map_instance = None
        self.pseudo_agent_instance = None
        self.img_patch_location = None
        

    def init_output_head(self):
        """
        Build heads for each task. 
        """          
        cls_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_cls_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.SiLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels, bias=False))
        cls_branch = nn.Sequential(*cls_branch)
        cls_branch.apply(self.xavier_uniform_linear)

        reg_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.SiLU(inplace=True))
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)
        reg_branch.apply(self.xavier_uniform_linear)

        traj_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims, self.embed_dims))
            traj_branch.append(nn.SiLU(inplace=True))
        traj_branch.append(Linear(self.embed_dims, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)
        traj_branch.apply(self.xavier_uniform_linear)

        traj_cls_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_cls_fcs):
            traj_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.SiLU(inplace=True))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(Linear(self.embed_dims, self.traj_num_cls, bias=False))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)
        traj_cls_branch.apply(self.xavier_uniform_linear)

        map_cls_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_cls_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.SiLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels, bias=False))
        map_cls_branch = nn.Sequential(*map_cls_branch)
        map_cls_branch.apply(self.xavier_uniform_linear)

        map_reg_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.SiLU(inplace=True))
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)
        map_reg_branch.apply(self.xavier_uniform_linear)
        
        if self.fut_ego_fix_dist:
            ego_traj_branch_fix_dist = [nn.LayerNorm(self.embed_dims)]
            for _ in range(self.num_reg_fcs):
                ego_traj_branch_fix_dist.append(Linear(self.embed_dims, self.embed_dims))
                ego_traj_branch_fix_dist.append(nn.SiLU(inplace=True))
            ego_traj_branch_fix_dist.append(Linear(self.embed_dims, self.fut_ts_ego_fix_dist))
            ego_traj_branch_fix_dist = nn.Sequential(*ego_traj_branch_fix_dist)
            ego_traj_branch_fix_dist.apply(self.xavier_uniform_linear)  
        
        ego_traj_branch_fix_time = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_reg_fcs):
            ego_traj_branch_fix_time.append(Linear(self.embed_dims, self.embed_dims))
            ego_traj_branch_fix_time.append(nn.SiLU(inplace=True))
        ego_traj_branch_fix_time.append(Linear(self.embed_dims, self.fut_ts_ego_fix_time*2))
        ego_traj_branch_fix_time = nn.Sequential(*ego_traj_branch_fix_time)
        ego_traj_branch_fix_time.apply(self.xavier_uniform_linear)  
           
        ego_traj_cls_branch = [nn.LayerNorm(self.embed_dims)]
        for _ in range(self.num_reg_fcs):
            ego_traj_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            ego_traj_cls_branch.append(nn.ReLU(inplace=True))
            ego_traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        ego_traj_cls_branch.append(Linear(self.embed_dims, 1))
        ego_traj_cls_branch = nn.Sequential(*ego_traj_cls_branch)
        ego_traj_cls_branch.apply(self.xavier_uniform_linear)
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
        num_mixed_up_layers = self.transformer.decoder.num_layers
        num_pred = ego_num_pred = motion_num_pred = map_num_pred = num_mixed_up_layers + 1
        self.cls_branches = _get_clones(cls_branch, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.traj_branches = _get_clones(traj_branch, motion_num_pred)
        self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
        self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
        self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        self.map_reg_branches[-1] = nn.Linear(self.embed_dims, self.map_code_size * self.map_num_pts_per_vec)
        self.ego_traj_branches_fix_dist = _get_clones(ego_traj_branch_fix_dist, ego_num_pred) if self.fut_ego_fix_dist else None
        self.ego_traj_branches_fix_time = _get_clones(ego_traj_branch_fix_time, ego_num_pred)
        self.ego_traj_cls_branches = _get_clones(ego_traj_cls_branch, ego_num_pred) if self.ego_multi_modal else None
    
    def xavier_uniform_linear(self, m):
        is_linear_layer = any([isinstance(m, nn.Linear), isinstance(m, nn.Conv2d), isinstance(m, nn.ConvTranspose2d)])
        if is_linear_layer:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_ln(self, m):
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None: 
                nn.init.constant_(m.bias, 0)
    
    def init_weights(self):
        #self.mode_mlp.apply(self.xavier_uniform_linear)
        self.agent_cls_embedding.apply(self.xavier_uniform_linear)
        self.map_cls_embedding.apply(self.xavier_uniform_linear)
        self.agent_ref_embedding.apply(self.xavier_uniform_linear)
        self.map_ref_embedding.apply(self.xavier_uniform_linear)
        self.featurized_pe.apply(self.xavier_uniform_linear)
        self.spatial_alignment.apply(self.xavier_uniform_linear)
        self.time_embedding.apply(self.xavier_uniform_linear)
        self.ego_pose_pe.apply(self.xavier_uniform_linear)
        self.ego_pose_memory.apply(self.xavier_uniform_linear)
        self.ego_lcf_encoder.apply(self.xavier_uniform_linear)
        self.img_position_encoder.apply(self.xavier_uniform_linear)
        self.apply(self.init_ln)
        num_grid_per_dim_agent = int(np.sqrt(self.agent_reference_points.weight.shape[0]))
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], steps=num_grid_per_dim_agent)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], steps=num_grid_per_dim_agent)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        with torch.no_grad():
            self.agent_reference_points.weight[..., 0] = x.flatten()
            self.agent_reference_points.weight[..., 1] = y.flatten()
            self.agent_reference_points.weight[..., 2] = 0.0
        nn.init.constant_(self.agent_query.weight, 0)
        
        num_grid_per_dim_map = int(np.sqrt(self.map_reference_points.weight.shape[0]))
        xs = torch.linspace(self.pc_range[0], self.pc_range[3], steps=num_grid_per_dim_map)
        ys = torch.linspace(self.pc_range[1], self.pc_range[4], steps=num_grid_per_dim_map)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        with torch.no_grad():
            self.map_reference_points.weight[..., 0] = x.flatten()
            self.map_reference_points.weight[..., 1] = y.flatten()
        nn.init.constant_(self.map_query.weight, 0)

    @force_fp32()
    def forward(self,
                img_feats,
                img_metas,
                ego_lcf_feat=None,
                ego_fut_cmd=None,
                ego_his_trajs=None,
                **data,
            ):
        
        # update the memory for current frame
        self.pre_update_memory(data)
        ## Img
        bs, num_cam, C, H, W = img_feats.shape
        dtype = img_feats.dtype
        ## PETR 3D Pos
        token_location = self.prepare_location(img_feats, data)
        num_tokens = num_cam * H * W
        img_feats = img_feats.permute(0, 1, 3, 4, 2).reshape(bs, num_tokens, C)
        img_pos_embed, cone = self.img_3d_position_embedding(data, token_location.clone(), img_metas)
        ## spatial_alignment in focal petr
        img_feats = self.spatial_alignment(img_feats, cone) # [B, NUM_TOKEN, D]
        img_pos_embed = self.featurized_pe(img_pos_embed, img_feats)
        ## Det & Motion
        agent_query = self.agent_query.weight.to(dtype).unsqueeze(0).expand(bs, -1, -1)
        agent_reference_points = self.agent_reference_points.weight.unsqueeze(0).repeat(bs, 1, 1)
        ## Online Mapping
        map_query = self.map_query.weight.unsqueeze(0).expand(bs, -1, -1)        
        map_reference_points = self.map_reference_points.weight.unsqueeze(0).repeat(bs, 1, 1)
        ## Temporal Alignment
        agent_query, map_query, \
            agent_temp_memory, map_temp_memory, \
                agent_temp_pos, map_temp_pos, ego_temp_pos, rec_ego_pose = self.temporal_alignment(agent_query, map_query)
        ## Init PE
        agent_pe = self.agent_ref_embedding(pos2posemb(agent_reference_points, self.embed_dims//2))
        ## Init Prediction
        agent_query = self.agent_prep_decoder(
            agent_query,
            img_feats,
            agent_pe,
            img_pos_embed,
        )
        # get preliminary reference points for detection
        agent_query = agent_query.float()
        agent_prep_class = self.cls_branches[-1](agent_query)
        agent_prep_ref = self.reg_branches[-1](agent_query + agent_pe)
        agent_prep_ref[..., 0:2] = agent_prep_ref[..., 0:2] + agent_reference_points[..., :2]
        agent_prep_ref[..., 4:5] = agent_prep_ref[..., 4:5] + agent_reference_points[..., 2:3]

        map_pe = self.map_ref_embedding(pos2posemb(map_reference_points, self.embed_dims//2))
        map_query = self.map_prep_decoder(
            map_query,
            img_feats,
            map_pe,
            img_pos_embed,
        )
        map_query = map_query.float()
        # get preliminary reference points for map
        map_prep_class = self.map_cls_branches[-1](map_query)
        map_prep_pts_coord = self.map_reg_branches[-1](map_query + map_pe)
        map_prep_pts_coord = map_prep_pts_coord.view(map_query.shape[0], map_query.shape[1], -1, 2) + map_reference_points.unsqueeze(-2)
        map_prep_box, map_prep_ref  = map_transform_box(map_prep_pts_coord.unsqueeze(0))
        # get preliminary reference points for motion
        mode_query = self.mode_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
        agent_query_mode = (agent_query.unsqueeze(2) + mode_query.unsqueeze(1))
        agent_prep_traj_ref = self.traj_branches[-1](agent_query_mode).view(bs, agent_query_mode.shape[1], agent_query_mode.shape[2], self.fut_ts, 2) # [bs, num*mode, fut_ts, 2]
        agent_prep_traj_cls = self.traj_cls_branches[-1](agent_query_mode)

        ## Planning
        ## ego_lcf_feat: (vx, vy, ax, ay, w, length, width, vel, steer)
        ego_query = self.ego_lcf_encoder(torch.cat([ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx], ego_his_trajs.flatten(-2, -1), ego_fut_cmd.squeeze(1)], dim=-1)) # [B,1,D]
        if len(ego_query.shape) == 2:
            ego_query = ego_query.unsqueeze(0)
            
        if self.ego_multi_modal:
            mode_anchor_ref = self.anchor_ref.unsqueeze(0).expand(bs, -1, -1, -1)
            ego_mode_query = self.mode_mlp(pos2posemb(mode_anchor_ref[...,0:2], self.embed_dims//2))  
            ego_query = (ego_query.unsqueeze(2) + ego_mode_query.unsqueeze(1)).flatten(1,2) # [B,N_mode,D]
        # get preliminary reference points for planning
        ego_ref = torch.zeros((ego_query.shape[0], ego_query.shape[1], 3),device=ego_query.device, dtype=ego_query.dtype) 
        ego_prep_traj_ref_fix_time = self.ego_traj_branches_fix_time[-1](ego_query).view(bs, ego_query.shape[1], self.fut_ts_ego_fix_time, 2)         
        ego_prep_traj_ref_fix_dist = self.ego_traj_branches_fix_dist[-1](ego_query).view(bs, ego_query.shape[1], self.fut_ts_ego_fix_dist, 1) if self.fut_ego_fix_dist else None
        ego_prep_traj_cls = self.traj_cls_branches[-1](ego_query) if self.ego_multi_modal else None
        # major transformer 
        agent_query, map_query, ego_query, results = self.transformer(
            agent_query, # [B, N_agent_query, D] ||| queries and position embeddings
            map_query, # [B, N_map_query, D]
            ego_query, # [B, N_ego_mode, D]
            img_feats, # [B, N_image_token,D]
            img_pos_embed, # [B, N_image_token,D]
            agent_temp_memory, # [B, L_memory * N_memory_agent_per_frame, D] ||| memorized queries and position embeddings
            agent_temp_pos, # [B, L_memory * N_memory_agent_per_frame, D]
            map_temp_memory, # [B, L_memory * N_memory_map_per_frame, D]
            map_temp_pos, # [B, L_memory * N_memory_map_per_frame, D]
            self.ego_memory_embedding, # [B, L_memory, D] 
            ego_temp_pos, # [B, L_memory,D]  ||| reference points
            agent_prep_ref, # [B, N_agent_query, C_box]
            map_prep_ref, # [B, N_map_query,2]
            map_prep_pts_coord,  # [B,N_map_query, N_pts_per_line, 2]
            ego_ref, #[B ,N_ego_mode,3]
            agent_prep_traj_ref, # [B,N_agent_query, N_mode,N_future, 2]
            ego_prep_traj_ref_fix_time, # [B,N_mode, N_future_ego_time, 2]
            ego_prep_traj_ref_fix_dist, # [B,N_mode, N_future_ego_dist, 2]
            mode_query=mode_query,  # [B, N_mode, D] 
            agent_cls=agent_prep_class, # [B, N_agent_query, N_object_type]
            map_cls=map_prep_class, # [B, N_map_query, N_map_type]
            agent_ref_embedding=self.agent_ref_embedding, # network layers and heads
            agent_cls_embedding=self.agent_cls_embedding,
            map_ref_embedding=self.map_ref_embedding,
            map_cls_embedding=self.map_cls_embedding,
            ego_pos_embedding=self.agent_ref_embedding,
            ego_traj_ref_fix_time_embedding=self.ego_traj_ref_fix_time_embedding,
            ego_traj_ref_fix_dist_embedding=self.ego_traj_ref_fix_dist_embedding,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            traj_branches=self.traj_branches,
            traj_cls_branches=self.traj_cls_branches,
            map_reg_branches=self.map_reg_branches,
            map_cls_branches=self.map_cls_branches,
            temp_attn_masks=self.memory_prev_exists,
            ego_traj_branches_fix_dist=self.ego_traj_branches_fix_dist,
            ego_traj_branches_fix_time=self.ego_traj_branches_fix_time,
            ego_traj_cls_branches=self.ego_traj_cls_branches,
        )
        # collect results
        agent_traj_coords, agent_traj_cls, agent_coords_bev, agent_coords, agent_class, map_pts_coords, map_class, \
        ego_traj_fix_time, ego_traj_fix_dist, ego_traj_cls = list(map(lambda x: torch.stack(x) if not (x is None or x[0] is None) else None, results)) #
        map_boxes, map_refs = map_transform_box(map_pts_coords)
        # add current feature to memory
        self.post_update_memory(data, rec_ego_pose, agent_class, agent_coords, \
                                map_class, map_refs[..., :2], agent_query, map_query, ego_query)
        # results from all layers(including initial)
        agent_traj_coords = torch.cat([agent_prep_traj_ref.unsqueeze(0), agent_traj_coords], dim=0)
        agent_traj_cls = torch.cat([agent_prep_traj_cls.unsqueeze(0), agent_traj_cls], dim=0)
        agent_coords = torch.cat([agent_prep_ref.unsqueeze(0), agent_coords], dim=0)
        agent_class = torch.cat([agent_prep_class.unsqueeze(0), agent_class], dim=0)
        map_coords = torch.cat([map_prep_box.unsqueeze(0), map_boxes], dim=0)
        map_pts_coords = torch.cat([map_prep_pts_coord.unsqueeze(0), map_pts_coords], dim=0)
        map_class = torch.cat([map_prep_class.unsqueeze(0), map_class], dim=0)
        ego_traj_fix_time = torch.cat([ego_prep_traj_ref_fix_time.unsqueeze(0), ego_traj_fix_time], dim=0)
        ego_traj_fix_dist = torch.cat([ego_prep_traj_ref_fix_dist.unsqueeze(0), ego_traj_fix_dist], dim=0) if ego_traj_fix_dist is not None else None
        ego_traj_cls = torch.cat([ego_prep_traj_cls.unsqueeze(0), ego_traj_cls], dim=0) if ego_traj_cls is not None else None
        outs = {
            'all_cls_scores': agent_class, # [N_layers, B, N_agent_query, N_object_type]
            'all_bbox_preds': agent_coords, # [N_layers, B, N_map_query, N_map_type]
            'all_traj_preds': agent_traj_coords.view(agent_traj_coords.size(0), agent_traj_coords.size(1), 
                                                     self.agent_num_query_sifted, self.fut_mode, self.fut_ts, 2).flatten(-2,-1), # [N_layers, B, N_agent_query, N_mode, N_future*2]
            'all_traj_cls_scores': agent_traj_cls.unsqueeze(-1).view(agent_traj_cls.size(0), agent_traj_cls.size(1), self.agent_num_query_sifted, self.fut_mode), # [N_layers, B, N_agent_query, N_mode]
            'map_all_cls_scores': map_class, # [N_layers, B, N_map_query, N_map_type]
            'map_all_bbox_preds': map_coords, # [N_layers, B, N_map_query, 4]
            'map_all_pts_preds': map_pts_coords, # [N_layers, B, N_map_query, N_pts_per_line, 2]
            'ego_fut_preds_fix_time': ego_traj_fix_time, # [N_layers, B, N_ego_mode, N_ego_future_time, 2]
            'ego_fut_preds_fix_dist': ego_traj_fix_dist, # [N_layers, B, N_ego_mode, N_ego_future_dist, 1] or none
            'ego_traj_cls_scores': ego_traj_cls, # [N_layers, B, N_ego_mode] or none
            'map_pre_cls_scores': map_prep_class, # [B, N_map_query, N_map_type]
            'map_pre_coord_preds': map_prep_box, # [B, N_map_query, 4]
            'map_pre_pts_coord_preds': map_prep_pts_coord, # [B, N_map_query, N_pts_per_line, 2]
            'agent_pre_cls_scores': agent_prep_class, # [B, N_agent_query, N_object_type]
            'agent_pre_coord_preds': agent_prep_ref, # [B, N_agent_query, C_box]
        }
        return outs
        
    def reset_memory(self):
        self.agent_memory_embedding = None
        self.map_memory_embedding = None
        self.ego_memory_embedding = None
        self.agent_memory_reference_point = None
        self.agent_memory_reference_yaw = None
        self.agent_memory_class = None
        self.map_memory_reference_point = None
        self.map_memory_class = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    @force_fp32()
    def img_3d_position_embedding(self, data, memory_centers, img_metas, topk_indexes=None):
        # get the 3D PE of image tokens
        eps = 1e-5
        H, W = memory_centers.shape[:2]
        B, N = data['lidar2img'].shape[:2]
        intrinsic = torch.stack([data['cam_intrinsic'][..., 0, 0], data['cam_intrinsic'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)
        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN
        memory_centers[..., 0] = memory_centers[..., 0] * self.pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * self.pad_h
        D = self.coords_d.shape[0]
        memory_centers = memory_centers.detach().view(1, H*W, 1, 2).repeat(B, N, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([memory_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        coords = coords.unsqueeze(-1)
        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(B*N, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.img_position_encoder(pos_embed)
        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone

    @auto_fp16()
    def prepare_location(self, img_feats, data):
        # prepare image token locations
        if self.img_patch_location is None:
            bs, n = img_feats.shape[:2]
            pad_h, pad_w = img_feats.shape[-2]*self.img_stride, img_feats.shape[-1]*self.img_stride
            self.pad_h = pad_h
            self.pad_w = pad_w
            self.img_patch_location = locations(img_feats.size()[-2:], img_feats.device, self.img_stride, pad_h, pad_w).detach()
        return self.img_patch_location
    
    @force_fp32()
    def prepare_anchor_ref(self):
        # get the reference points for ego planning. Same to the points in TrajPreprocess pipeline.
        x_coords = torch.linspace(start=0, end=self.pc_range[3], steps=self.fut_ts, 
                                device=self.pc_range.device).unsqueeze(1).unsqueeze(0)
        z_coords = 0 * torch.ones_like(x_coords, device=self.pc_range.device)
        ego_ref = []
        traj_theta = [pi/180*65, pi/180*40]
        for theta in traj_theta:
            y_coords = torch.clamp(tan(theta) * x_coords, min=0, max=self.pc_range[4])
            coords_po = torch.cat([x_coords, y_coords, z_coords], dim=-1)
            coords_ne = torch.cat([-x_coords, y_coords, z_coords], dim=-1)
            ego_ref.append(coords_po)
            ego_ref.append(coords_ne)
        # front ref point
        y_coords = torch.linspace(start=0, end=self.pc_range[4], steps=self.fut_ts, 
                                device=self.pc_range.device).unsqueeze(1).unsqueeze(0)
        coords = torch.cat([torch.zeros_like(y_coords), y_coords, z_coords], dim=-1)
        ego_ref.append(coords)
        y_coords = torch.linspace(start=0, end=0.25 * self.pc_range[4], steps=self.fut_ts, 
                                  device=self.pc_range.device).unsqueeze(1).unsqueeze(0)
        coords = torch.cat([torch.zeros_like(y_coords), y_coords, z_coords], dim=-1)
        ego_ref.append(coords)  
        ego_ref = torch.cat(ego_ref, dim=0)
        return ego_ref

    @force_fp32()
    def pre_update_memory(self, data):
        with torch.no_grad():
            x = data['prev_exists'].type(torch.float32)
            B = x.size(0)
            # update the memory at the beginning of forward.
            # memory length differs for different types of queries, but the stored number of history frames is the same
            # for map and agent embedding: memlen = frame * num_propagated
            # for ego: memlen = frame * fut_mode (all six modes)
            # for agent ref: memlen = frame * num_propagated * 2 (one for agent itself, one for agent's future position)
            # for map ref: memlen = frame * num_propagated * pts_per_vec
            if self.agent_memory_embedding is None: # init the memory with all zeros
                self.agent_memory_embedding = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, self.embed_dims)
                self.map_memory_embedding = x.new_zeros(B, self.memory_len_frame * self.map_num_propagated, self.embed_dims)
                self.ego_memory_embedding = x.new_zeros(B, self.memory_len_frame, self.embed_dims)
                self.agent_memory_class = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, self.num_classes)
                self.agent_memory_reference_point = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 3)
                self.agent_memory_reference_yaw = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 1)
                self.map_memory_reference_point = x.new_zeros(B, self.memory_len_frame * self.map_num_propagated, 3) 
                self.map_memory_class = x.new_zeros(B, self.memory_len_frame * self.map_num_propagated, self.map_num_classes) 
                self.memory_timestamp = np.zeros((B, self.memory_len_frame, 1),dtype=np.float64)
                self.memory_egopose = x.new_zeros(B, self.memory_len_frame, 4, 4)
                self.memory_velo = x.new_zeros(B, self.memory_len_frame * self.agent_num_propagated, 2)
                self.memory_prev_exists = x.new_zeros(B, self.memory_len_frame, 1)
            else: 
                # transpose from global to local
                self.memory_timestamp -= data['timestamp'].reshape(B,1,1)
                self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
                self.agent_memory_reference_point = transform_reference_points(self.agent_memory_reference_point, data['ego_pose_inv'], reverse=False)
                self.map_memory_reference_point = transform_reference_points(self.map_memory_reference_point, data['ego_pose_inv'], reverse=False)
                self.agent_memory_reference_yaw -= torch.arctan2(data['ego_pose'][:,1,0],data['ego_pose'][:,0,0]).unsqueeze(-1).unsqueeze(-1)
                # pop with FIFO policy, reset memory if meet a new sequence (prev_exists=False)
                self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len_frame], x)
                self.agent_memory_reference_point = memory_refresh(self.agent_memory_reference_point[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.map_memory_reference_point = memory_refresh(self.map_memory_reference_point[:, :self.memory_len_frame * self.map_num_propagated], x)
                self.agent_memory_embedding = memory_refresh(self.agent_memory_embedding[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.agent_memory_class = memory_refresh(self.agent_memory_class[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.agent_memory_reference_yaw = memory_refresh(self.agent_memory_reference_yaw[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.map_memory_embedding = memory_refresh(self.map_memory_embedding[:, :self.memory_len_frame * self.map_num_propagated], x)
                self.map_memory_class = memory_refresh(self.map_memory_class[:, :self.memory_len_frame * self.map_num_propagated], x)
                self.ego_memory_embedding = memory_refresh(self.ego_memory_embedding[:, :self.memory_len_frame], x)
                self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len_frame], x)
                self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len_frame * self.agent_num_propagated], x)
                self.memory_prev_exists = memory_refresh(self.memory_prev_exists[:, :self.memory_len_frame], x)
            self.memory_egopose  = self.memory_egopose + (1-x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)
    
    @force_fp32()
    def post_update_memory(
        self, 
        data, 
        rec_ego_pose,  # [B,1,4,4]
        agent_cls_scores,  # [N_layer,B,N_agent_query,N_type]
        agent_bbox_preds,  # [N_layer,B,N_agent_query,C]
        map_cls_scores, # [N_layer,B,N_map_query,N_map_type]
        map_ref_preds, # [N_layer,B,N_map_query,C]
        agent_memory, # [B,N_query,D]
        map_memory, # [B,N_map_query,D]
        ego_memory, # [B,N_ego_fut_mode,D]
    ):
        with torch.no_grad():
            # update the memory at the end of forward
            # for agent and map queries, only top-K confidence scores are kept
            bs = agent_cls_scores.shape[1]
            agent_ref_point = torch.cat(
                [agent_bbox_preds[..., :2], agent_bbox_preds[..., 4:5]], dim=-1)[-1]
            agent_ref_yaw = -torch.arctan2(agent_bbox_preds[-1,..., 6], agent_bbox_preds[-1,..., 7]).unsqueeze(-1)-torch.pi/2

            # z axis of map element is always 0
            map_ref_point = torch.cat(
                [map_ref_preds[..., :2], torch.zeros_like(map_ref_preds[..., 0:1])], dim=-1)[-1]
            rec_agent_velo = agent_bbox_preds[..., -2:][-1]
            rec_agent_memory = agent_memory
            rec_map_memory = map_memory
            rec_agent_score = agent_cls_scores[-1].sigmoid().topk(1, dim=-1).values
            rec_map_score = map_cls_scores[-1].sigmoid().topk(1, dim=-1).values
            rec_timestamp = np.zeros((rec_agent_score.shape[0],1,1), dtype=np.float64)

            _, agent_topk_indexes = torch.topk(rec_agent_score, self.agent_num_propagated, dim=1)
            rec_agent_score = topk_gather(rec_agent_score, agent_topk_indexes).detach()
            agent_ref_point = topk_gather(agent_ref_point, agent_topk_indexes).detach()
            agent_ref_yaw = topk_gather(agent_ref_yaw, agent_topk_indexes).detach()
            rec_agent_memory = topk_gather(rec_agent_memory, agent_topk_indexes).detach()
            rec_velo = topk_gather(rec_agent_velo, agent_topk_indexes).detach()
            agent_class = topk_gather(agent_cls_scores[-1], agent_topk_indexes).detach()

            _, map_topk_indexes = torch.topk(rec_map_score, self.map_num_propagated, dim=1)
            rec_map_score = topk_gather(rec_map_score, map_topk_indexes).detach()
            map_ref_point = topk_gather(map_ref_point, map_topk_indexes).detach()
            rec_map_memory = topk_gather(rec_map_memory, map_topk_indexes).detach()
            map_class = topk_gather(map_cls_scores[-1], map_topk_indexes).detach()
            # add the feature of current frame to memory
            self.agent_memory_embedding = torch.cat([rec_agent_memory, self.agent_memory_embedding], dim=1)
            self.map_memory_embedding = torch.cat([rec_map_memory, self.map_memory_embedding], dim=1)
            self.ego_memory_embedding = torch.cat([ego_memory.detach(), self.ego_memory_embedding], dim=1)
            self.memory_timestamp = np.concatenate([rec_timestamp, self.memory_timestamp], axis=1)
            self.memory_egopose = torch.cat([rec_ego_pose.detach(), self.memory_egopose], dim=1)
            self.agent_memory_reference_point = torch.cat([agent_ref_point, self.agent_memory_reference_point], dim=1)
            self.agent_memory_reference_yaw = torch.cat([agent_ref_yaw, self.agent_memory_reference_yaw], dim=1)
            self.agent_memory_class = torch.cat([agent_class, self.agent_memory_class], dim=1)
            self.map_memory_reference_point = torch.cat([map_ref_point, self.map_memory_reference_point], dim=1)
            self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
            # transform from local to global
            self.agent_memory_reference_point = transform_reference_points(self.agent_memory_reference_point, data['ego_pose'], reverse=False)
            self.agent_memory_reference_yaw += torch.arctan2(data['ego_pose'][:,1,0],data['ego_pose'][:,0,0]).unsqueeze(-1).unsqueeze(-1)
            self.map_memory_reference_point = transform_reference_points(self.map_memory_reference_point, data['ego_pose'], reverse=False)
            self.map_memory_class = torch.cat([map_class, self.map_memory_class], dim=1)
            self.memory_timestamp += data['timestamp'].reshape(data['timestamp'].shape[0],1,1)
            self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose
            self.memory_prev_exists = torch.cat([data['prev_exists'].type(torch.float32).unsqueeze(-1).unsqueeze(-1), self.memory_prev_exists], dim=1)

    @force_fp32()
    def temporal_alignment(self, agent_query, map_query):
        # We have aligned the memory refence points with current frame in pre_update_memory.
        # Now we align the memory features and get position embeddings with memory refence points
        B = agent_query.size(0)
        agent_temp_reference_point = self.agent_memory_reference_point 
        map_temp_reference_point = self.map_memory_reference_point[...,0:2] 
        ego_temp_reference_point = self.memory_egopose[:,:,0:3,3] 
        agent_temp_pos = self.agent_ref_embedding(pos2posemb(agent_temp_reference_point, self.embed_dims//2))
        map_temp_pos = self.map_ref_embedding(pos2posemb(map_temp_reference_point, self.embed_dims//2))
        ego_temp_pos = self.agent_ref_embedding(pos2posemb(ego_temp_reference_point, self.embed_dims//2))
        agent_temp_memory = self.agent_memory_embedding
        map_temp_memory = self.map_memory_embedding
        rec_ego_pose = torch.eye(4, device=agent_query.device).unsqueeze(0).unsqueeze(0).repeat(B, agent_query.size(1), 1, 1)
        memory_timestamp_tensor = torch.tensor(self.memory_timestamp,dtype=torch.float32,device=agent_query.device)
        
        # further tuning agent's query & ref
        if self.with_ego_pos:
            memory_ego_motion = []
            # [1, num_frame, dim] -> [1, num_frame*num_propagate_agent, dim] to align with agent pos mem queue
            for t in [memory_timestamp_tensor, self.memory_egopose[..., :3, :].flatten(-2)]:
                tmp_t = t.unsqueeze(2).repeat(1, 1, self.agent_num_propagated, 1).flatten(1, 2)
                memory_ego_motion.append(tmp_t)
            memory_ego_motion.append(self.memory_velo)
            memory_ego_motion = torch.cat(memory_ego_motion, dim=-1).float()
            memory_ego_motion = pos2posemb(memory_ego_motion, 12)
            agent_temp_pos = self.ego_pose_pe(agent_temp_pos, memory_ego_motion)
            agent_temp_memory = self.ego_pose_memory(agent_temp_memory, memory_ego_motion)
        
        memory_timestamp = memory_timestamp_tensor.unsqueeze(2).repeat(1, 1, self.agent_num_propagated, 1).flatten(1, 2)
        time_emded = self.time_embedding(pos2posemb(memory_timestamp, self.embed_dims).float())
        agent_cls_emded = self.agent_cls_embedding(self.agent_memory_class)
        map_cls_emded = self.map_cls_embedding(self.map_memory_class)
        agent_temp_pos += time_emded # temporal alignment
        agent_temp_pos += agent_cls_emded
        assert self.agent_num_propagated >= self.map_num_propagated
        map_temp_pos += time_emded[:, :self.map_num_propagated*self.memory_len_frame] # temporal alignment
        map_temp_pos += map_cls_emded
        ego_temp_pos += time_emded[:, ::self.agent_num_propagated]
        return agent_query, map_query, agent_temp_memory, map_temp_memory, agent_temp_pos, map_temp_pos, ego_temp_pos, rec_ego_pose[:, 0:1]
   
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           traj_cls_scores_pred,
                           gt_traj_cls_scores,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights = torch.zeros_like(traj_targets)
        traj_weights[pos_inds] = 1.0
        # trajs class scores
        gt_traj_cls_scores = gt_traj_cls_scores[:num_gt_bbox]
        traj_cls_scores = torch.zeros(num_bboxes, dtype=torch.int64, device=bbox_pred.device)
        traj_cls_scores[pos_inds] = gt_traj_cls_scores[sampling_result.pos_assigned_gt_inds].to(torch.int64)
        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks
        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask
        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets, traj_cls_scores,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )
    
    def _get_target_single_box_only(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_bbox_c = gt_bboxes.shape[-1]
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (
            labels, label_weights, bbox_targets, bbox_weights, 
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    traj_cls_scores_list,
                    gt_traj_fut_classes_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, traj_targets_list, traj_cls_scores_list, traj_weights_list,
        gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list,
            traj_cls_scores_list, gt_traj_fut_classes_list, gt_bboxes_ignore_list
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_cls_scores_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, pts_targets_list, pts_weights_list,
        pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds_traj_fix_time, 
                      ego_fut_preds_traj_fix_dist, 
                      ego_fut_preds_cls, 
                      ego_fut_gt_fix_time, 
                      ego_fut_masks_fix_time, 
                      ego_fut_gt_fix_dist, 
                      ego_fut_masks_fix_dist,
                      ego_fut_classes):
        
        num_dec = ego_fut_preds_traj_fix_time.shape[0]
        ego_fut_gt_fix_time = ego_fut_gt_fix_time.unsqueeze(0).expand(num_dec, -1, -1, -1).flatten(-2)
        ego_fut_masks_fix_time = ego_fut_masks_fix_time[None,:,:,None].expand(num_dec, -1, -1, 2).flatten(-2)
        
        if self.ego_multi_modal:
            ego_fut_preds_cls = ego_fut_preds_cls.squeeze(-1) if ego_fut_preds_cls is not None else None
            ego_fut_cls_gt = ego_fut_classes.reshape(1,ego_fut_classes.shape[0]).expand(num_dec, -1).to(torch.int64)
            ego_fut_cls_gt_tmp1 = ego_fut_classes.reshape(1,ego_fut_classes.shape[0], 1, 1).expand(num_dec, -1, -1, ego_fut_preds_traj_fix_time.shape[-2]* ego_fut_preds_traj_fix_time.shape[-1]).to(torch.int64)
            ego_fut_preds_traj_fix_time = torch.gather(ego_fut_preds_traj_fix_time.flatten(-2), dim=2, index=ego_fut_cls_gt_tmp1).squeeze(-2)
            loss_plan_cls_list = []
            
        else:
            ego_fut_preds_traj_fix_time = ego_fut_preds_traj_fix_time.squeeze(2).flatten(-2)
            loss_plan_cls_list = None
        
        if ego_fut_gt_fix_dist is not None:
            ego_fut_gt_fix_dist = ego_fut_gt_fix_dist.unsqueeze(0).expand(num_dec, -1, -1, -1).flatten(-2)
            ego_fut_masks_fix_dist = ego_fut_masks_fix_dist[None,:,:,None].expand(num_dec, -1, -1, -1).flatten(-2)
            loss_plan_reg_fix_dist_list = []   
            if self.ego_multi_modal:
                ego_fut_cls_gt_tmp2 = ego_fut_classes.reshape(1,ego_fut_classes.shape[0], 1, 1).expand(num_dec, -1, -1, ego_fut_preds_traj_fix_dist.shape[-2]* ego_fut_preds_traj_fix_dist.shape[-1]).to(torch.int64)
                ego_fut_preds_traj_fix_dist = torch.gather(ego_fut_preds_traj_fix_dist.flatten(-2), dim=2, index=ego_fut_cls_gt_tmp2).squeeze(-2)
            else:
                ego_fut_preds_traj_fix_dist = ego_fut_preds_traj_fix_dist.squeeze(2).flatten(-2)
        else: 
            loss_plan_reg_fix_dist_list = None
        loss_plan_reg_fix_time_list = []
        for dec_index in range(num_dec):
            loss_plan_l1_fix_time = self.loss_plan_reg_fix_time(
                ego_fut_preds_traj_fix_time[dec_index],
                ego_fut_gt_fix_time[dec_index],
                ego_fut_masks_fix_time[dec_index],
            )
            loss_plan_l1_fix_time = torch.nan_to_num(loss_plan_l1_fix_time) 
            loss_plan_reg_fix_time_list.append(loss_plan_l1_fix_time)               
            if ego_fut_gt_fix_dist is not None:
                loss_plan_l1_fix_dist = self.loss_plan_reg_fix_dist(
                    ego_fut_preds_traj_fix_dist[dec_index],
                    ego_fut_gt_fix_dist[dec_index],
                    ego_fut_masks_fix_dist[dec_index],
                ) 
                loss_plan_l1_fix_dist = torch.nan_to_num(loss_plan_l1_fix_dist)
                loss_plan_reg_fix_dist_list.append(loss_plan_l1_fix_dist)
            if self.ego_multi_modal:
                loss_plan_cls = self.loss_plan_cls(
                    ego_fut_preds_cls[dec_index],
                    ego_fut_cls_gt[dec_index],
                )    
                loss_plan_cls_list.append(loss_plan_cls)
        return loss_plan_reg_fix_time_list, loss_plan_reg_fix_dist_list, loss_plan_cls_list
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_traj_fut_classes,
                    gt_bboxes_mask,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        gt_traj_fut_classes_list = [gt_traj_fut_classes[i].squeeze(0) for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, traj_cls_preds,
                                           gt_traj_fut_classes_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_cls_scores_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_cls_scores = torch.cat(traj_cls_scores_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        ## adjust loss masks
        gt_bboxes_mask = gt_bboxes_mask.unsqueeze(-1).repeat(1, self.agent_num_query_sifted).flatten()
        label_weights = label_weights * gt_bboxes_mask
        bbox_weights = bbox_weights * gt_bboxes_mask.unsqueeze(-1)
        traj_weights = traj_weights * gt_bboxes_mask.unsqueeze(-1)

        # classification loss
        cls_scores = cls_scores.view(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.view(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        final_traj_preds = torch.gather(traj_preds.view(traj_preds.size(0),traj_preds.size(1), self.fut_mode, self.fut_ts, 2), dim=2, index=traj_cls_scores.view(traj_cls_preds.size(0), traj_cls_preds.size(1), 1, 1, 1).expand(-1, -1, -1, self.fut_ts, 2))
        final_traj_preds = final_traj_preds.view(-1, self.fut_ts*2)
        loss_traj = self.loss_traj(final_traj_preds[isnotnan], traj_targets[isnotnan], traj_weights[isnotnan], avg_factor=num_total_pos)
        
        # traj classification loss
        traj_cls_scores_pred = traj_cls_preds.view(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores_pred.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(traj_cls_scores_pred, traj_cls_scores)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        if loss_cls.shape == torch.Size([1]):
            loss_cls = loss_cls[0]  #tensor[1] -> num
        if loss_traj_cls.shape == torch.Size([1]):
            loss_traj_cls = loss_traj_cls[0]  #tensor[1] -> num
        return loss_cls, loss_bbox, loss_traj, loss_traj_cls
    
    def loss_single_box_only(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_mask,
                    ):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        labels_list = []
        label_weights_list = []
        bbox_targets_list= [] 
        bbox_weights_list= [] 
        num_total_pos= 0
        num_total_neg= 0

        for i in range(len(cls_scores_list)):
            cls_reg_targets = self._get_target_single_box_only(cls_scores_list[i], bbox_preds_list[i],
                                           gt_labels_list[i],gt_bboxes_list[i])
            labels_list.append(cls_reg_targets[0])
            label_weights_list.append(cls_reg_targets[1])
            bbox_targets_list.append(cls_reg_targets[2])
            bbox_weights_list.append(cls_reg_targets[3])
            num_total_pos += len(cls_reg_targets[4])
            num_total_neg += len(cls_reg_targets[5])

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        ## adjust loss masks
        gt_bboxes_mask = gt_bboxes_mask.unsqueeze(-1).repeat(1, self.agent_num_query).flatten()
        label_weights = label_weights * gt_bboxes_mask
        bbox_weights = bbox_weights * gt_bboxes_mask.unsqueeze(-1)
        # classification loss
        cls_scores = cls_scores.view(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        cls_scores[torch.where(labels!=10)[0], :].sigmoid()
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.view(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)


        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)

        if loss_cls.shape == torch.Size([1]):
            loss_cls = loss_cls[0]  #tensor[1] -> num

        return loss_cls, loss_bbox



    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    map_loss_gt_mask,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list, pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        ## adjust loss masks
        map_loss_gt_mask = map_loss_gt_mask.unsqueeze(-1).repeat(1, cls_scores_list[0].shape[0]).flatten()
        label_weights = label_weights * map_loss_gt_mask
        bbox_weights = bbox_weights * map_loss_gt_mask.unsqueeze(-1)
        pts_weights = pts_weights * map_loss_gt_mask.unsqueeze(-1).unsqueeze(-1)

        # classification loss
        cls_scores = cls_scores.view(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus for normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        # bbox_preds = bbox_preds.view(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_preds = normalize_2d_bbox(bbox_preds, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1) 

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)
        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.view(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()
        normalized_pts_preds = normalize_2d_pts(pts_preds, self.pc_range)
        loss_pts = self.loss_map_pts(
            normalized_pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = pts_preds
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)

        if loss_cls.shape == torch.Size([1]):
            loss_cls = loss_cls[0]  #tensor[1] -> num
        return loss_cls, loss_pts, loss_dir

    @force_fp32()
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_traj_fut_classes,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt_fix_time,
             ego_fut_masks_fix_time,
             ego_fut_gt_fix_dist,
             ego_fut_masks_fix_dist,
             ego_fut_cmd,
             ego_fut_classes,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        ego_fut_preds_traj_fix_time = preds_dicts['ego_fut_preds_fix_time']
        ego_fut_preds_traj_fix_dist = preds_dicts['ego_fut_preds_fix_dist'] if self.fut_ego_fix_dist else None
        ego_fut_preds_cls = preds_dicts['ego_traj_cls_scores']
        map_pre_cls_scores = preds_dicts['map_pre_cls_scores']
        map_pre_coord_preds = preds_dicts['map_pre_coord_preds']
        map_pre_pts_coord_preds = preds_dicts['map_pre_pts_coord_preds']
        agent_pre_cls_scores = preds_dicts['agent_pre_cls_scores']
        agent_pre_coord_preds = preds_dicts['agent_pre_coord_preds']


        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        if self.pseudo_agent_instance is None:
            self.pseudo_agent_instance = [LiDARInstance3DBoxes(torch.Tensor([[0.0, 0.0, 0.0, 3.0, 1.5, 1.5, 0.0, 0.0, 0.0]]), box_dim=9), torch.zeros(1).long().cuda(), torch.zeros(1, 34).cuda()]
        
        new_gt_bboxes_list = []
        gt_bboxes_mask = torch.ones(len(gt_labels_list), device=device)
        for i, gt_labels in enumerate(gt_labels_list):
            # if there is no gt, mask the corresponding loss, and use the pseudo agent instance
            if len(gt_labels) == 0:
                gt_bboxes_mask[i] = 0
                new_gt_bboxes_list.append(torch.cat(
                    (self.pseudo_agent_instance[0].gravity_center, self.pseudo_agent_instance[0].tensor[:, 3:]),
                    dim=1).to(device))
                gt_labels_list[i] = self.pseudo_agent_instance[1].to(device)
                # no need to adjust gt_traj_fut_classes, cuz it's already padded
                gt_attr_labels[i] = self.pseudo_agent_instance[2].to(device)
            else :
                new_gt_bboxes_list.append(torch.cat(
                    (gt_bboxes_list[i].gravity_center, gt_bboxes_list[i].tensor[:, 3:]), # (x, y, z, x_size, y_size, z_size, yaw, vx, vy)
                    dim=1).to(device))
        all_gt_bboxes_list = [new_gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_traj_fut_classes = [gt_traj_fut_classes for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        all_gt_bboxes_mask = [gt_bboxes_mask for _ in range(num_dec_layers)]
        
        losses_cls, losses_bbox, losses_traj, losses_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_traj_fut_classes, all_gt_bboxes_mask, all_gt_bboxes_ignore_list)
        
        l0_agent_loss_cls_all, l0_aggent_loss_bbox_all = self.loss_single_box_only(agent_pre_cls_scores, agent_pre_coord_preds, all_gt_bboxes_list[0], all_gt_labels_list[0], all_gt_bboxes_mask[0])

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        # randomly grab a pseudo map_bbox_instance
        if self.pseudo_map_instance is None:
            for i in range(len(map_gt_labels_list)):
                if len(map_gt_labels_list[i]) != 0:
                    self.pseudo_map_instance = [map_gt_vecs_list[i], map_gt_labels_list[i]]
                    break
            assert self.pseudo_map_instance is not None
        new_map_gt_bboxes_list = []
        map_gt_pts_list = []
        map_gt_shifts_pts_list = []
        map_loss_gt_mask = torch.ones(len(map_gt_vecs_list), device=device)
        for i, map_gt_labels in enumerate(map_gt_labels_list):
            # if there is no gt, mask the corresponding loss, and use the pseudo map instance
            if len(map_gt_labels) == 0:
                map_loss_gt_mask[i] = 0
                new_map_gt_bboxes_list.append(self.pseudo_map_instance[0].bbox.to(device))
                map_gt_pts_list.append(self.pseudo_map_instance[0].fixed_num_sampled_points.to(device))
                map_gt_shifts_pts_list.append(self.pseudo_map_instance[0].shift_fixed_num_sampled_points.to(device))
                map_gt_labels_list[i] = self.pseudo_map_instance[1].to(device)
            else :
                new_map_gt_bboxes_list.append(map_gt_bboxes_list[i].bbox.to(device))
                map_gt_pts_list.append(map_gt_bboxes_list[i].fixed_num_sampled_points.to(device))
                if self.map_gt_shift_pts_pattern == 'v0':
                    map_gt_shifts_pts_list.append(map_gt_bboxes_list[i].shift_fixed_num_sampled_points.to(device))
                elif self.map_gt_shift_pts_pattern == 'v1':
                    map_gt_shifts_pts_list.append(map_gt_bboxes_list[i].shift_fixed_num_sampled_points_v1.to(device))
                elif self.map_gt_shift_pts_pattern == 'v2':
                    map_gt_shifts_pts_list.append(map_gt_bboxes_list[i].shift_fixed_num_sampled_points_v2.to(device))
                elif self.map_gt_shift_pts_pattern == 'v3':
                    map_gt_shifts_pts_list.append(map_gt_bboxes_list[i].shift_fixed_num_sampled_points_v3.to(device))
                elif self.map_gt_shift_pts_pattern == 'v4':
                    map_gt_shifts_pts_list.append(map_gt_bboxes_list[i].shift_fixed_num_sampled_points_v4.to(device))
                else:
                    raise NotImplementedError

        map_all_gt_bboxes_list = [new_map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        map_all_loss_gt_mask = [map_loss_gt_mask for _ in range(num_dec_layers)]
        map_losses_cls, map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_loss_gt_mask, map_all_gt_bboxes_ignore_list)
            
        l0_loss_map_cls_all, l0_loss_map_pts_all, l0_loss_map_dir_all = self.map_loss_single(map_pre_cls_scores, map_pre_coord_preds, map_pre_pts_coord_preds, map_all_gt_bboxes_list[0], map_all_gt_labels_list[0],map_all_gt_shifts_pts_list[0], map_all_loss_gt_mask[0], map_all_gt_bboxes_ignore_list[0])
        
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = losses_traj[-1]
        loss_dict['loss_traj_cls'] = losses_traj_cls[-1]
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt_fix_time = ego_fut_gt_fix_time.squeeze(1)
        ego_fut_masks_fix_time = ego_fut_masks_fix_time.squeeze(1).squeeze(1)
        ego_fut_gt_fix_dist = ego_fut_gt_fix_dist.squeeze(1) if ego_fut_gt_fix_dist is not None else None
        ego_fut_masks_fix_dist = ego_fut_masks_fix_dist.squeeze(1).squeeze(1) if ego_fut_masks_fix_dist is not None else None

        batch, num_agent = all_traj_preds[-1].shape[:2]
        loss_plan_input = [ego_fut_preds_traj_fix_time, ego_fut_preds_traj_fix_dist, ego_fut_preds_cls, ego_fut_gt_fix_time, ego_fut_masks_fix_time, 
                           ego_fut_gt_fix_dist, ego_fut_masks_fix_dist, ego_fut_classes]

        loss_plan_reg_fix_time, loss_plan_reg_fix_dist, loss_plan_cls = self.loss_planning(*loss_plan_input)
        for i in range(len(loss_plan_reg_fix_time)):
            loss_dict[f"d{i}.loss_plan_l1_fix_time"] = loss_plan_reg_fix_time[i] 
            if loss_plan_reg_fix_dist is not None:
                loss_dict[f"d{i}.loss_plan_l1_fix_dist"] = loss_plan_reg_fix_dist[i] 
            if self.ego_multi_modal:
                loss_dict[f"d{i}.loss_plan_cls"] = loss_plan_cls[i] 

        loss_dict['d0.loss_cls_all'] = l0_agent_loss_cls_all
        loss_dict['d0.loss_bbox_all'] = l0_aggent_loss_bbox_all

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        num_dec_layer = 0
        for loss_traj_cls_i, loss_traj_i in zip(losses_traj_cls[:-1], losses_traj[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_traj_cls'] = loss_traj_cls_i
            loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i
            num_dec_layer += 1

        loss_dict['d0.loss_map_cls_all'] = l0_loss_map_cls_all
        loss_dict['d0.loss_map_pts_all'] = l0_loss_map_pts_all
        loss_dict['d0.loss_map_dir_all'] = l0_loss_map_dir_all

        num_dec_layer = 0
        for map_loss_cls_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1
        
        return loss_dict

class MLN(BaseModule):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''
    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.SiLU(inplace=True),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()
        self.fp16_enabled = False

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    @auto_fp16()
    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta
        return out