_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
# #
# #backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'adzoo/drivetransformer/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

NameMapping = {
    #=================vehicle=================
    # bicycle
    'vehicle.bh.crossbike': 'bicycle',
    "vehicle.diamondback.century": 'bicycle',
    "vehicle.gazelle.omafiets": 'bicycle',
    # car
    "vehicle.audi.etron": 'car',
    "vehicle.chevrolet.impala": 'car',
    "vehicle.dodge.charger_2020": 'car',
    "vehicle.dodge.charger_police": 'car',
    "vehicle.dodge.charger_police_2020": 'car',
    "vehicle.lincoln.mkz_2017": 'car',
    "vehicle.lincoln.mkz_2020": 'car',
    "vehicle.mini.cooper_s_2021": 'car',
    "vehicle.mercedes.coupe_2020": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.nissan.patrol_2021": 'car',
    "vehicle.audi.tt": 'car',
    "vehicle.audi.etron": 'car',
    "vehicle.ford.crown": 'car',
    "vehicle.ford.mustang": 'car',
    "vehicle.tesla.model3": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked": 'car',
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": 'car',
    # bus
    # van
    "/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked": "van",
    "vehicle.ford.ambulance": "van",
    # truck
    "vehicle.carlamotors.firetruck": 'truck',
    #=========================================

    #=================traffic sign============
    # traffic.speed_limit
    "traffic.speed_limit.30": 'traffic_sign',
    "traffic.speed_limit.40": 'traffic_sign',
    "traffic.speed_limit.50": 'traffic_sign',
    "traffic.speed_limit.60": 'traffic_sign',
    "traffic.speed_limit.90": 'traffic_sign',
    "traffic.speed_limit.120": 'traffic_sign',
    
    "traffic.stop": 'traffic_sign',
    "traffic.yield": 'traffic_sign',
    "traffic.traffic_light": 'traffic_light',
    #=========================================

    #===================Construction===========
    "static.prop.warningconstruction" : 'traffic_cone',
    "static.prop.warningaccident": 'traffic_cone',
    "static.prop.trafficwarning": "traffic_cone",

    #===================Construction===========
    "static.prop.constructioncone": 'traffic_cone',

    #=================pedestrian==============
    "walker.pedestrian.0001": 'pedestrian',
    "walker.pedestrian.0003": 'pedestrian',
    "walker.pedestrian.0004": 'pedestrian',
    "walker.pedestrian.0005": 'pedestrian',
    "walker.pedestrian.0007": 'pedestrian',
    "walker.pedestrian.0010": 'pedestrian',
    "walker.pedestrian.0013": 'pedestrian',
    "walker.pedestrian.0014": 'pedestrian',
    "walker.pedestrian.0015": 'pedestrian',
    "walker.pedestrian.0016": 'pedestrian',
    "walker.pedestrian.0017": 'pedestrian',
    "walker.pedestrian.0018": 'pedestrian',
    "walker.pedestrian.0019": 'pedestrian',
    "walker.pedestrian.0020": 'pedestrian',
    "walker.pedestrian.0021": 'pedestrian',
    "walker.pedestrian.0022": 'pedestrian',
    "walker.pedestrian.0025": 'pedestrian',
    "walker.pedestrian.0027": 'pedestrian',
    "walker.pedestrian.0030": 'pedestrian',
    "walker.pedestrian.0031": 'pedestrian',
    "walker.pedestrian.0032": 'pedestrian',
    "walker.pedestrian.0034": 'pedestrian',
    "walker.pedestrian.0035": 'pedestrian',
    "walker.pedestrian.0041": 'pedestrian',
    "walker.pedestrian.0042": 'pedestrian',
    "walker.pedestrian.0046": 'pedestrian',
    "walker.pedestrian.0047": 'pedestrian',

    # ==========================================
    "static.prop.dirtdebris01": 'others',
    "static.prop.dirtdebris02": 'others',
}

collect_keys=['lidar2img', 'cam_intrinsic', 'cam_extrinsic','timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes']
# For nuScenes we usually do 10-class detection
class_names = [
'car','van','truck','bicycle','traffic_sign','traffic_cone','traffic_light','pedestrian','others'
]
num_classes = len(class_names)

# map has classes: divider, ped_crossing, boundary
map_classes = ['Broken', 'Solid','SolidSolid','Center','TrafficLight','StopSign']

map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)
agent_query_num_vec = 900
agent_num_topk_sift = 900
agent_num_propagated = 50
map_query_num_vec = 100
map_num_topk_sift = 100
map_num_propagated = 50
memory_len_frame = 10
num_mode = 6
num_gpus = 8
batch_size = 24
num_iters_per_epoch = 200000 // (num_gpus * batch_size)

data_aug_conf = {
    "resize_lim": (0.64, 0.69),
    "final_dim": (384, 1056),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 512
queue_length = 1 # each sequence contains `queue_length` frames.
total_epochs = 60
dropout = 0.1

model = dict(
    type='DriveTransformer',
    use_grid_mask=False,
    pretrained=dict(img="./ckpts/resnet50-19c8e357.pth"),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    ## Model Size
    pts_bbox_head=dict(
        type='DriveTransformerlHead',
        ## Model Size
        ego_lcf_feat_idx=[0,1,2,3,4,5,6,7,8],
        ego_command_dim=140,
        img_stride=32,
        embed_dims=_dim_,
        num_reg_fcs=2,
        num_cls_fcs=2,
        agent_num_propagated=agent_num_propagated, ## Per Frame Per Type
        map_num_propagated=map_num_propagated, ## Per Frame Per Type
        memory_len_frame=memory_len_frame,
        ## Det & Pred
        agent_num_query=agent_query_num_vec,
        agent_num_query_sifted=agent_num_topk_sift,
        fut_mode=num_mode,
        fut_ego_mode=1,
        fut_ts=6,
        fut_ego_fix_dist=True,  # predict ego trajetory in 
        fut_ts_ego_fix_dist=20, # 20 points with interval of 1m
        fut_ts_ego_fix_time=30, # 30 points with interval of 0.1s
        num_classes=num_classes,
        code_size=10,
        ## Online Mapping
        map_num_query=map_query_num_vec,
        map_num_query_sifted=map_num_topk_sift,
        map_num_classes=map_num_classes,
        map_num_pts_per_vec=map_fixed_ptsnum_per_pred_line,
        map_num_pts_per_gt_vec=map_fixed_ptsnum_per_gt_line,
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
        position_range=point_cloud_range,    
        depth_start=1,
        depth_step=0.8,
        depth_num=64,
  
        ## InitLayer
        agent_prep_decoder=dict(
            type='DriveTransformerPreDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='DriveTransformerPreDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout),
                    dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout),
                    ],
                ffn_cfgs=dict(
                    type="SwiGLULayer",
                    embed_dims=_dim_,
                    feedforward_channels=int(_dim_*4),
                    ffn_drop=dropout,
                ),
                with_cp=False,  ###use checkpoint to save memory
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))),
        map_prep_decoder=dict(
            type='DriveTransformerPreDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='DriveTransformerPreDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout),
                    dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout),
                    ],
                ffn_cfgs=dict(
                    type="SwiGLULayer",
                    embed_dims=_dim_,
                    feedforward_channels=int(_dim_*4),
                    ffn_drop=dropout,
                ),
                with_cp=False,  ###use checkpoint to save memory
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))),
        ## Major Layer
        transformer=dict(
            type='DriveTransformerWrapper',
            embed_dims=_dim_,
            decoder=dict(
                type='DriveTransformerDecoder',
                num_layers=6,
                fut_mode = num_mode,
                agent_num_query = agent_num_topk_sift,
                map_num_query = map_num_topk_sift,
                map_num_pts_per_vec = map_fixed_ptsnum_per_pred_line,
                return_intermediate=True,
                embed_dims=_dim_,
                refine=True,
                transformerlayers=dict(
                    type='DriveTransformerDecoderLayer',
                    agent_query_num = agent_num_topk_sift,
                    map_query_num = map_num_topk_sift,
                    memory_len_frame = memory_len_frame,
                    agent_num_propagated=agent_num_propagated,
                    map_num_propagated=map_num_propagated,
                    map_pts_per_vec=map_fixed_ptsnum_per_pred_line,
                    feedforward_channels=int(_dim_*4),
                    ffn_dropout=dropout,
                    with_cp=False,  ###use checkpoint to save memory
                    attn_cfgs=[
                        dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout,
                        layer_scale=1e-2),
                        dict(
                        type='AttentionLayer',
                        embed_dims=_dim_,
                        head_dim=64,
                        attn_drop=dropout,
                        layer_scale=1e-2),
                        dict(
                            type='AttentionLayer',
                            embed_dims=_dim_,
                            head_dim=64,
                            attn_drop=dropout,
                            no_wq=True),
                        ],
                    ffn_cfgs=dict(
                        type="SwiGLULayer",
                        embed_dims=_dim_,
                        feedforward_channels=int(_dim_*4),
                        ffn_drop=dropout,
                    ),
                    operation_order=('task_self_attn', 'norm', 'temporal_cross_attn', 'norm', 'sensor_cross_attn', 'norm',  'ffn', 'norm'),
                ),
            ),
        ),
        ## Det Loss
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=point_cloud_range,
            max_num=100,
            voxel_size=voxel_size,
            num_classes=num_classes),
        
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        ## Motion Loss
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,
            loss_weight=0.2),
        ## Online Mapping Loss
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=map_num_classes),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        ## Planning Loss
        loss_plan_reg_fix_time=dict(type='L1Loss', loss_weight=3.5),
        loss_plan_reg_fix_dist=dict(type='L1Loss', loss_weight=10.0),
        loss_plan_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.5,
            loss_weight=20.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0, gamma=2.0, alpha=0.25),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range),
        map_assigner=dict(
            type='MapHungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0, gamma=2.0, alpha=0.25),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
            pc_range=point_cloud_range))))

dataset_type = "B2D_DriveTransformer_Dataset"
data_root = "data/bench2drive"
info_root = "data/infos"
map_root = "data/bench2drive/maps"
map_file = "data/infos/b2d_map_infos.pkl"
file_client_args = dict(backend="disk")
ann_file_train=info_root + f"/b2d_infos_v1_train_drivetransformer_meta.pkl"
ann_file_val=info_root + f"/b2d_infos_v1_val_drivetransformer_meta.pkl"
ann_file_test=info_root + f"/b2d_infos_v1_val_drivetransformer_meta.pkl"
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='TrajPreprocess', pc_range=point_cloud_range, with_ego_fix_dist=True, ego_fut_offset_input=False, assign_class_for_ego=False),
    dict(type='CustomFormatBundle3D', class_names=class_names, with_ego=True, collect_keys=collect_keys),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'fut_valid_flag_fix_time',
               'ego_fut_trajs_fix_time', 'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
               'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist', 'ego_fut_cmd', 'ego_lcf_feat', 
               'gt_attr_labels', 'prev_exists', 'index'] + collect_keys)
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='TrajPreprocess', pc_range=point_cloud_range, with_ego_fix_dist=True, ego_fut_offset_input=False, assign_class_for_ego=False),
    dict(type='CustomFormatBundle3D', class_names=class_names, with_ego=True, collect_keys=collect_keys),
    dict(type='CustomCollect3D',\
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'fut_valid_flag_fix_time',
               'ego_fut_trajs_fix_time', 'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
               'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist', 'ego_fut_cmd', 'ego_lcf_feat', 
               'gt_attr_labels', 'prev_exists', 'index'] + collect_keys)

]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=12,
    train = dict(
        type = dataset_type,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_aug_conf=data_aug_conf,
        classes=class_names,
        name_mapping=NameMapping,
        map_file=map_file,
        modality=input_modality,
        test_mode=False,
        point_cloud_range=point_cloud_range,
        collect_keys=collect_keys,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        filter_empty_gt=False,
        sub_seq_lenth=-1,
        use_splited_data = True,
        cache_lenth = batch_size+1,
        box_type_3d='LiDAR',
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,       
        use_angle_as_dis_traj=True, 
        ),
    val=dict(        
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        classes=class_names,
        name_mapping=NameMapping,
        map_file=map_file,
        modality=input_modality,
        test_mode=True,
        point_cloud_range=point_cloud_range,
        collect_keys=collect_keys,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        filter_empty_gt=False,
        use_splited_data=True,
        cache_lenth=batch_size+1,
        box_type_3d='LiDAR',
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,        
        use_angle_as_dis_traj=True, 
        ),
    test=dict(        
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        classes=class_names,
        name_mapping=NameMapping,
        map_file=map_file,
        modality=input_modality,
        test_mode=True,
        point_cloud_range=point_cloud_range,
        collect_keys=collect_keys,
        polyline_points_num=map_fixed_ptsnum_per_gt_line,
        filter_empty_gt=False,
        use_splited_data = True,
        cache_lenth = batch_size+1,
        box_type_3d='LiDAR',
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,   
        use_angle_as_dis_traj=True, 
        ),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-2)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-2)

runner = dict(type='IterBasedRunner', max_iters=total_epochs * num_iters_per_epoch)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
find_unused_parameters = True
checkpoint_config = dict(interval=3000)

custom_hooks = [dict(type='CustomSetEpochInfoHook')]

