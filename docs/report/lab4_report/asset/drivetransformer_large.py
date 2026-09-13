point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
class_names = [
    'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
    'traffic_light', 'pedestrian', 'others'
]
dataset_type = 'B2D_DriveTransformer_Dataset'
data_root = '/root/autodl-tmp/Bench2DriveZoo/data/bench2drive'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='TrajPreprocess',
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        with_ego_fix_dist=True,
        ego_fut_offset_input=False,
        assign_class_for_ego=False),
    dict(
        type='CustomFormatBundle3D',
        class_names=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        with_ego=True,
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
            'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
            'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist', 'ego_fut_cmd',
            'ego_lcf_feat', 'gt_attr_labels', 'prev_exists', 'index',
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ]),
    dict(
        type='TrajPreprocess',
        pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        with_ego_fix_dist=True,
        ego_fut_offset_input=False,
        assign_class_for_ego=False),
    dict(
        type='CustomFormatBundle3D',
        class_names=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        with_ego=True,
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ]),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
            'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
            'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist', 'ego_fut_cmd',
            'ego_lcf_feat', 'gt_attr_labels', 'prev_exists', 'index',
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='B2D_DriveTransformer_Dataset',
        data_root='/root/autodl-tmp/Bench2DriveZoo/data/bench2drive',
        ann_file=
        '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_train_drivetransformer_meta.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='TrajPreprocess',
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                with_ego_fix_dist=True,
                ego_fut_offset_input=False,
                assign_class_for_ego=False),
            dict(
                type='CustomFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_ego=True,
                collect_keys=[
                    'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
                    'ego_pose', 'ego_pose_inv', 'pad_shape',
                    'gt_traj_fut_classes', 'ego_fut_classes'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
                    'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
                    'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist',
                    'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels',
                    'prev_exists', 'index', 'lidar2img', 'cam_intrinsic',
                    'cam_extrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv',
                    'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes'
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        data_aug_conf=dict(
            resize_lim=(0.64, 0.69),
            final_dim=(384, 1056),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_file='/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_map_infos.pkl',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ],
        polyline_points_num=20,
        filter_empty_gt=False,
        sub_seq_lenth=-1,
        use_splited_data=True,
        cache_lenth=5,
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,
        use_angle_as_dis_traj=True),
    val=dict(
        type='B2D_DriveTransformer_Dataset',
        ann_file=
        '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_val_drivetransformer_meta.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='TrajPreprocess',
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                with_ego_fix_dist=True,
                ego_fut_offset_input=False,
                assign_class_for_ego=False),
            dict(
                type='CustomFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_ego=True,
                collect_keys=[
                    'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
                    'ego_pose', 'ego_pose_inv', 'pad_shape',
                    'gt_traj_fut_classes', 'ego_fut_classes'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
                    'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
                    'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist',
                    'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels',
                    'prev_exists', 'index', 'lidar2img', 'cam_intrinsic',
                    'cam_extrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv',
                    'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes'
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='/root/autodl-tmp/Bench2DriveZoo/data/bench2drive',
        data_aug_conf=dict(
            resize_lim=(0.64, 0.69),
            final_dim=(384, 1056),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_file='/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_map_infos.pkl',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ],
        polyline_points_num=20,
        filter_empty_gt=False,
        use_splited_data=True,
        cache_lenth=5,
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,
        use_angle_as_dis_traj=True),
    test=dict(
        type='B2D_DriveTransformer_Dataset',
        data_root='/root/autodl-tmp/Bench2DriveZoo/data/bench2drive',
        ann_file=
        '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_val_drivetransformer_meta.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ]),
            dict(
                type='TrajPreprocess',
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                with_ego_fix_dist=True,
                ego_fut_offset_input=False,
                assign_class_for_ego=False),
            dict(
                type='CustomFormatBundle3D',
                class_names=[
                    'car', 'van', 'truck', 'bicycle', 'traffic_sign',
                    'traffic_cone', 'traffic_light', 'pedestrian', 'others'
                ],
                with_ego=True,
                collect_keys=[
                    'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
                    'ego_pose', 'ego_pose_inv', 'pad_shape',
                    'gt_traj_fut_classes', 'ego_fut_classes'
                ]),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'fut_valid_flag_fix_time', 'ego_fut_trajs_fix_time',
                    'ego_fut_masks_fix_time', 'fut_valid_flag_fix_dist',
                    'ego_fut_trajs_fix_dist', 'ego_fut_masks_fix_dist',
                    'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels',
                    'prev_exists', 'index', 'lidar2img', 'cam_intrinsic',
                    'cam_extrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv',
                    'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes'
                ])
        ],
        classes=[
            'car', 'van', 'truck', 'bicycle', 'traffic_sign', 'traffic_cone',
            'traffic_light', 'pedestrian', 'others'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_aug_conf=dict(
            resize_lim=(0.64, 0.69),
            final_dim=(384, 1056),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        name_mapping=dict({
            'vehicle.bh.crossbike':
            'bicycle',
            'vehicle.diamondback.century':
            'bicycle',
            'vehicle.gazelle.omafiets':
            'bicycle',
            'vehicle.audi.etron':
            'car',
            'vehicle.chevrolet.impala':
            'car',
            'vehicle.dodge.charger_2020':
            'car',
            'vehicle.dodge.charger_police':
            'car',
            'vehicle.dodge.charger_police_2020':
            'car',
            'vehicle.lincoln.mkz_2017':
            'car',
            'vehicle.lincoln.mkz_2020':
            'car',
            'vehicle.mini.cooper_s_2021':
            'car',
            'vehicle.mercedes.coupe_2020':
            'car',
            'vehicle.ford.mustang':
            'car',
            'vehicle.nissan.patrol_2021':
            'car',
            'vehicle.audi.tt':
            'car',
            'vehicle.ford.crown':
            'car',
            'vehicle.tesla.model3':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
            'car',
            '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
            'van',
            'vehicle.ford.ambulance':
            'van',
            'vehicle.carlamotors.firetruck':
            'truck',
            'traffic.speed_limit.30':
            'traffic_sign',
            'traffic.speed_limit.40':
            'traffic_sign',
            'traffic.speed_limit.50':
            'traffic_sign',
            'traffic.speed_limit.60':
            'traffic_sign',
            'traffic.speed_limit.90':
            'traffic_sign',
            'traffic.speed_limit.120':
            'traffic_sign',
            'traffic.stop':
            'traffic_sign',
            'traffic.yield':
            'traffic_sign',
            'traffic.traffic_light':
            'traffic_light',
            'static.prop.warningconstruction':
            'traffic_cone',
            'static.prop.warningaccident':
            'traffic_cone',
            'static.prop.trafficwarning':
            'traffic_cone',
            'static.prop.constructioncone':
            'traffic_cone',
            'walker.pedestrian.0001':
            'pedestrian',
            'walker.pedestrian.0003':
            'pedestrian',
            'walker.pedestrian.0004':
            'pedestrian',
            'walker.pedestrian.0005':
            'pedestrian',
            'walker.pedestrian.0007':
            'pedestrian',
            'walker.pedestrian.0010':
            'pedestrian',
            'walker.pedestrian.0013':
            'pedestrian',
            'walker.pedestrian.0014':
            'pedestrian',
            'walker.pedestrian.0015':
            'pedestrian',
            'walker.pedestrian.0016':
            'pedestrian',
            'walker.pedestrian.0017':
            'pedestrian',
            'walker.pedestrian.0018':
            'pedestrian',
            'walker.pedestrian.0019':
            'pedestrian',
            'walker.pedestrian.0020':
            'pedestrian',
            'walker.pedestrian.0021':
            'pedestrian',
            'walker.pedestrian.0022':
            'pedestrian',
            'walker.pedestrian.0025':
            'pedestrian',
            'walker.pedestrian.0027':
            'pedestrian',
            'walker.pedestrian.0030':
            'pedestrian',
            'walker.pedestrian.0031':
            'pedestrian',
            'walker.pedestrian.0032':
            'pedestrian',
            'walker.pedestrian.0034':
            'pedestrian',
            'walker.pedestrian.0035':
            'pedestrian',
            'walker.pedestrian.0041':
            'pedestrian',
            'walker.pedestrian.0042':
            'pedestrian',
            'walker.pedestrian.0046':
            'pedestrian',
            'walker.pedestrian.0047':
            'pedestrian',
            'static.prop.dirtdebris01':
            'others',
            'static.prop.dirtdebris02':
            'others'
        }),
        map_file='/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_map_infos.pkl',
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        collect_keys=[
            'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp',
            'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes',
            'ego_fut_classes'
        ],
        polyline_points_num=20,
        filter_empty_gt=False,
        use_splited_data=True,
        cache_lenth=5,
        future_frames=6,
        future_frames_ego_fix_time=30,
        future_frames_ego_fix_dist=20,
        sample_interval_ego_fut=1,
        sample_interval=5,
        fix_future_dis=1,
        use_angle_as_dis_traj=True),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=24,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5,
            file_client_args=dict(backend='disk')),
        dict(
            type='LoadPointsFromMultiSweeps',
            sweeps_num=10,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                'barrier'
            ],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=3000)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_large/'
load_from = None
resume_from = 'adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_large/latest.pth'
workflow = [('train', 1)]
plugin = True
plugin_dir = 'adzoo/drivetransformer/mmdet3d_plugin/'
voxel_size = [0.15, 0.15, 4]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
NameMapping = dict({
    'vehicle.bh.crossbike':
    'bicycle',
    'vehicle.diamondback.century':
    'bicycle',
    'vehicle.gazelle.omafiets':
    'bicycle',
    'vehicle.audi.etron':
    'car',
    'vehicle.chevrolet.impala':
    'car',
    'vehicle.dodge.charger_2020':
    'car',
    'vehicle.dodge.charger_police':
    'car',
    'vehicle.dodge.charger_police_2020':
    'car',
    'vehicle.lincoln.mkz_2017':
    'car',
    'vehicle.lincoln.mkz_2020':
    'car',
    'vehicle.mini.cooper_s_2021':
    'car',
    'vehicle.mercedes.coupe_2020':
    'car',
    'vehicle.ford.mustang':
    'car',
    'vehicle.nissan.patrol_2021':
    'car',
    'vehicle.audi.tt':
    'car',
    'vehicle.ford.crown':
    'car',
    'vehicle.tesla.model3':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/FordCrown/SM_FordCrown_parked.SM_FordCrown_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Charger/SM_ChargerParked.SM_ChargerParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Lincoln/SM_LincolnParked.SM_LincolnParked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/MercedesCCC/SM_MercedesCCC_Parked.SM_MercedesCCC_Parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/Mini2021/SM_Mini2021_parked.SM_Mini2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/NissanPatrol2021/SM_NissanPatrol2021_parked.SM_NissanPatrol2021_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/TeslaM3/SM_TeslaM3_parked.SM_TeslaM3_parked':
    'car',
    '/Game/Carla/Static/Car/4Wheeled/ParkedVehicles/VolkswagenT2/SM_VolkswagenT2_2021_Parked.SM_VolkswagenT2_2021_Parked':
    'van',
    'vehicle.ford.ambulance':
    'van',
    'vehicle.carlamotors.firetruck':
    'truck',
    'traffic.speed_limit.30':
    'traffic_sign',
    'traffic.speed_limit.40':
    'traffic_sign',
    'traffic.speed_limit.50':
    'traffic_sign',
    'traffic.speed_limit.60':
    'traffic_sign',
    'traffic.speed_limit.90':
    'traffic_sign',
    'traffic.speed_limit.120':
    'traffic_sign',
    'traffic.stop':
    'traffic_sign',
    'traffic.yield':
    'traffic_sign',
    'traffic.traffic_light':
    'traffic_light',
    'static.prop.warningconstruction':
    'traffic_cone',
    'static.prop.warningaccident':
    'traffic_cone',
    'static.prop.trafficwarning':
    'traffic_cone',
    'static.prop.constructioncone':
    'traffic_cone',
    'walker.pedestrian.0001':
    'pedestrian',
    'walker.pedestrian.0003':
    'pedestrian',
    'walker.pedestrian.0004':
    'pedestrian',
    'walker.pedestrian.0005':
    'pedestrian',
    'walker.pedestrian.0007':
    'pedestrian',
    'walker.pedestrian.0010':
    'pedestrian',
    'walker.pedestrian.0013':
    'pedestrian',
    'walker.pedestrian.0014':
    'pedestrian',
    'walker.pedestrian.0015':
    'pedestrian',
    'walker.pedestrian.0016':
    'pedestrian',
    'walker.pedestrian.0017':
    'pedestrian',
    'walker.pedestrian.0018':
    'pedestrian',
    'walker.pedestrian.0019':
    'pedestrian',
    'walker.pedestrian.0020':
    'pedestrian',
    'walker.pedestrian.0021':
    'pedestrian',
    'walker.pedestrian.0022':
    'pedestrian',
    'walker.pedestrian.0025':
    'pedestrian',
    'walker.pedestrian.0027':
    'pedestrian',
    'walker.pedestrian.0030':
    'pedestrian',
    'walker.pedestrian.0031':
    'pedestrian',
    'walker.pedestrian.0032':
    'pedestrian',
    'walker.pedestrian.0034':
    'pedestrian',
    'walker.pedestrian.0035':
    'pedestrian',
    'walker.pedestrian.0041':
    'pedestrian',
    'walker.pedestrian.0042':
    'pedestrian',
    'walker.pedestrian.0046':
    'pedestrian',
    'walker.pedestrian.0047':
    'pedestrian',
    'static.prop.dirtdebris01':
    'others',
    'static.prop.dirtdebris02':
    'others'
})
collect_keys = [
    'lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp', 'ego_pose',
    'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes'
]
num_classes = 9
map_classes = [
    'Broken', 'Solid', 'SolidSolid', 'Center', 'TrafficLight', 'StopSign'
]
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = 6
agent_query_num_vec = 900
agent_num_topk_sift = 900
agent_num_propagated = 50
map_query_num_vec = 100
map_num_topk_sift = 100
map_num_propagated = 50
memory_len_frame = 10
num_mode = 6
num_gpus = 4
batch_size = 4
num_iters_per_epoch = 3750
data_aug_conf = dict(
    resize_lim=(0.64, 0.69),
    final_dim=(384, 1056),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0])
_dim_ = 768
total_epochs = 60
dropout = 0.1
extra_post_norm = True
model = dict(
    type='DriveTransformer',
    use_grid_mask=False,
    pretrained=dict(img='./ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=768,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DriveTransformerlHead',
        ego_lcf_feat_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        ego_command_dim=140,
        img_stride=32,
        embed_dims=768,
        num_reg_fcs=2,
        num_cls_fcs=2,
        agent_num_propagated=50,
        map_num_propagated=50,
        memory_len_frame=10,
        agent_num_query=900,
        agent_num_query_sifted=900,
        fut_mode=6,
        fut_ego_mode=1,
        fut_ts=6,
        fut_ego_fix_dist=True,
        fut_ts_ego_fix_dist=20,
        fut_ts_ego_fix_time=30,
        num_classes=9,
        code_size=10,
        map_num_query=100,
        map_num_query_sifted=100,
        map_num_classes=6,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        with_box_refine=True,
        LID=True,
        with_ego_pos=True,
        position_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        depth_start=1,
        depth_step=0.8,
        depth_num=64,
        agent_prep_decoder=dict(
            type='DriveTransformerPreDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='DriveTransformerPreDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='AttentionLayer',
                        embed_dims=768,
                        head_dim=64,
                        attn_drop=0.1,
                        extra_post_norm=True),
                    dict(
                        type='AttentionLayer',
                        embed_dims=768,
                        head_dim=64,
                        attn_drop=0.1,
                        extra_post_norm=True)
                ],
                ffn_cfgs=dict(
                    type='SwiGLULayer',
                    embed_dims=768,
                    feedforward_channels=3072,
                    ffn_drop=0.1,
                    extra_post_norm=True),
                with_cp=True,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))),
        map_prep_decoder=dict(
            type='DriveTransformerPreDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='DriveTransformerPreDecoderLayer',
                attn_cfgs=[
                    dict(
                        type='AttentionLayer',
                        embed_dims=768,
                        head_dim=64,
                        attn_drop=0.1,
                        extra_post_norm=True),
                    dict(
                        type='AttentionLayer',
                        embed_dims=768,
                        head_dim=64,
                        attn_drop=0.1,
                        extra_post_norm=True)
                ],
                ffn_cfgs=dict(
                    type='SwiGLULayer',
                    embed_dims=768,
                    feedforward_channels=3072,
                    ffn_drop=0.1,
                    extra_post_norm=True),
                with_cp=True,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))),
        transformer=dict(
            type='DriveTransformerWrapper',
            embed_dims=768,
            decoder=dict(
                type='DriveTransformerDecoder',
                num_layers=12,
                fut_mode=6,
                agent_num_query=900,
                map_num_query=100,
                map_num_pts_per_vec=20,
                return_intermediate=True,
                embed_dims=768,
                refine=True,
                transformerlayers=dict(
                    type='DriveTransformerDecoderLayer',
                    agent_query_num=900,
                    map_query_num=100,
                    memory_len_frame=10,
                    agent_num_propagated=50,
                    map_num_propagated=50,
                    map_pts_per_vec=20,
                    feedforward_channels=3072,
                    ffn_dropout=0.1,
                    with_cp=True,
                    attn_cfgs=[
                        dict(
                            type='AttentionLayer',
                            embed_dims=768,
                            head_dim=64,
                            attn_drop=0.1,
                            layer_scale=0.01,
                            extra_post_norm=True),
                        dict(
                            type='AttentionLayer',
                            embed_dims=768,
                            head_dim=64,
                            attn_drop=0.1,
                            layer_scale=0.01,
                            extra_post_norm=True),
                        dict(
                            type='AttentionLayer',
                            embed_dims=768,
                            head_dim=64,
                            attn_drop=0.1,
                            no_wq=True,
                            extra_post_norm=True)
                    ],
                    ffn_cfgs=dict(
                        type='SwiGLULayer',
                        embed_dims=768,
                        feedforward_channels=3072,
                        ffn_drop=0.1,
                        extra_post_norm=True),
                    operation_order=('task_self_attn', 'norm',
                                     'temporal_cross_attn', 'norm',
                                     'sensor_cross_attn', 'norm', 'ffn',
                                     'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-20, -35, -10.0, 20, 35, 10.0],
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            max_num=100,
            voxel_size=[0.15, 0.15, 4],
            num_classes=9),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.5,
            loss_weight=0.2),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 4],
            num_classes=6),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg_fix_time=dict(type='L1Loss', loss_weight=3.5),
        loss_plan_reg_fix_dist=dict(type='L1Loss', loss_weight=10.0),
        loss_plan_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.5,
            loss_weight=20.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 4],
            point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(
                    type='FocalLossCost', weight=2.0, gamma=2.0, alpha=0.25),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]),
            map_assigner=dict(
                type='MapHungarianAssigner3D',
                cls_cost=dict(
                    type='FocalLossCost', weight=2.0, gamma=2.0, alpha=0.25),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]))))
info_root = '/root/autodl-tmp/Bench2DriveZoo/data/infos'
map_root = '/root/autodl-tmp/Bench2DriveZoo/data/bench2drive/maps'
map_file = '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_map_infos.pkl'
ann_file_train = '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_train_drivetransformer_meta.pkl'
ann_file_val = '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_val_drivetransformer_meta.pkl'
ann_file_test = '/root/autodl-tmp/Bench2DriveZoo/data/infos/b2d_infos_v1_val_drivetransformer_meta.pkl'
optimizer = dict(type='AdamW', lr=2e-05, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=0.01)
runner = dict(type='IterBasedRunner', max_iters=225000)
fp16 = dict(loss_scale=512.0)
find_unused_parameters = True
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
gpu_ids = range(0, 4)
