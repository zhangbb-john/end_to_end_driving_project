import copy
import numpy as np
import os
from os import path as osp
import torch
import random
import json, pickle
import tempfile
import cv2
from pyquaternion import Quaternion
import mmcv
from mmcv.datasets import DATASETS
from mmcv.utils import save_tensor
from mmcv.parallel import DataContainer as DC
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.fileio.io import load, dump
from mmcv.utils import track_iter_progress, mkdir_or_exist
from mmcv.datasets.pipelines import to_tensor
from mmcv.datasets.custom_3d import Custom3DDataset
from mmcv.datasets.pipelines import Compose
from mmcv.datasets.map_utils.struct import LiDARInstanceLines
from shapely.geometry import LineString
import joblib
import time
from matplotlib import pyplot as plt


@DATASETS.register_module()
class B2D_DriveTransformer_Dataset(Custom3DDataset):
    def __init__(self, 
                 with_velocity=True, # bounding boxes with velocity attribute.
                 sample_interval=5,  # sample interval for ego history and agent trajectories. Bench2Drive data is 10hz, interval=5 means downsampling to 2hz.
                 sample_interval_ego_fut=2, # sample interval for ego future trajectories formed with fixed time interval.
                 name_mapping=None, # Bench2Drive object class mapping.
                 map_file=None, # file path of map annotation.
                 past_frames=2, # number of past frames.
                 future_frames=6, # number of future frames for agents.
                 future_frames_ego_fix_time=15, # number of future frames for ego future trajectories formed with fixed time interval.
                 future_frames_ego_fix_dist=20, # number of future frames for ego future trajectories formed with fixed dist interval.
                 fix_future_dis=1, # distance interval of ego future trajectories. 
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 polyline_points_num=20, # number of points per polyline.
                 data_aug_conf=None, # data augmentation configuration.
                 collect_keys=['lidar2img', 'cam_intrinsic', 'cam_extrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape', 'gt_traj_fut_classes', 'ego_fut_classes'], # keys to collect after pipeline.
                 use_angle_as_dis_traj=False, # output ego future trajectories with fixed dist interval as angles if true, output as x in lidar coordinate system if flase.
                 sub_seq_lenth=20, # max sequence lenth for sampler. -1 means using whole clip as a sequence.
                 use_splited_data=True, # To reduce time of initialization and save memory, we do not read data of all clips at beginning (like UniAD/VAD reproduced in Bench2DriveZoo).
                                        # instead, we read a clip data if it is needed during training.
                 cache_lenth=4, # the number of clip data cached in memory. 
                 *args,
                 **kwargs):
        
        print('loading dataset...')
        start_time = time.time()
        self.use_splited_data = use_splited_data
        super().__init__(*args, **kwargs)
        self.with_velocity = with_velocity 
        self.NameMapping  = name_mapping
        self.sample_interval = sample_interval
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.map_file = map_file
        self.sample_interval_ego_fut = sample_interval_ego_fut
        self.future_frames_ego_fix_time = future_frames_ego_fix_time
        self.future_frames_ego_fix_dist = future_frames_ego_fix_dist
        self.point_cloud_range = np.array(point_cloud_range)
        self.polyline_points_num = polyline_points_num
        self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
        self.MAPCLASSES = list(self.map_element_class.keys())
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.map_eval_use_same_gt_sample_num_flag = True
        self.use_angle_as_dis_traj = use_angle_as_dis_traj

        map_start_time = time.time()    
        with open(self.map_file,'rb') as f: 
            self.map_infos = joblib.load(f)
        map_end_time = time.time()    
        print('loading map infos: '+str(map_end_time-map_start_time)+'s')

        self.fix_future_dis = fix_future_dis
        self.collect_keys = collect_keys 
        self.data_aug_conf = data_aug_conf
        self.sub_seq_lenth = sub_seq_lenth
        
        # for clip data cache
        self.cache_lenth = cache_lenth
        self.current_route_name = None
        self.current_route_data = None
        self.current_route_start_idx = None
        self.current_route_end_idx = None
        self.cached_route_start_idx = [None] * self.cache_lenth
        self.cached_route_end_idx = [None] * self.cache_lenth       
        self.cached_route_names = [None] * self.cache_lenth
        self.cached_data = [None] * self.cache_lenth
        self.visit_time = np.zeros(self.cache_lenth,dtype=np.int64)
        
        if self.use_splited_data:
            self.infos_dir_name = self.data_infos['infos_dir_name'].replace('dtr','drivetransformer') # dictionary name that contain the clips
            self.routes_names = self.data_infos['routes_names'] # name of each clip
            self.divide_nums = self.data_infos['divide_nums'] # indexs of division points of clips
            self.flag = self._set_sequence_group_flag_with_split_data()
        else:
            self.flag = self._set_sequence_group_flag()
        
        end_time = time.time()
        print('finish loading. dataset lenth: '+str(len(self.flag)) +' loading time: '+str(end_time-start_time)+'s')
        
        
    def __len__(self):
        if self.use_splited_data:
            return self.data_infos['divide_nums'][-1]
        else:
            return len(self.data_infos)
        

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group. Pass the flags to InfiniteGroupEachSampleInBatchSampler.
        """
        pre_folder = self.data_infos[0]['folder']
        pre_idx = 0
        total_sub_sequence_num = 0
        total_flag = []

        for i,data_info in enumerate(self.data_infos):
            if data_info['folder'] != pre_folder or i==len(self.data_infos)-1: # find a new clip
                if i==len(self.data_infos)-1:
                    sequence_lenth = i-pre_idx+1 
                else:
                    sequence_lenth = i-pre_idx
                if self.sub_seq_lenth < 0: # assgin the same group number for whole clip
                    total_flag.extend([total_sub_sequence_num]*sequence_lenth)
                    total_sub_sequence_num+=1
                else: # split the clip to sub_sequence
                    each_sequence_lenth = [self.sub_seq_lenth] * (sequence_lenth // self.sub_seq_lenth) 
                    if sequence_lenth % self.sub_seq_lenth !=0:
                        each_sequence_lenth.append(sequence_lenth % self.sub_seq_lenth)
                    for j in range(len(each_sequence_lenth)):
                        total_flag.extend([total_sub_sequence_num]*each_sequence_lenth[j])
                        total_sub_sequence_num += 1

                pre_folder = data_info['folder']
                pre_idx = i
            
        return np.array(total_flag, dtype=np.int64)
    
    
    def _set_sequence_group_flag_with_split_data(self):
        """
        Set each sequence to be a different group. A version for data infos splited by clips.
        """
        pre_divide = 0
        total_sub_sequence_num = 0
        total_flag = []

        for divide in self.divide_nums:
            sequence_lenth = divide - pre_divide
            if self.sub_seq_lenth < 0:
                total_flag.extend([total_sub_sequence_num]*sequence_lenth)
                total_sub_sequence_num+=1
            else:
                each_sequence_lenth = [self.sub_seq_lenth] * (sequence_lenth // self.sub_seq_lenth)
                if sequence_lenth % self.sub_seq_lenth !=0:
                    each_sequence_lenth.append(sequence_lenth % self.sub_seq_lenth)
                for j in range(len(each_sequence_lenth)):
                    total_flag.extend([total_sub_sequence_num]*each_sequence_lenth[j])
                    total_sub_sequence_num += 1
            pre_divide = divide
            
        return np.array(total_flag, dtype=np.int64)
    
    
    def get_data_by_index(self, index):
        """
        Read data with a clip cache. 
        """        
        if not self.use_splited_data:
            return self.data_infos[index]
        else:
            if self.current_route_start_idx is not None and index >= self.current_route_start_idx and index < self.current_route_end_idx:
                return self.current_route_data[index-self.current_route_start_idx] # hit the cache
            # update cache with LRU policy
            for i in range(self.cache_lenth):
                if self.cached_route_start_idx[i] is None:
                    continue
                elif index >= self.cached_route_start_idx[i] and index < self.cached_route_end_idx[i]:
                    self.current_route_name = self.cached_route_names[i]
                    self.current_route_end_idx = self.cached_route_end_idx[i]
                    self.current_route_start_idx = self.cached_route_start_idx[i]
                    self.current_route_data = self.cached_data[i]
                    self.visit_time += 1
                    self.visit_time[i] = 0
                    return self.current_route_data[index-self.current_route_start_idx]
            route_idx = 0
            for divide in self.divide_nums:
                if index >= divide:
                    route_idx+=1
                else:
                    break
            self.current_route_name = self.routes_names[route_idx]
            self.current_route_start_idx = self.divide_nums[route_idx-1] if route_idx>0 else 0
            self.current_route_end_idx =  self.divide_nums[route_idx]     
            with open(os.path.join('data','infos',self.infos_dir_name,self.current_route_name+'.pkl'),'rb') as f:
                self.current_route_data = pickle.load(f)
            replace_idx = np.argmax(self.visit_time)
            self.cached_route_names[replace_idx] = self.current_route_name
            self.cached_route_end_idx[replace_idx] = self.current_route_end_idx
            self.cached_route_start_idx[replace_idx] = self.current_route_start_idx
            self.cached_data[replace_idx] = self.current_route_data
            self.visit_time += 1
            self.visit_time[replace_idx] = 0

            return self.current_route_data[index-self.current_route_start_idx]
        
    def invert_pose(self, pose):
        """
        Get the inverse of a transform matrix.
        """
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def prepare_train_data(self, index, aug_config):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        input_dict["aug_config"] = aug_config
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        gt_labels,gt_bboxes = self.get_map_info(index)
        example['map_gt_labels_3d'] = DC(gt_labels, cpu_only=False)
        example['map_gt_bboxes_3d'] = DC(gt_bboxes, cpu_only=True)
        if self.filter_empty_gt and (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
            
        return self.union2one([example])
    
    def prepare_test_data(self, index, aug_config):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        input_dict["aug_config"] = aug_config
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        gt_labels,gt_bboxes = self.get_map_info(index)
        example['map_gt_labels_3d'] = DC(gt_labels, cpu_only=False)
        example['map_gt_bboxes_3d'] = DC(gt_bboxes, cpu_only=True) 
        return example


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        if self.collect_keys is not None:
            for key in self.collect_keys:
                if key == 'img_metas':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                elif key == 'timestamp':
                    queue[-1][key] = DC(np.stack([each[key].data for each in queue]), cpu_only=True , stack=False, pad_dims=None)
                else:    
                    queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.get_data_by_index(index)

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            frame_idx=info['frame_idx'],
            ego_yaw=np.nan_to_num(info['ego_yaw'],nan=np.pi/2),
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            world2lidar=info['sensors']['LIDAR_TOP']['world2lidar'],
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'],
            gt_names=info['gt_names'],
            ego_vel=info['ego_vel'].astype(np.float32),
            ego_accel=info['ego_accel'].astype(np.float32),
            ego_rotation_rate=info['ego_rotation_rate'].astype(np.float32),
            npc2world=info['npc2world'].astype(np.float32),
            timestamp=info['frame_idx']/2,
        )
        
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            lidar2global =  self.invert_pose(info['sensors']['LIDAR_TOP']['world2lidar'])
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root,cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    cam_extrinsic=lidar2cam_rts,
                    l2g_r_mat=lidar2global[0:3,0:3],
                    l2g_t=lidar2global[0:3,3]

                ))
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=yaw).rotation_matrix
        ego2world[0:3,3] = input_dict['ego_translation']
        lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
        lidar2world = ego2world @ lidar2ego
        world2lidar = self.invert_pose(lidar2world)
        input_dict['ego_pose'] = lidar2world
        input_dict['ego_pose_inv'] = world2lidar
        # normalize yaw
        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180 
        # can bus
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        # ego_lcf_feat
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0:2] = input_dict['ego_vel'][0:2]
        ego_lcf_feat[2:4] = input_dict['ego_accel'][2:4]
        ego_lcf_feat[4] = input_dict['ego_rotation_rate'][-1]
        ego_lcf_feat[5] = info['ego_size'][1]
        ego_lcf_feat[6] = info['ego_size'][0]
        ego_lcf_feat[7] = info['ego_vel'][0]
        ego_lcf_feat[8] = info['steer']
        # command. embedding of far and near command type and point.
        command = np.zeros(140)
        command[0:6] = self.command2hot(info['command_far'])
        command[6:70] =  self.pos2posemb(self.get_command_xy_in_local(info['command_far_xy'],info['ego_translation'][0:2],yaw))
        command[70:76] = self.command2hot(info['command_near'])
        command[76:140] = self.pos2posemb(self.get_command_xy_in_local(info['command_near_xy'],info['ego_translation'][0:2],yaw))
        # ego history and future
        ego_his_trajs = self.get_ego_past_trajs(index,self.sample_interval,self.past_frames)
        ego_fut_trajs_fix_time, ego_fut_masks_fix_time = self.get_ego_future_trajs(index,self.sample_interval_ego_fut,self.future_frames_ego_fix_time)        
        ego_fut_trajs_fix_dist, ego_fut_masks_fix_dist = self.get_ego_future_trajs_fix_dis(index,1,self.future_frames_ego_fix_dist,self.use_angle_as_dis_traj)
        
        input_dict['ego_his_trajs'] = ego_his_trajs
        input_dict['ego_fut_trajs_fix_time'] = ego_fut_trajs_fix_time
        input_dict['ego_fut_masks_fix_time'] = ego_fut_masks_fix_time
        input_dict['ego_fut_trajs_fix_dist'] = ego_fut_trajs_fix_dist
        input_dict['ego_fut_masks_fix_dist'] = ego_fut_masks_fix_dist
        input_dict['ego_fut_cmd'] = command
        input_dict['ego_lcf_feat'] = ego_lcf_feat
        input_dict['fut_valid_flag_fix_time'] = input_dict['ego_fut_masks_fix_time'][-1]
        input_dict['fut_valid_flag_fix_dist'] = input_dict['ego_fut_masks_fix_dist'][-1]
        prev_exists = not (index == 0 or self.flag[index - 1] != self.flag[index]) 
        input_dict['index'] = index
        input_dict['prev_exists'] = prev_exists
        return input_dict
    
    def get_command_xy_in_local(self, command_xy, ego_xy, ego_theta):
        """Transform command points into lidar coordinate system

        """        
        theta_to_lidar = -ego_theta + np.pi/2
        rotate_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        command_xy_in_local = rotate_matrix @ (command_xy-ego_xy)
        return command_xy_in_local
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot

    def pos2posemb(self,pos, num_pos_feats=32, temperature=10000):
        scale = 2 * np.pi
        pos = pos * scale
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = temperature ** (2 * (dim_t//2) / num_pos_feats)
        pos_tmp = pos[..., None] / dim_t
        posemb = np.stack((np.sin(pos_tmp[..., 0::2]), np.cos(pos_tmp[..., 1::2])), axis=-1)
        return posemb.reshape(-1)


    def get_map_info(self, index):

        """Get map annotation info according to the given index.
        Args:
            index (int): Index of the annotation data to get.        
        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes (:obj:`LiDARInstance3DBoxes`): \
                    ground truth bboxes
                - gt_labels (np.ndarray): Labels of ground truths.   
        """

        gt_masks = []
        gt_labels = []
        gt_bboxes = []
        ann_info = self.get_data_by_index(index)
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2,3]
        max_distance = 50
        chosed_idx = []
        # hierarchical search to get map elements in certain range
        # Roughly search the lines with downsampled points and a larger range
        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx)
        # precisely search and crop the lines of dense points
        polylines = []
        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys():
                continue
            points = lane_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = (points_in_lidar[:,0] > self.point_cloud_range[0]) & (points_in_lidar[:,0] < self.point_cloud_range[3]) & (points_in_lidar[:,1] > self.point_cloud_range[1]) & (points_in_lidar[:,1] < self.point_cloud_range[4])
            points_in_lidar_range = points_in_lidar[mask,0:2]
            if len(points_in_lidar_range) > 1:
                polylines.append(LineString(points_in_lidar_range))
                gt_label = self.map_element_class[lane_types[idx]]
                gt_labels.append(gt_label)
        # for trigger_volumes
        for idx in range(len(trigger_volumes_points)):
            if not trigger_volumes_types[idx] in self.map_element_class.keys():
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))], axis=-1)
            points_in_lidar = (world2lidar @ points.T).T
            mask = (points_in_lidar[:,0] > self.point_cloud_range[0]) & (points_in_lidar[:,0] < self.point_cloud_range[3]) & (points_in_lidar[:,1] > self.point_cloud_range[1]) & (points_in_lidar[:,1] < self.point_cloud_range[4])
            points_in_lidar_range = points_in_lidar[mask,0:2]
            if mask.all():
                polylines.append(LineString(np.concatenate((points_in_lidar_range, points_in_lidar_range[0:1]),axis=0)))
                gt_label = self.map_element_class[trigger_volumes_types[idx]]
                gt_labels.append(gt_label)
        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = LiDARInstanceLines(polylines,fixed_num=self.polyline_points_num,patch_size=(self.point_cloud_range[4]-self.point_cloud_range[1],self.point_cloud_range[3]-self.point_cloud_range[0]))
        return gt_labels, gt_bboxes



    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.get_data_by_index(index)

        # filter out bbox containing no points

        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]

        mask = (info['num_points'] != 0)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_inds = info['gt_ids']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        if not self.with_velocity:
            gt_bboxes_3d = gt_bboxes_3d[:,0:7]
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        attr_labels = self.get_box_attr_labels(index,self.sample_interval,self.future_frames)
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            attr_labels=attr_labels[mask],
            )
        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        aug_config = self.get_augmentation()
        if self.test_mode:
            return self.prepare_test_data(idx, aug_config)
        while True:
            data = self.prepare_train_data(idx, aug_config)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
        
        
    def is_in_same_route(self, cur_idx, adj_idx):
        
        if adj_idx <0 or adj_idx>=len(self):
            return False
        if self.use_splited_data:
            return (adj_idx >= self.current_route_start_idx and adj_idx < self.current_route_end_idx)
        else:
            return self.data_infos[cur_idx]['folder'] == self.data_infos[adj_idx]['folder']
        

    def get_ego_past_trajs(self, idx, sample_rate, past_frames):
        # get past differential trajectory of ego.
        adj_idx_list = range(idx-past_frames*sample_rate,idx+sample_rate,sample_rate)
        cur_frame = self.get_data_by_index(idx)
        full_adj_track = np.zeros((past_frames+1,2))
        full_adj_adj_mask = np.zeros(past_frames+1)
        lidar2ego = cur_frame['sensors']['LIDAR_TOP']['lidar2ego']
        ego2world = np.eye(4)
        ego2world[0:2,3] = cur_frame['ego_translation'][0:2]
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=cur_frame['ego_yaw']).rotation_matrix
        lidar2world = ego2world @ lidar2ego
        world2lidar_lidar_cur = self.invert_pose(lidar2world)
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if not self.is_in_same_route(idx,adj_idx): # end if noit in the same route
                break
            adj_frame = self.get_data_by_index(adj_idx)
            ego2world_adj = np.eye(4)
            ego2world_adj[0:2,3] = adj_frame['ego_translation'][0:2]
            ego2world_adj[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=adj_frame['ego_yaw']).rotation_matrix
            lidar2world_ego_adj = ego2world_adj @ lidar2ego
            adj2cur_lidar = world2lidar_lidar_cur @ lidar2world_ego_adj
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_adj_mask[j] = 1
        offset_track = full_adj_track[1:] - full_adj_track[:-1]
        for j in range(past_frames-2,-1,-1):
            if full_adj_adj_mask[j] == 0:
                offset_track[j] = offset_track[j+1]
        
        return offset_track


    def get_ego_future_trajs_fix_dis(self,idx,sample_rate,future_frames,use_angle=False):
        # get future trajectory of ego with fixed interval of distance
        cur_frame = self.get_data_by_index(idx)
        full_adj_track = np.zeros((future_frames,2))
        full_adj_mask = np.zeros(future_frames)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        pre_xy = np.zeros(2)
        sampled_num = 0
        pre_dis = 0
        all_node_list = []
        
        while True:
            idx += sample_rate
            if idx <0 or idx>=len(self):
                break            
            if self.current_route_start_idx is not None and (idx < self.current_route_start_idx or idx >= self.current_route_end_idx):
                break
            adj_frame = self.get_data_by_index(idx)
            if adj_frame['folder'] != cur_frame['folder']:
                break     
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            cur_xy = adj2cur_lidar[0:2,3]  
            dis = np.linalg.norm(cur_xy-pre_xy) # distance between two frames
            if (dis + pre_dis)> self.fix_future_dis: 
                num_samples = (dis + pre_dis) // self.fix_future_dis
                for i in range(int(num_samples)):
                    ratio = (self.fix_future_dis*(i+1) - pre_dis ) / dis #interpolation
                    sampled_xy = pre_xy + ratio * (cur_xy - pre_xy)
                    full_adj_track[sampled_num,0:2] = sampled_xy
                    full_adj_mask[sampled_num] = 1   
                    sampled_num += 1
                    if sampled_num >= future_frames:
                        break
                pre_dis = dis + pre_dis - self.fix_future_dis * num_samples 
                if sampled_num >= future_frames:
                    break
            else:
                pre_dis += dis
            pre_xy = cur_xy.copy()
        xs = full_adj_track[:,0] # x coordinates in lidar coordinate system
        if use_angle:
            xs = xs / (np.linalg.norm(full_adj_track,axis=-1) + 1e-9)  # output as angles to y-axis
        xs[~full_adj_mask.astype(bool)] = 0        
        return xs[:,None], full_adj_mask
    
    
    def get_ego_future_trajs(self,idx,sample_rate,future_frames):
        # get future trajectory of ego with fixed interval of time
        adj_idx_list = range(idx+sample_rate,idx+(future_frames+1)*sample_rate,sample_rate)
        cur_frame = self.get_data_by_index(idx)
        full_adj_track = np.zeros((future_frames,2))
        full_adj_mask = np.zeros(future_frames)
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if not self.is_in_same_route(idx,adj_idx):
                break
            adj_frame = self.get_data_by_index(adj_idx)
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_lidar_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            full_adj_track[j,0:2] = xy
            full_adj_mask[j] = 1
        full_adj_track[~full_adj_mask.astype(bool)] = 0
        return full_adj_track, full_adj_mask

    def get_box_attr_labels(self,idx,sample_rate,frames):
        # get attributes of boxes
        adj_idx_list = range(idx,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.get_data_by_index(idx)
        cur_box_names = cur_frame['gt_names']
        for i in range(len(cur_box_names)):
            if cur_box_names[i] in self.NameMapping.keys():
                cur_box_names[i] = self.NameMapping[cur_box_names[i]]
        cur_boxes = cur_frame['gt_boxes'].copy()
        box_ids = cur_frame['gt_ids']
        future_track = np.zeros((len(box_ids),frames+1,2))
        future_mask = np.zeros((len(box_ids),frames+1))
        future_yaw = np.zeros((len(box_ids),frames+1))
        gt_fut_goal = np.zeros((len(box_ids),1))
        agent_lcf_feat = np.zeros((len(box_ids),9))
        world2lidar_lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            agent_lcf_feat[i,0:2] = cur_boxes[i,0:2]
            agent_lcf_feat[i,2] = cur_boxes[i,6]
            agent_lcf_feat[i,3:5] = cur_boxes[i,7:]
            agent_lcf_feat[i,5:8] = cur_boxes[i,3:6]
            cur_box_name = cur_box_names[i]
            if cur_box_name in self.CLASSES:
                agent_lcf_feat[i, 8] = self.CLASSES.index(cur_box_name)
            else:
                agent_lcf_feat[i, 8] = -1
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_lidar_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]      
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx <0 or adj_idx>=len(self):
                    break
                if self.current_route_start_idx is not None and (idx < self.current_route_start_idx or idx >= self.current_route_end_idx):
                    break
                adj_frame = self.get_data_by_index(adj_idx)
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                if len(np.where(adj_frame['gt_ids']==box_id)[0])==0:
                    continue
                assert len(np.where(adj_frame['gt_ids']==box_id)[0]) == 1 , np.where(adj_frame['gt_ids']==box_id)[0]
                adj_idx = np.where(adj_frame['gt_ids']==box_id)[0][0]
                adj_box2lidar = world2lidar_lidar_cur @ adj_frame['npc2world'][adj_idx]
                adj_xy = adj_box2lidar[0:2,3]    
                future_track[i,j,:] = adj_xy
                future_mask[i,j] = 1
                future_yaw[i,j] = np.arctan2(adj_box2lidar[1,0],adj_box2lidar[0,0])
            coord_diff = future_track[i,-1] - future_track[i,0]
            if coord_diff.max() < 1.0: # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class
        future_track_offset = future_track[:,1:,:] - future_track[:,:-1,:]
        future_mask_offset = future_mask[:,1:]
        future_track_offset[future_mask_offset==0] = 0
        future_yaw_offset = future_yaw[:,1:] - future_yaw[:,:-1]
        mask1 = np.where(future_yaw_offset>np.pi)
        mask2 = np.where(future_yaw_offset<-np.pi)
        future_yaw_offset[mask1] -=np.pi*2 
        future_yaw_offset[mask2] +=np.pi*2
        attr_labels = np.concatenate([future_track_offset.reshape(-1,frames*2), future_mask_offset, gt_fut_goal, agent_lcf_feat, future_yaw_offset],axis=-1).astype(np.float32)
        return attr_labels.copy()

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int(
                    (1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"]))
                    * newH
                )
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config


