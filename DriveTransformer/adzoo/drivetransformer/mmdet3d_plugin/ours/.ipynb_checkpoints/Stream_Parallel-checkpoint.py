import time
import copy

import torch
#from mmdet.models import DETECTORS
from mmcv.models.builder import DETECTORS
#from mmdet3d.core import bbox3d2result
from mmcv.core.bbox import bbox3d2result
#from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
#from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector

from adzoo.drivetransformer.mmdet3d_plugin.models.utils.grid_mask import GridMask
from .metric_stp3 import PlanningMetric
from collections import OrderedDict
import numpy as np

@DETECTORS.register_module()
class StreamParallel(MVXTwoStageDetector):
    """VAD model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6,
                 ):

        super(StreamParallel,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        #self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_scene_token = None
        self.planning_metric = None
        self.test_flag = False
        self.position_level = 0
    
    @force_fp32()
    def forward(self, data, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)   

    @auto_fp16()
    def extract_img_feat(self, img, img_metas=None, len_queue=1):
        """Extract features from images and points."""
        B = img.size(0)
        if img is not None:
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    @force_fp32()
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_traj_fut_classes=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs_fix_time=None,
                      ego_fut_trajs_fix_dist=None,
                      ego_fut_classes=None,
                      ego_fut_masks_fix_time=None,
                      ego_fut_masks_fix_dist=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      **data,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False
        img_feats = self.extract_img_feat(img)
        data_t = dict()
        if 'timestamp' in data:
            if isinstance(data['timestamp'],list):
                data['timestamp'] = np.concatenate(data['timestamp'])
        for key in data:
            if len(data[key].shape) == 1: 
                data_t[key] = data[key]
                continue
            data_t[key] = data[key][:, 0]
        img_feats = img_feats[self.position_level][:, 0]
        
        preds_dicts = self.pts_bbox_head(img_feats, img_metas, ego_lcf_feat, ego_fut_cmd, ego_his_trajs, **data_t)
        losses = self.pts_bbox_head.loss(
            ## Labels
            gt_bboxes_list=gt_bboxes_3d,
            gt_labels_list=gt_labels_3d,
            gt_traj_fut_classes=gt_traj_fut_classes,
            map_gt_bboxes_list=map_gt_bboxes_3d,
            map_gt_labels_list=map_gt_labels_3d,
            ego_fut_gt_fix_time=ego_fut_trajs_fix_time,
            ego_fut_masks_fix_time=ego_fut_masks_fix_time,
            ego_fut_gt_fix_dist=ego_fut_trajs_fix_dist,
            ego_fut_masks_fix_dist=ego_fut_masks_fix_dist,
            ego_fut_cmd=ego_fut_cmd,
            ego_fut_classes=ego_fut_classes,
            gt_attr_labels=gt_attr_labels,
            ## Prediction
            preds_dicts=preds_dicts,
            img_metas=img_metas
        )
        return self._parse_losses(losses, num_samples=img.shape[0])

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs_fix_time=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        fut_valid_flag_fix_time=None,
        **data,
    ):
        ## Prepare Inputs        
        for key in data:
            if key == "map_gt_bboxes_3d":
                data[key] = data[key][0]
                continue
            # in case their is a nested list
            if not isinstance(data[key], list):
                continue
            data[key] = data[key][0]
            if not isinstance(data[key], list):
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key].reshape(1,-1)
                else:
                    data[key] = data[key].unsqueeze(0)
            else:
                data[key] = data[key][0].unsqueeze(0)
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = img[0].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = img[0].new_ones(1)   
        if ego_his_trajs is not None:
            ego_his_trajs = ego_his_trajs[0]      
        if ego_fut_cmd is not None:
            ego_fut_cmd = ego_fut_cmd[0]    
        if ego_lcf_feat is not None:
            ego_lcf_feat = ego_lcf_feat[0]            

        ## Inference
        img_feats = self.extract_img_feat(img=img.unsqueeze(0), img_metas=img_metas)[self.position_level][:, 0]
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]
        outs = self.pts_bbox_head(img_feats, img_metas, ego_lcf_feat, ego_fut_cmd, ego_his_trajs, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = []
        #import pdb;pdb.set_trace()
        for i, (bboxes, scores, labels, trajs, map_bboxes, \
                map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result['trajs_3d'] = trajs.cpu()
            map_bbox_result = dict(
            map_boxes_3d=map_bboxes.to('cpu'),
            map_scores_3d=map_scores.cpu(),
            map_labels_3d=map_labels.cpu(),
            map_pts_3d=map_pts.to('cpu'))
            bbox_result.update(map_bbox_result)
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_result['ego_fut_preds_fix_time'] = outs['ego_fut_preds_fix_time'][-1].cpu()
            if 'ego_traj_cls_scores' in outs and outs['ego_traj_cls_scores'] is not None:
                bbox_result['ego_traj_cls_scores'] = outs['ego_traj_cls_scores'][-1].cpu()
            if 'ego_fut_preds_fix_dist' in outs and outs['ego_fut_preds_fix_dist'] is not None:
                bbox_result['ego_fut_preds_fix_dist'] = outs['ego_fut_preds_fix_dist'][-1].cpu()
            bbox_results.append(bbox_result)

        if gt_attr_labels is None:
            return bbox_results
        
        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        score_threshold = 0.6
        match_dis_thresh= 2.0
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)
            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0]
            gt_label = gt_labels_3d[0].to('cpu')
            gt_attr_label = gt_attr_labels[0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag_fix_time[0])
            # filter pred bbox by score_threshold
            mask = bbox_result['scores_3d'] > score_threshold
            bbox_result['boxes_3d'] = bbox_result['boxes_3d'][mask]
            bbox_result['scores_3d'] = bbox_result['scores_3d'][mask]
            bbox_result['labels_3d'] = bbox_result['labels_3d'][mask]
            bbox_result['trajs_3d'] = bbox_result['trajs_3d'][mask]
            matched_bbox_result = self.assign_pred_to_gt_vip3d(
                bbox_result, gt_bbox, gt_label, match_dis_thresh=match_dis_thresh)
            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result,
                matched_bbox_result, mapped_class_names)

            # ego planning metric
            # assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'

            ego_fut_preds = bbox_result['ego_fut_preds_fix_time'][0]
            ego_fut_trajs = ego_fut_trajs_fix_time.to('cpu')
            # ego_fut_preds_cls = bbox_result['ego_traj_cls_scores'][0]
            # ego_fut_cmd = ego_fut_cmd[0, 0]
            # ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0]
            # _, selected_idx = torch.max(ego_fut_preds_cls, dim=0)
            # ego_fut_pred = ego_fut_preds[selected_idx]
            ego_fut_preds = ego_fut_preds[:,4::5,:]
            ego_fut_trajs = ego_fut_trajs[:,4::5,:]
            
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_preds,
                gt_ego_fut_trajs = ego_fut_trajs,
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)
            
        output_bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(output_bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict
        return output_bbox_list
    
    @force_fp32()
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)
        

        pred_ego_fut_trajs_xy = pred_ego_fut_trajs
        gt_ego_fut_trajs_xy = gt_ego_fut_trajs
        
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs_xy[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs_xy[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs_xy[:, :cur_time].detach(),
                    gt_ego_fut_trajs_xy[:, :cur_time],
                    occupancy)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean().item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
            
        return metric_dict
    
    @force_fp32()
    def assign_pred_to_gt_vip3d(
        self,
        bbox_result,
        gt_bbox,
        gt_label,
        match_dis_thresh=2.0
    ):
        """Assign pred boxs to gt boxs according to object center preds in lcf.
        Args:
            bbox_result (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
        """     
        dynamic_list = [0,1,3,4,6,7,8]
        matched_bbox_result = torch.ones(
            (len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result['boxes_3d'].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result['labels_3d']]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6
        r_list, c_list = linear_sum_assignment(dist)
        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]
        return matched_bbox_result

    def rotate_agent_trajs_to_ego(self, bbox_yaw, trajs):
        bbox_yaw = -bbox_yaw-torch.pi/2
        yaws_to_rotate = bbox_yaw - torch.pi/2
        rot_matrix = torch.stack([torch.cos(yaws_to_rotate), -torch.sin(yaws_to_rotate), torch.sin(yaws_to_rotate), torch.cos(yaws_to_rotate)],dim=-1).reshape(-1,2,2)
        if len(trajs.shape) == 2:
            return (rot_matrix.unsqueeze(0) @ trajs.unsqueeze(-1)).squeeze(-1)
        if len(trajs.shape) == 3:
            return (rot_matrix.unsqueeze(0) @ trajs.unsqueeze(-1)).squeeze(-1)
    
    @force_fp32()
    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        """Compute EPA metric for one sample.
        Args:
            gt_bboxs (LiDARInstance3DBoxes): GT Bboxs.
            gt_label (Tensor): GT labels for gt_bbox, [num_gt_bbox].
            pred_bbox (dict): Predictions.
                'boxes_3d': (LiDARInstance3DBoxes)
                'scores_3d': (Tensor), [num_pred_bbox]
                'labels_3d': (Tensor), [num_pred_bbox]
                'trajs_3d': (Tensor), [fut_ts*2]
            matched_bbox_result (np.array): assigned pred index for each gt box [num_gt_bbox].
            match_dis_thresh (float): dis thresh for determine a positive sample for a gt bbox.

        Returns:
            EPA_dict (dict): EPA metric dict of each cared class.
        """
        motion_cls_names = ['car', 'pedestrian']
        motion_metric_names = ['gt', 'cnt_ade', 'cnt_fde', 'hit',
                               'fp', 'ADE', 'FDE', 'MR']
        
        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met+'_'+cls] = 0.0

        veh_list = [0,1,3,4]
        ignore_list = ['construction_vehicle', 'barrier',
                       'traffic_cone', 'motorcycle', 'bicycle']

        for i in range(pred_bbox['labels_3d'].shape[0]):
            pred_bbox['labels_3d'][i] = 0 if pred_bbox['labels_3d'][i] in veh_list else pred_bbox['labels_3d'][i]
            box_name = mapped_class_names[int(pred_bbox['labels_3d'][i])]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict['fp_'+box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[int(gt_label[i])]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts*2:self.fut_ts*3]
            num_valid_ts = sum(gt_fut_masks==1)
            if num_valid_ts == self.fut_ts:
                metric_dict['gt_'+box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                
                metric_dict['cnt_ade_'+box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][:self.fut_ts*2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox['trajs_3d'][int(m_pred_idx)].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                #gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                #pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                #gt_fut_trajs = self.rotate_agent_trajs_to_ego(bbox_yaw=gt_bbox[i].yaw, trajs=gt_fut_trajs)
                pred_fut_trajs = self.rotate_agent_trajs_to_ego(bbox_yaw=pred_bbox['boxes_3d'][int(m_pred_idx)].yaw, trajs=pred_fut_trajs)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox['boxes_3d'][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict['ADE_'+box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict['cnt_fde_'+box_name] += 1
                    metric_dict['FDE_'+box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict['hit_'+box_name] += 1
                    else:
                        metric_dict['MR_'+box_name] += 1
                        
                #import pdb;pdb.set_trace()

        return metric_dict



    @force_fp32()
    def _parse_losses(self, losses, num_samples):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()
        log_vars['loss'] = loss.item()
        return {"loss":loss, "log_vars": log_vars, "num_samples":num_samples}
    
    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch