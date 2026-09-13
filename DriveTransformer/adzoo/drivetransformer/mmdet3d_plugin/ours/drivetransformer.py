import time
import copy

import torch
from mmcv.models.builder import DETECTORS
from mmcv.core.bbox import bbox3d2result
from mmcv.utils import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from adzoo.drivetransformer.mmdet3d_plugin.models.utils.grid_mask import GridMask
from collections import OrderedDict
import numpy as np

@DETECTORS.register_module()
class DriveTransformer(MVXTwoStageDetector):
    """
    DriveTransformer model. 
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
                 fut_ts=6,
                 fut_mode=6,
                 ):

        super(DriveTransformer,
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
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_traj_fut_classes=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      img=None,
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
            img_metas (list[dict], optional): Meta information of each sample.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional): Ground truth 3D boxes.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels of 3D boxes. 
            gt_traj_fut_classes(torch.Tensor): Assignd trajectory class.
            map_gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional): Ground truth of map boxes. 
            map_gt_labels_3d (list[torch.Tensor], optional): Ground truth labels of map boxes.         
            img (torch.Tensor optional): Images of each sample with shape (N, C, H, W). 
            ego_his_trajs(torch.Tensor): Historical trajectory of ego.
            ego_fut_trajs_fix_time(torch.Tensor): Future trajectory of ego formed in fixed time interval.
            ego_fut_trajs_fix_dist(torch.Tensor): Future trajectory of ego formed in fixed distance interval.
            ego_fut_classes(torch.Tensor): Assignd trajectory class for geo.
            ego_fut_masks_fix_time(torch.Tensor): Valid mask of ego trajectory with fixed time.
            ego_fut_masks_fix_dist(torch.Tensor): Valid mask of ego trajectory with fixed distance.
            ego_fut_cmd(torch.Tensor): Commond provided by planner.
            ego_lcf_feat(torch.Tensor): feature of ego
            gt_attr_labels(list[torch.Tensor]): CT attribute of boxes (trajctories of agents)
        Return:
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
        **data,
    ):
        
        """Forward inference function for closed-loop evaluation. 
        """
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

        return bbox_results

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