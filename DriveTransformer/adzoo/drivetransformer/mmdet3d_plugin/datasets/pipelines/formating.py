
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmcv.core.bbox.structures.base_box3d import BaseInstance3DBoxes
from mmcv.core.points import BasePoints
from mmcv.datasets.builder import PIPELINES
from mmcv.datasets.pipelines import to_tensor
from mmcv.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """
    def __init__(self, class_names, with_gt=True, with_label=True, with_ego=True, collect_keys=None):
        super(CustomFormatBundle3D, self).__init__(class_names, with_gt, with_label)
        self.with_ego = with_ego
        self.collect_keys = collect_keys

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # results['gt_map_masks'] = DC(to_tensor(results['gt_map_masks']), stack=True)
        if self.with_ego:
            if 'ego_his_trajs' in results:
                results['ego_his_trajs'] = DC(to_tensor(results['ego_his_trajs'][None, ...]), stack=True)
            if 'ego_fut_trajs' in results:
                results['ego_fut_trajs'] = DC(to_tensor(results['ego_fut_trajs'][None, ...]), stack=True)
            if 'ego_fut_masks' in results:
                results['ego_fut_masks'] = DC(to_tensor(results['ego_fut_masks'][None, None, ...]), stack=True)
            if 'ego_fut_cmd' in results:
                results['ego_fut_cmd'] = DC(to_tensor(results['ego_fut_cmd'][None, None, ...]), stack=True)
            if 'ego_lcf_feat' in results:
                results['ego_lcf_feat'] = DC(to_tensor(results['ego_lcf_feat'][None, None, ...]), stack=True)
            if 'gt_attr_labels' in results:
                results['gt_attr_labels'] = DC(to_tensor(results['gt_attr_labels']), cpu_only=False)
        if self.collect_keys is not None:    
            for key in self.collect_keys:
                if key in ['img_timestamp']:
                    results[key] = DC(to_tensor(np.array(results[key], dtype=np.float64)))
                elif key in ['timestamp']:  
                    results[key] = DC(np.array(results[key], dtype=np.float64))  
                else:
                    results[key] = DC(to_tensor(np.array(results[key], dtype=np.float32)))
                
                     
        # Format 3D data
        results = super(CustomFormatBundle3D, self).__call__(results)        
        
        return results