from math import pi, tan
import torch
import mmcv
from mmcv.datasets.builder import PIPELINES

@PIPELINES.register_module()
class TrajPreprocess:
    ''' 
        1. Transform the trajectory from ego coordinate system to agent's self coordinate system. 
        2. Cumsum the differential trajectory to get the absolute traj.
        3. Assign class label to the trajectory.
        
    '''
    def __init__(self, pc_range, num_traj_classes=6, fut_ts=6, with_ego_fix_dist=False, ego_fut_offset_input=True, assign_class_for_ego=True):
        self.pc_range = pc_range
        self.num_traj_classes = num_traj_classes
        self.fut_ts = fut_ts # future frames for agents
        self.with_ego_fix_dist = with_ego_fix_dist # whether ego trajectories with fixed distance in input 
        self.ego_fut_offset_input = ego_fut_offset_input # whether the future trajectory is differential
        self.assign_class_for_ego = assign_class_for_ego # whether assgin class for ego (false when single modal)

    def __call__(self, results):
        gt_agent_bboxes = results["gt_bboxes_3d"]
        gt_fut_trajs = torch.tensor(results["gt_attr_labels"][..., :self.fut_ts*2].reshape(-1, self.fut_ts, 2))
        gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2) # Add the differential trajectoris of agents.
        # normal yaw, 0 -> +x axis, pi/2 -> +y axis
        yaws = -gt_agent_bboxes.tensor[:, 6] - pi/2
        gt_fut_trajs = self.ego_2_agent_self(gt_fut_trajs, yaws) # ego coordinate system -> agent's self coordinate system. 
        gt_traj_fut_classes = self.assign_class(gt_fut_trajs) # assign label for agents
        agent_num = gt_traj_fut_classes.size(-1)
        gt_traj_fut_classes = torch.nn.functional.pad(gt_traj_fut_classes, (0, 100 - agent_num),"constant", 0)
        results["gt_attr_labels"][..., :self.fut_ts*2] = gt_fut_trajs.reshape(-1, self.fut_ts*2)
        results["gt_traj_fut_classes"] = gt_traj_fut_classes
        
        if self.with_ego_fix_dist:
            ego_fut_trajs_fix_dist = torch.tensor(results["ego_fut_trajs_fix_dist"]) 
            ego_fut_trajs_fix_time = torch.tensor(results["ego_fut_trajs_fix_time"])
            if self.ego_fut_offset_input:
                ego_fut_trajs_fix_dist = ego_fut_trajs_fix_dist.cumsum(dim=-2) # Add the differential trajectoris
                ego_fut_trajs_fix_time = ego_fut_trajs_fix_time.cumsum(dim=-2)   
                results["ego_fut_trajs_fix_dist"] = ego_fut_trajs_fix_dist
                results["ego_fut_trajs_fix_time"] = ego_fut_trajs_fix_time
            # assign the label with both type of trajectories
            results["ego_fut_classes"] = self.assign_class_ego(ego_fut_trajs_fix_dist, ego_fut_trajs_fix_time) if self.assign_class_for_ego else torch.tensor([0])
        else:
            ego_fut_trajs_fix_time = torch.tensor(results["ego_fut_trajs_fix_time"])
            if self.ego_fut_offset_input:
                ego_fut_trajs_fix_time = ego_fut_trajs_fix_time.cumsum(dim=-2)    
                results["ego_fut_trajs_fix_time"] = ego_fut_trajs_fix_time
            results["ego_fut_classes"] = self.assign_class(ego_fut_trajs_fix_time.unsqueeze(0)) if self.assign_class_for_ego else torch.tensor([0])

        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pc_range={self.pc_range}, num_traj_classes={self.num_traj_classes}, fut_ts={self.fut_ts})'
        return repr_str
    
    def ego_2_agent_self(self,coords, yaws):
        yaws_to_rotate = pi/2-yaws
        rot_matrix = torch.stack([torch.cos(yaws_to_rotate), -torch.sin(yaws_to_rotate), torch.sin(yaws_to_rotate), torch.cos(yaws_to_rotate)],dim=-1).reshape(-1,2,2)
        return (rot_matrix.unsqueeze(1) @ coords.unsqueeze(-1)).squeeze(-1) 

    def assign_class(self, trajs):
        trajs_end_point = trajs[:, -1, :]
        distances = torch.norm(trajs_end_point,dim=-1)
        yaw_in_degrees = torch.arctan2(trajs_end_point[:,1],trajs_end_point[:,0]) /torch.pi*180

        # 0:straight_right,1:straight_left,2:right,3:left,4:straight_with_high_speed,5:straight_with_low_speed
        #40,65,90,115,140  
        traj_class = []
        for i in range(trajs.shape[0]):
            distance = distances[i]
            yaw_in_degree = yaw_in_degrees[i]
            if distance <=2:
                traj_class.append(5)
            elif (yaw_in_degree>-90 and yaw_in_degree<52.5):
                traj_class.append(2)
            elif (yaw_in_degree>=52.5 and yaw_in_degree<77.5):
                traj_class.append(0)
            elif (yaw_in_degree>102.5 and yaw_in_degree<=127.5):
                traj_class.append(1)
            elif (yaw_in_degree>127.5 or yaw_in_degree<=-90):
                traj_class.append(3)
            elif (yaw_in_degree>=77.5 and yaw_in_degree<=102.5 and distance<15):
                traj_class.append(5)
            elif (yaw_in_degree>=77.5 and yaw_in_degree<=102.5 and distance>=15):
                traj_class.append(4)                

        traj_class = torch.tensor(traj_class)
                     
        return traj_class
    
    def assign_class_ego(self, trajs_fix_dist, trajs_fix_time):

        yaw_in_degree =trajs_fix_dist / torch.pi*180
        distance = torch.norm(trajs_fix_time[-1,:])
        # 0:straight_right,1:straight_left,2:right,3:left,4:straight_with_high_speed,5:straight_with_low_speed
        #40,65,90,115,140  

        if distance <=2:
            traj_class = 5
        elif (yaw_in_degree>-90 and yaw_in_degree<52.5):
            traj_class = 2
        elif (yaw_in_degree>=52.5 and yaw_in_degree<77.5):
            traj_class = 0
        elif (yaw_in_degree>102.5 and yaw_in_degree<=127.5):
            traj_class = 1
        elif (yaw_in_degree>127.5 or yaw_in_degree<=-90):
            traj_class = 3
        elif (yaw_in_degree>=77.5 and yaw_in_degree<=102.5 and distance<15):
            traj_class = 5
        elif (yaw_in_degree>=77.5 and yaw_in_degree<=102.5 and distance>=15):
            traj_class = 4

        traj_class = torch.tensor([traj_class])
                     
        return traj_class
