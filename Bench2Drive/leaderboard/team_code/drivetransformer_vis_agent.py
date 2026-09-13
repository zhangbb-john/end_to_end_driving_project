import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

import sys
sys.path.append("/home/slxy/zca/backup/Bench2Drive")

from DriveTransformer.team_code.pid_controller import DecouplePIDController
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from team_code.planner import RoutePlanner
from pyquaternion import Quaternion
from scipy.interpolate import splprep, splev
import copy
import seaborn as sns

os.environ['SAVE_PATH'] = "/home/slxy/zca/backup/Bench2Drive/save_path"
os.environ['ROUTES'] = "leaderboard/data/drivetransformer_bench2drive_dev10.xml"

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

IS_BENCH2DRIVE = 1
VIS_EGO_MOTION = True  # NOTE: project-3 project-4 setting as True

    
def float_to_uint8_color(float_clr):
    assert all([c >= 0. for c in float_clr])
    assert all([c <= 1. for c in float_clr])
    return [int(c * 255.) for c in float_clr]


COLORS = [float_to_uint8_color(clr) for clr in sns.color_palette("bright", n_colors=10)]
COLORMAP = OrderedDict({
    6: COLORS[8],  # yellow
    4: COLORS[8],
    3: COLORS[0],  # blue
    1: COLORS[6],  # pink
    0: COLORS[2],  # green
    8: COLORS[7],  # gray
    7: COLORS[1],  # orange
    5: COLORS[3],  # red
    2: COLORS[5],  # brown
})



def get_entry_point():
    return 'DriveTransformerAgent'


class DriveTransformerAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.controller = DecouplePIDController(speed_k_p=2.0, speed_k_i=0.8, speed_k_d=1.5, steer_k_p=1.5, steer_k_i=0.2, steer_k_d=0.2)
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.device = "cuda"
        cfg = Config.fromfile(self.config_path)
        
        self.cameras = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        
        if self.ckpt_path != "None":
            ckpt = torch.load(self.ckpt_path)
            ckpt = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in ckpt.items():
                new_key = key.replace("model.","").replace("._orig_mod", "")
                new_state_dict[new_key] = value
            print(self.model.load_state_dict(new_state_dict, strict = False))
        wrap_fp16_model(self.model)
        self.model.to(self.device)
        self.model.eval()

        self.test_pipeline = []
        self.past_ego_pos_cache = []
        self.cache_lenth = 20

        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFiles','LoadAnnotations3D', "CustomObjectRangeFilter", "CustomObjectNameFilter", "TrajPreprocess"]:
                self.test_pipeline.append(test_pipeline)
            if test_pipeline["type"] == "CustomFormatBundle3D":
                test_pipeline["collect_keys"] = ['lidar2img', 'cam_intrinsic','timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
            if test_pipeline["type"] == "CustomCollect3D":
                test_pipeline["keys"] = ['img', 'ego_his_trajs', 'ego_lcf_feat', 'ego_fut_cmd', 'prev_exists', 'index', 'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
        self.test_pipeline = Compose(self.test_pipeline)

        self.stop_time = 0
   
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.pid_metadata = {}
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0	
        self.prev_control_cache = []
        self.prev_control_list = []
        self.step_time_avg = []
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            print("SAVE Result to ", string)
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'rgb_front').mkdir()
            (self.save_path / 'rgb_front_right').mkdir()
            (self.save_path / 'rgb_front_left').mkdir()
            (self.save_path / 'rgb_back').mkdir()
            (self.save_path / 'rgb_back_right').mkdir()
            (self.save_path / 'rgb_back_left').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
   
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                            [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                            [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
  
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        
        
        self.cam_intrinsics = {
        'CAM_FRONT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ,   0.        ],
                             [  0.        , 560.16603057, 450.        ,   0.        ],
                             [  0.        ,   0.        ,   1.        ,   0.        ],
                             [  0.        ,   0.        ,   0.        ,   1.        ]]),
     
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
  
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        }
        
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        topdown_extrinsics =  np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 50.0], [0.0, 0.0, 0.0, 1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ topdown_extrinsics
        
        self.all_sensors =  {
                # camera rgb
                'CAM_FRONT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                'CAM_FRONT_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                'CAM_FRONT_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                'CAM_BACK':{
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                'CAM_BACK_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                'CAM_BACK_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                'IMU':{
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                'GPS':{
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                'SPEED':{
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                'bev': {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }
                
        }
   
   


    def _init(self):
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
            
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner = RoutePlanner(7.5, 25.0, 257, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True
  
    def sensors(self):
        sensors = []
        select_sensor_names = self.cameras + ['IMU','GPS','SPEED']
        if IS_BENCH2DRIVE:
            select_sensor_names.append('bev')
        for key in select_sensor_names:
            sensors.append(self.all_sensors[key])
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        imgs = {}
        for cam in self.cameras:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        far_node, far_command = self._command_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node,
                'command_far':far_command,
                'command_far_xy':far_node,    
                }
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = np.array(self.step / 20)
        results['box_type_3d'], _ = get_box_type('LiDAR')
        results['index'] = self.step
        results['prev_exists'] = (self.step > 1) 
        for cam in self.cameras: 
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(self.cam_intrinsics[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        results['aug_config'] = {'resize': 0.66, 'resize_dims': (1056, 594), 'crop': (0, 210, 1056, 594), 'flip': False, 'rotate': 0, 'rotate_3d': 0}

        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0] = tick_data['speed']
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = can_bus[15]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = tick_data['speed']
        
        ego_lcf_feat[8] = 0 if len(self.prev_control_cache) < 2 else self.prev_control_cache[0].steer
        results['ego_lcf_feat'] = ego_lcf_feat
        command = np.zeros(140)
        command[0:6] = self.command2hot(tick_data['command_far'])
        command[70:76] = self.command2hot(tick_data['command_near'])
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        command_far_xy = np.array([tick_data['command_far_xy'][0]-can_bus[0],-tick_data['command_far_xy'][1]-can_bus[1]])  
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_near_xy = rotation_matrix @ command_near_xy
        local_command_far_xy = rotation_matrix @ command_far_xy
        command[6:70] = self.pos2posemb(local_command_far_xy)
        command[76:140] = self.pos2posemb(local_command_near_xy)
        results['ego_fut_cmd'] = command
  
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        current_pose = lidar2global
        current_pose_inv = self.invert_pose(current_pose)
        results['ego_pose'] = current_pose
        results['ego_pose_inv'] = current_pose_inv   
        past_pose_1 = self.past_ego_pos_cache[-10] if len(self.past_ego_pos_cache) >= 10 else lidar2global
        past_pose_2 = self.past_ego_pos_cache[0] if len(self.past_ego_pos_cache) == 20 else lidar2global         
        past2current_1 = current_pose_inv @ past_pose_1
        past2current_2 = current_pose_inv @ past_pose_2
        past2current_1_xy = past2current_1[0:2,3]
        past2current_2_xy = past2current_2[0:2,3]
            
        ego_his_trajs = np.zeros((2,2))
        ego_his_trajs[0] = past2current_1_xy - past2current_2_xy
        ego_his_trajs[1] = -past2current_1_xy
        results['ego_his_trajs'] = ego_his_trajs

        if len(self.past_ego_pos_cache)==20:
            self.past_ego_pos_cache.pop(0)
        self.past_ego_pos_cache.append(current_pose)
        if self.step%2 == 1:
            return self.prev_control
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        results = self.test_pipeline(results)      
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if isinstance(data,torch.Tensor):
                    input_data_batch[key] = data.to(self.device)
                    if input_data_batch[key].dtype==torch.float64:
                        input_data_batch[key] = input_data_batch[key].to(torch.float32)
                elif isinstance(data,list):
                    if torch.is_tensor(data[0]):
                        input_data_batch[key][0] = input_data_batch[key][0].to(self.device)
                        if input_data_batch[key][0].dtype==torch.float64:
                            input_data_batch[key][0] = input_data_batch[key][0].to(torch.float32)

        step_start_time = time.time()
        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)
        self.step_time_avg.append(float(time.time()-step_start_time))
        if len(self.step_time_avg)==20:
            # print("Model Avg Step Time:", np.mean(self.step_time_avg))
            self.step_time_avg.pop(0)
        all_out_truck = None
        ego_traj_cls_scores = None
        selected_mode = 0 
        # pdb; pdb.set_trace()
        # angles = output_data_batch[0]['pts_bbox']['ego_fut_preds_fix_dist'][0,selected_mode,:,0].float().cpu().numpy()
        angles = output_data_batch[0]['ego_fut_preds_fix_dist'][0,selected_mode,:,0].float().cpu().numpy()
        ego_traj_fix_dist = np.arange(1,21,dtype=np.float64).reshape(-1,1).repeat(2,1)
        ego_traj_fix_dist[:,0] *= np.cos(angles)
        ego_traj_fix_dist[:,1] *= np.sin(angles)
        # ego_traj_fix_time = output_data_batch[0]['pts_bbox']['ego_fut_preds_fix_time'][0,selected_mode,:,[1,0]].float().cpu().numpy()
        ego_traj_fix_time = output_data_batch[0]['ego_fut_preds_fix_time'][0,selected_mode,:,[1,0]].float().cpu().numpy()
        if self.step <= 20: 
            steer, throttle, brake = 0.0, 0.0, 1.0
        else:
            steer, throttle, brake = self.controller.step(ego_traj_fix_time, ego_traj_fix_dist, tick_data['speed'])
        
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['speed'] = float(tick_data['speed'])
        #if SAVE_PATH is not None and self.step % 10 == 0:
        self.save(tick_data, ego_traj_fix_time[:,[1,0]].copy(), output_data_batch)
        
        self.prev_control = control
        if len(self.prev_control_cache)==2:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        
        return control
    
    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
    
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
    
    def save(self, tick_data,ego_traj,result=None):
        frame = self.step // 2
        imgs_with_box = {}
        new_ego_traj = ego_traj[4::5]
        for cam, img in tick_data['imgs'].items():
            #import pdb; pdb.set_trace()
            # imgs_with_box[cam] = self.draw_lidar_bbox3d_on_img(result[0]['pts_bbox']['boxes_3d'], tick_data['imgs'][cam], self.lidar2img[cam], scores=result[0]['pts_bbox']['scores_3d'],labels=result[0]['pts_bbox']['labels_3d'],canvas_size=(900,1600))
            imgs_with_box[cam] = self.draw_lidar_bbox3d_on_img(result[0]['boxes_3d'], tick_data['imgs'][cam], self.lidar2img[cam], scores=result[0]['scores_3d'],labels=result[0]['labels_3d'],canvas_size=(900,1600))
            
        # imgs_with_box['bev'] = self.draw_lidar_bbox3d_on_img(result[0]['pts_bbox']['boxes_3d'], tick_data['bev'], self.coor2topdown, scores=result[0]['pts_bbox']['scores_3d'],labels=result[0]['pts_bbox']['labels_3d'],canvas_size=(512,512))
        imgs_with_box['bev'] = self.draw_lidar_bbox3d_on_img(result[0]['boxes_3d'], tick_data['bev'], self.coor2topdown, scores=result[0]['scores_3d'],labels=result[0]['labels_3d'],canvas_size=(512,512))
        imgs_with_box['bev'] = self.draw_traj_bev(new_ego_traj, imgs_with_box['bev'],is_ego=True)
        for i in range(len(result[0]['map_scores_3d'])):
            score = result[0]['map_scores_3d'][i].item()
            if score > 0.8:
                print(f"draw map points on bev image.")
                map_pts = np.array(result[0]['map_pts_3d'][i])
                imgs_with_box['bev'] = self.draw_map_bev(map_pts, imgs_with_box['bev'],is_ego=False)
                # for cam, img in tick_data['imgs'].items():
                #     imgs_with_box[cam] = self.draw_lidar_map_on_img(map_pts, imgs_with_box[cam], self.lidar2img[cam], )
        imgs_with_box['CAM_FRONT'] = self.draw_traj(new_ego_traj, imgs_with_box['CAM_FRONT'])
        for cam, img in imgs_with_box.items():
            Image.fromarray(img).save(self.save_path / str.lower(cam).replace('cam','rgb') / ('%04d.png' % frame))   
        print(f"save result at:{self.save_path}")
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()
        
    def draw_traj(self, traj, raw_img,canvas_size=(900,1600),thickness=3,is_ego=True,hue_start=120,hue_end=80):
        line = traj
        lidar2img_rt = self.lidar2img['CAM_FRONT']
        img = raw_img.copy()
        pts_4d = np.stack([line[:,0],line[:,1],np.ones((line.shape[0]))*(-1.84),np.ones((line.shape[0]))])
        pts_2d = ((lidar2img_rt @ pts_4d).T)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask,0:2]
        if is_ego:
            pts_2d = np.concatenate([np.array([[800,900]]),pts_2d],axis=0)
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T
        
        num_points = len(smoothed_pts)
        for i in range(num_points-1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
            if VIS_EGO_MOTION:
                cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)  
      
        return img



    def draw_traj_bev(self, traj, raw_img, is_ego=True, canvas_size=(512,512),thickness=3,hue_start=120,hue_end=80):

        line = traj
        img = raw_img.copy()        
        pts_4d = np.stack([line[:,0],line[:,1],np.zeros((line.shape[0])),np.ones((line.shape[0]))])
        pts_2d = (self.coor2topdown @ pts_4d).T
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        mask = (pts_2d[:, 0]>0) & (pts_2d[:, 0]<canvas_size[1]) & (pts_2d[:, 1]>0) & (pts_2d[:, 1]<canvas_size[0])
        if not mask.any():
            return img
        pts_2d = pts_2d[mask,0:2]
        try:
            tck, u = splprep([pts_2d[:, 0], pts_2d[:, 1]], s=0)
        except:
            return img
        unew = np.linspace(0, 1, 100)
        smoothed_pts = np.stack(splev(unew, tck)).astype(int).T

        num_points = len(smoothed_pts)
        for i in range(num_points-1):
            hue = hue_start + (hue_end - hue_start) * (i / num_points)
            hsv_color = np.array([hue, 255, 255], dtype=np.uint8)
            rgb_color = cv2.cvtColor(hsv_color[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2RGB).reshape(-1)
            rgb_color_tuple = (float(rgb_color[0]),float(rgb_color[1]),float(rgb_color[2]))
            if VIS_EGO_MOTION:
                if smoothed_pts[i,0]>0 and smoothed_pts[i,0]<canvas_size[1] and smoothed_pts[i,1]>0 and smoothed_pts[i,1]<canvas_size[0]:
                    cv2.line(img,(smoothed_pts[i,0],smoothed_pts[i,1]),(smoothed_pts[i+1,0],smoothed_pts[i+1,1]),color=rgb_color_tuple, thickness=thickness)   
                elif i==0:
                    break
        return img
    
    def draw_map_bev(self, map_points, raw_img,canvas_size=(512,512),thickness=3,is_ego=False,hue_start=120,hue_end=80):
        

        img = raw_img.copy()
        P = self.coor2topdown[:3, :] if self.coor2topdown.shape[0] == 4 else self.coor2topdown
        
        # 投影
        N = len(map_points)
        points_3d = np.column_stack([map_points[:, :2], np.zeros(N)])
        points_4d = np.column_stack([points_3d, np.ones(N)]).T
        points_2d_h = (P @ points_4d).T
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
        
        points_2d = points_2d.astype(int)
        
        # 绘制连线
        if N <= 1:
            return img
        for i in range(N-1):
            x1, y1 = points_2d[i]
            x2, y2 = points_2d[i+1]
            
            if (0 <= x1 < canvas_size[1] and 0 <= y1 < canvas_size[0] and
                0 <= x2 < canvas_size[1] and 0 <= y2 < canvas_size[0]):
                
                cv2.line(img, (x1, y1), (x2, y2), (0, 200, 255), 2)
    
        # 绘制点
        for i, (x, y) in enumerate(points_2d):
            if 0 <= x < canvas_size[1] and 0 <= y < canvas_size[0]:
                # 绘制实心点
                cv2.circle(img, (x, y), 2, (0, 127, 127), -1)
    
        return img

        # img = raw_img.copy()
        
        # # 投影转换
        # N = len(map_points)
        # pts_4d = np.stack([map_points[:, 0], map_points[:, 1], 
        #                 np.zeros(N), np.ones(N)])
        # pts_2d = (self.coor2topdown @ pts_4d).T
        # pts_2d[:, 0] /= pts_2d[:, 2]
        # pts_2d[:, 1] /= pts_2d[:, 2]
        
        # # 绘制每个点（固定颜色，固定大小）
        # for point in pts_2d:
        #     x, y = int(point[0]), int(point[1])
            
        #     if 0 <= x < canvas_size[1] and 0 <= y < canvas_size[0]:
        #         cv2.circle(img, (x, y), 6, (0, 255, 0), -1)  # 绿色，大小6，实心
        
        # return img
    
    def draw_lidar_map_on_img(self,map_points,raw_img,lidar2img_rt,canvas_size=(900,1600),img_metas=None,color=(0, 255, 0),thickness=1):
        img = raw_img.copy()

        N = len(map_points)
        points_3d = np.column_stack([map_points[:, :2], np.zeros(N)])
        points_4d = np.column_stack([points_3d, np.ones(N)])

        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()      
            
        pts_2d = (lidar2img_rt @ points_4d.T).T
        points_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
        
        points_2d = points_2d.astype(int)

        # 绘制连线
        if N <= 1:
            return img
        for i in range(N-1):
            x1, y1 = points_2d[i]
            x2, y2 = points_2d[i+1]
            
            if (0 <= x1 < canvas_size[1] and 0 <= y1 < canvas_size[0] and
                0 <= x2 < canvas_size[1] and 0 <= y2 < canvas_size[0]):
                
                cv2.line(img, (x1, y1), (x2, y2), (127, 200, 127), 2)

        return img
    
    def draw_lidar_bbox3d_on_img(self,bboxes3d,raw_img,lidar2img_rt,canvas_size=(900,1600),img_metas=None,scores=None,labels=None,trajs=None,color=(0, 255, 0),thickness=1):
        img = raw_img.copy()
        bboxes3d_numpy = bboxes3d.tensor.cpu().numpy()
        if len(bboxes3d_numpy) == 0:
            return img
        corners_3d = bboxes3d.corners
        num_bbox = corners_3d.shape[0]
        pts_4d = np.concatenate(
            [corners_3d.reshape(-1, 3),
            np.ones((num_bbox * 8, 1))], axis=-1)
        lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
        if isinstance(lidar2img_rt, torch.Tensor):
            lidar2img_rt = lidar2img_rt.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()            
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()            
            
        ################################################
        # project-1    
        # TODO-9 vis 3d bbox
        # project pts_4d to image2d using lidar2img_rt
        # 替换此处代码
        pts_2d = pts_4d
        ################################################
        imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)
        depth = pts_2d[..., 2].reshape(num_bbox, 8)
        mask1 = ((imgfov_pts_2d[:,:,0]>-1e5) & (imgfov_pts_2d[:,:,0]<1e5)&(imgfov_pts_2d[:,:,1]>-1e5) & (imgfov_pts_2d[:,:,1]<1e5) & (depth > -1) ).all(-1)
        mask2 = (imgfov_pts_2d.reshape(num_bbox,16).max(axis=-1) - imgfov_pts_2d.reshape(num_bbox,16).min(axis=-1))< 2000
        mask = mask1 & mask2
        if scores is not None:
            mask3 = (scores>=0.3)
            mask = mask & mask3
            
        if not mask.any():
            return img

        scores = scores[mask] if scores is not None else None
        labels = labels[mask] if labels is not None else None
        
        imgfov_pts_2d = imgfov_pts_2d[mask]
        num_bbox = mask.sum()
        if trajs is not None:
            
            trajs = trajs[mask]
            agent_boxes = bboxes3d_numpy[mask]
            for traj,agent_box,label in zip(trajs,agent_boxes,labels):
                if label in [0,1,2,3,7]:
                    for i in range(6):
                        traj1 = np.concatenate([np.zeros((1,2)),traj[i].reshape(6,2)],axis=0)
                        traj1 = np.cumsum(traj1,axis=0) + agent_box[None,0:2]
                        #traj1 = (r_m @ traj1.T).T + agent_box[None,0:2]
                        if canvas_size==(900,1600):
                            img = self.draw_traj(traj1,img,hue_start=0,hue_end=20)
                        else:
                            img = self.draw_traj_bev(traj1,img,hue_start=0,hue_end=20)

        return self.plot_rect3d_on_img(img, num_bbox, imgfov_pts_2d, scores, labels, color, thickness,bev=(canvas_size!=(900,1600))) 
    
    

    
    def plot_rect3d_on_img(self,img,num_rects,rect_corners,scores=None,labels=None,color=(0, 255, 0),thickness=1,bev=False):
        line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                        (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
        if bev:
            line_indices = ((0, 3), (3, 7),(4, 7), (0, 4))
        for i in range(num_rects):
            c = COLORMAP[labels[i]]
            thinck = 2
            corners = rect_corners[i].astype(np.int)
            # if scores is not None:
            #     cv2.putText(img, "{:.2f}".format(scores[i]), corners[0], cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
            if scores[i] < 0.3:
                continue
                #     c=(255,255,255)
                #     thinck=1
            for start, end in line_indices:
                cv2.line(img, (corners[start, 0], corners[start, 1]),
                        (corners[end, 0], corners[end, 1]), c, thinck,
                        cv2.LINE_AA)
        return img.astype(np.uint8)
    
    # def save(self, tick_data, ego_fut_preds_fix_time, ego_fut_preds_fix_dist, draw_traj=False):
    #     frame = self.step //10
    #     Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
    #     Image.fromarray(tick_data['imgs']['CAM_FRONT_LEFT']).save(self.save_path / 'rgb_front_left' / ('%04d.png' % frame))
    #     Image.fromarray(tick_data['imgs']['CAM_FRONT_RIGHT']).save(self.save_path / 'rgb_front_right' / ('%04d.png' % frame))
    #     Image.fromarray(tick_data['imgs']['CAM_BACK']).save(self.save_path / 'rgb_back' / ('%04d.png' % frame))
    #     Image.fromarray(tick_data['imgs']['CAM_BACK_LEFT']).save(self.save_path / 'rgb_back_left' / ('%04d.png' % frame))
    #     Image.fromarray(tick_data['imgs']['CAM_BACK_RIGHT']).save(self.save_path / 'rgb_back_right' / ('%04d.png' % frame))
        
    #     if draw_traj:
    #         ego_fut_preds_fix_time = ego_fut_preds_fix_time[:,[1,0]]
    #         ego_fut_preds_fix_time = np.concatenate([ego_fut_preds_fix_time[:,], np.zeros((ego_fut_preds_fix_time.shape[0], 1)), np.ones((ego_fut_preds_fix_time.shape[0], 1))], axis=-1)
    #         ego_fut_preds_fix_time = np.dot(self.coor2topdown, ego_fut_preds_fix_time.T).T
    #         ego_fut_preds_fix_time[:, :2] /= ego_fut_preds_fix_time[:, 2:3]
    #         ego_fut_preds_fix_time = np.nan_to_num(ego_fut_preds_fix_time)
    #         for k in range(ego_fut_preds_fix_time.shape[0]):
    #             cv2.circle(tick_data['bev'], (int(ego_fut_preds_fix_time[k, 0]), int(ego_fut_preds_fix_time[k, 1])), 0, (0, 0, 255), 5)
            
    #         ego_fut_preds_fix_dist = ego_fut_preds_fix_dist[:,[1,0]]
    #         ego_fut_preds_fix_dist = np.concatenate([ego_fut_preds_fix_dist, np.zeros((ego_fut_preds_fix_dist.shape[0], 1)), np.ones((ego_fut_preds_fix_dist.shape[0], 1))], axis=-1)
    #         ego_fut_preds_fix_dist = np.dot(self.coor2topdown, ego_fut_preds_fix_dist.T).T
    #         ego_fut_preds_fix_dist[:, :2] /= ego_fut_preds_fix_dist[:, 2:3]
    #         ego_fut_preds_fix_dist = np.nan_to_num(ego_fut_preds_fix_dist)
    #         for k in range(ego_fut_preds_fix_dist.shape[0]):
    #             cv2.circle(tick_data['bev'], (int(ego_fut_preds_fix_dist[k, 0]), int(ego_fut_preds_fix_dist[k, 1])), 0, (255, 0, 0), 5)
    #     Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
    #     outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
    #     json.dump(self.pid_metadata, outfile, indent=4)
    #     outfile.close()

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])
    
    
    
    
    

