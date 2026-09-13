import cv2
import os
import numpy as np
import json
from tqdm import trange
import collections

def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    
    # 相机顺序
    cameras = ['rgb_front', 'rgb_back', 
               'rgb_front_left', 'rgb_front_right', 
               'rgb_back_left', 'rgb_back_right',
               'bev']
        # 相机显示名称映射
    camera_names = {
        'rgb_front': 'FRONT',
        'rgb_back': 'BACK',
        'rgb_front_left': 'FRONT-LEFT',
        'rgb_front_right': 'FRONT-RIGHT',
        'rgb_back_left': 'BACK-LEFT',
        'rgb_back_right': 'BACK-RIGHT',
        'bev': 'BEV'
    }
    # 收集图片
    image_dict = {}
    for cam in cameras:
        cam_path = os.path.join(images_folder, cam)
        if os.path.exists(cam_path):
            images = sorted([f for f in os.listdir(cam_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            image_dict[cam] = images
    
    if not image_dict:
        return
    
    # 获取尺寸
    first_cam = 'rgb_front'
    sample_img = cv2.imread(os.path.join(images_folder, first_cam, image_dict[first_cam][0]))
    h, w = sample_img.shape[:2]
    
    # 布局：2行3列 + BEV
    rows, cols = 2, 3
    bev_scale = 2
    bev_h = h * bev_scale
    
    # 最终尺寸
    final_w = w * cols + w
    final_h = max(h * rows, bev_h)
    
    # 调整BEV尺寸
    bev_img = cv2.imread(os.path.join(images_folder, 'bev', image_dict['bev'][0]))
    if bev_img is not None:
        bev_img = cv2.resize(bev_img, (w, bev_h))
        if bev_img.shape[0] < final_h:
            bev_img = cv2.resize(bev_img, (w, final_h))
    
    # 创建视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (final_w, final_h))
    
    num_frames = min(len(images) for images in image_dict.values())
    
    for i in range(num_frames):
        # 创建画布
        frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        
        # 绘制6个视角
        for idx, cam in enumerate(['rgb_front_left', 'rgb_front', 
                                  'rgb_front_right', 'rgb_back_left', 
                                  'rgb_back', 'rgb_back_right']):
            if cam in image_dict and i < len(image_dict[cam]):
                img = cv2.imread(os.path.join(images_folder, cam, image_dict[cam][i]))
                if img is not None:
                    img = cv2.resize(img, (w, h))
                    
                    # 计算位置
                    row = idx // cols
                    col = idx % cols
                    x = col * w
                    y = row * h
                    
                    frame[y:y+h, x:x+w] = img

                    # 添加相机名称标签
                    cam_name = camera_names.get(cam, cam)
                    
                    # 方法1: 简单文字标签
                    cv2.putText(frame, cam_name, 
                               (x + 10, y + 125),  # 左上角偏移
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
                    
        # 绘制BEV
        if 'bev' in image_dict and i < len(image_dict['bev']):
            bev_img = cv2.imread(os.path.join(images_folder, 'bev', image_dict['bev'][i]))
            if bev_img is not None:
                bev_img = cv2.resize(bev_img, (w, final_h))
                frame[0:final_h, w*cols:w*cols+w] = bev_img
        
                        # 添加BEV标签
                cv2.putText(frame, 'BEV', 
                           (w*cols + 10, 125),  # 左上角
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 4)
        # 添加分隔线
        cv2.line(frame, (w*cols, 0), (w*cols, final_h), (100, 100, 100), 2)
        cv2.line(frame, (0, h), (w*cols, h), (100, 100, 100), 1)
        
        video.write(frame)
    
    video.release()

root = "/home/slxy/zca/code/Bench2Drive/save_path/"

case_name = "drivetransformer_bench2drive_dev10_RouteScenario_2091_rep0_Town12_NonSignalizedJunctionLeftTurn_1_5_12_21_20_13_02"

images_folder = os.path.join(root, case_name)
output_video = os.path.join(images_folder, 'demo.mp4')
font_scale = 1
text_color = (255, 255, 255)
text_position = (50, 50)
fps = 20

create_video(images_folder, output_video, fps, font_scale, text_color, text_position)
