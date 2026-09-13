# DriveTransformer 开环测试
开环测试的定义为：模型仅依据输入的历史数据或离线数据完成推理并输出控制指令，不将车辆的实时位置信息反馈至仿真器或实车，以此独立验证模型决策逻辑的合理性。
目前开环测试主要有两种实现方案：
第一种为纯开环测试方案，在测试全程直接采用每一帧的真值航点（GT waypoint），通过自车（ego_vehicle）的坐标变换关系，将航点精准映射到 CARLA 仿真环境的世界坐标系中，实现自车位置的逐帧真值锚定。
第二种为序列对齐式开环方案，测试初期仅对首帧数据进行真值对齐，后续帧则通过读取完整的真值航点序列，额外引入控制器将前一帧的航点信息传递至当前帧，以此作为模型的输入完成连续推理。

# 数据序列
leaderboard/data/drivetransformer_bench2drive_dev10_open_loop.xml

# 代码运行
bash start_eval_open_loop.sh

# 实现细节
直接通过scenearioc初始化貌似并不能对应GT的第一个位置，使用初始化后的第一个点，重新对齐。
核心接口：ego_vehicle.set_transform(carla.Transform(target_location, target_rotation))
详细操作：  
```
 # On first call, align vehicle to the first GT waypoint
    if not self.gt_trajectory_aligned and ego_vehicle is not None:
        first_waypoint_transform = plan_to_use[0][0]
        target_location = carla.Location(
            x=first_waypoint_transform.location.x,
            y=first_waypoint_transform.location.y,
            z=first_waypoint_transform.location.z
        )
        target_rotation = first_waypoint_transform.rotation
        
        # Teleport vehicle to first waypoint
        ego_vehicle.set_transform(carla.Transform(target_location, target_rotation))
        
        # Update current position
        current_pos = np.array([target_location.x, target_location.y])
        
        print(f"[GT Control] *** ALIGNED vehicle to first GT waypoint ***")
        print(f"[GT Control] Target position: x={target_location.x:.2f}, y={target_location.y:.2f}, z={target_location.z:.2f}")
        print(f"[GT Control] Target rotation: pitch={target_rotation.pitch:.2f}, yaw={target_rotation.yaw:.2f}, roll={target_rotation.roll:.2f}")
        
        self.gt_trajectory_aligned = True
```
