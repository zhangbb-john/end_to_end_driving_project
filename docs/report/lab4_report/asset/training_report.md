# DriveTransformer Large 训练报告

## 1. 模型概述

DriveTransformer 是一种端到端自动驾驶模型，基于纯视觉输入（多视角相机图像），同时完成 3D 目标检测、轨迹预测、地图构建和自车规划等多任务。本次训练使用的是 DriveTransformer Large 配置。

### 1.1 模型架构

| 组件 | 配置 |
|------|------|
| 模型类型 | DriveTransformer |
| 图像骨干网络 | ResNet-50（预训练权重：resnet50-19c8e357.pth，冻结第一阶段） |
| 图像颈部网络 | FPN（输入通道 2048，输出通道 768，单层输出） |
| 嵌入维度 | 768 |
| 前馈网络维度 | 3072 |
| Decoder 层数 | 12 层（DriveTransformerDecoder） |
| Pre-Decoder | Agent Pre-Decoder (1层) + Map Pre-Decoder (1层) |
| 注意力头维度 | 64 |
| Dropout | 0.1 |
| 激活函数 | SwiGLU |

### 1.2 查询设置

| 参数 | 值 |
|------|------|
| Agent 查询数量 | 900 |
| Agent 筛选后查询数量 | 900 |
| Map 查询数量 | 100 |
| Map 筛选后查询数量 | 100 |
| Agent 传播数量（时序） | 50 |
| Map 传播数量（时序） | 50 |
| 时序记忆帧数 | 10 |
| 未来轨迹模态数 | 6 |
| 自车未来模态数 | 1 |
| 未来时间步（轨迹） | 6 |
| 未来时间步（自车固定时间） | 30 |
| 未来时间步（自车固定距离） | 20 |

### 1.3 检测范围

| 参数 | 值 |
|------|------|
| 点云范围 (X) | [-15.0, 15.0] m |
| 点云范围 (Y) | [-30.0, 30.0] m |
| 点云范围 (Z) | [-2.0, 2.0] m |
| 体素大小 | [0.15, 0.15, 4] m |
| 深度起始 | 1 m |
| 深度步长 | 0.8 m |
| 深度层数 | 64 |

### 1.4 检测类别（9类）

car, van, truck, bicycle, traffic_sign, traffic_cone, traffic_light, pedestrian, others

---

## 2. 训练参数设置

### 2.1 优化器

| 参数 | 值 |
|------|------|
| 优化器类型 | AdamW |
| 学习率 | 2e-5 |
| 权重衰减 | 0.01 |
| 梯度裁剪 | max_norm=35, norm_type=2 |

### 2.2 学习率调度

| 参数 | 值 |
|------|------|
| 调度策略 | CosineAnnealing |
| 预热方式 | Linear |
| 预热迭代数 | 1000 |
| 预热比率 | 0.1 |
| 最小学习率比率 | 0.01 |

### 2.3 训练配置

| 参数 | 值 |
|------|------|
| 训练方式 | IterBasedRunner |
| 总迭代次数 | 225,000 |
| 每 GPU 样本数 | 4 |
| 每 GPU 工作线程 | 12 |
| GPU 数量 | 4 |
| 总批量大小 | 16（4 × 4 GPUs） |
| 混合精度训练（FP16） | 是（loss_scale=512.0） |
| 分布式后端 | NCCL |
| Checkpoint 保存间隔 | 3000 迭代 |
| 日志记录间隔 | 50 迭代 |
| Gradient Checkpointing | 是（with_cp=True） |

### 2.4 数据增强

| 参数 | 值 |
|------|------|
| 输入图像原始尺寸 | 900 × 1600 |
| Resize 范围 | (0.64, 0.69) |
| 最终输入尺寸 | 384 × 1056 |
| 随机翻转 | 是 |
| 旋转范围 | (-5.4°, 5.4°) |
| 图像归一化 | mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375] |

### 2.5 损失函数配置

| 损失项 | 类型 | 权重 |
|--------|------|------|
| 检测分类 (loss_cls) | FocalLoss (gamma=2.0, alpha=0.25) | 2.0 |
| 检测回归 (loss_bbox) | L1Loss | 0.25 |
| 轨迹回归 (loss_traj) | L1Loss | 0.2 |
| 轨迹分类 (loss_traj_cls) | FocalLoss (gamma=2.0, alpha=0.5) | 0.2 |
| 地图分类 (loss_map_cls) | FocalLoss (gamma=2.0, alpha=0.25) | 2.0 |
| 地图点回归 (loss_map_pts) | PtsL1Loss | 1.0 |
| 地图方向 (loss_map_dir) | PtsDirCosLoss | 0.005 |
| 规划回归-固定时间 (loss_plan_reg_fix_time) | L1Loss | 3.5 |
| 规划回归-固定距离 (loss_plan_reg_fix_dist) | L1Loss | 10.0 |
| 规划分类 (loss_plan_cls) | FocalLoss (gamma=4.0, alpha=0.5) | 20.0 |

### 2.6 匹配策略

- Agent 匹配：HungarianAssigner3D（cls_cost=2.0, reg_cost=0.25, iou_cost=0.0）
- Map 匹配：MapHungarianAssigner3D

---

## 3. 训练硬件

| 项目 | 配置 |
|------|------|
| GPU 型号 | NVIDIA RTX 6000 |
| GPU 数量 | 4 |
| 每 GPU 显存占用 | ~26.6 GB |
| 分布式训练 | 是（4卡数据并行，NCCL 后端） |
| 训练框架 | MMDetection3D + MMCV |
| 混合精度 | FP16 |

---

## 4. 训练数据

| 项目 | 配置 |
|------|------|
| 数据集 | Bench2Drive (B2D) |
| 数据集类型 | B2D_DriveTransformer_Dataset |
| 输入模态 | 纯相机（多视角） |
| 训练集标注文件 | b2d_infos_v1_train_drivetransformer_meta.pkl |
| 验证集标注文件 | b2d_infos_v1_val_drivetransformer_meta.pkl |
| 地图信息文件 | b2d_map_infos.pkl |
| 时序缓存长度 | 5 帧 |
| 未来帧数（Agent） | 6 |
| 采样间隔 | 5 |
| 子序列长度 | -1（全序列） |
| 使用分片数据 | 是 |
| Map 每条线点数 | 20 |

---

## 5. 训练结果

### 5.1 训练时间

| 项目 | 值 |
|------|------|
| 训练开始时间 | 2026-05-30 23:15 |
| 训练结束时间 | 2026-06-04 07:32 |
| 总训练时间 | 约 4 天 8 小时（~104.3 小时） |
| 平均每迭代耗时 | ~1.4 秒 |
| 总迭代次数 | 225,000 |
| 总训练样本数 | 3,600,000（225,000 × batch_size 16） |

### 5.2 损失收敛情况

#### 初始损失（iter 50）

| 损失项 | 值 |
|--------|------|
| Total Loss | 2142.10 |
| loss_cls | 81.36 |
| loss_bbox | 1.91 |
| loss_traj | 7.55 |
| loss_traj_cls | 0.037 |
| loss_map_cls | 1.85 |
| loss_map_pts | 5.94 |

#### 最终损失（iter 225,000）

| 损失项 | 值 |
|--------|------|
| Total Loss | 36.29 |
| loss_cls | 0.061 |
| loss_bbox | 0.218 |
| loss_traj | 0.276 |
| loss_traj_cls | 0.000 |
| loss_map_cls | 0.095 |
| loss_map_pts | 0.640 |
| loss_map_dir | 0.005 |
| d12.loss_plan_l1_fix_time | 0.294 |
| d12.loss_plan_l1_fix_dist | 0.102 |

#### 最终损失（最后 50 次迭代平均值，平滑）

| 损失项 | 平均值 |
|--------|--------|
| Total Loss | 47.09 |
| loss_cls | 0.102 |
| loss_bbox | 0.301 |
| loss_traj | 0.530 |
| loss_traj_cls | 0.000 |
| loss_map_cls | 0.098 |
| loss_map_pts | 0.594 |
| loss_map_dir | 0.005 |
| d12.loss_plan_l1_fix_time | 0.952 |
| d12.loss_plan_l1_fix_dist | 0.154 |

### 5.3 收敛分析

1. **总损失**：从 2142.10 下降至约 36~47，下降了约 98.2%，整体收敛良好。

2. **检测损失（loss_cls + loss_bbox）**：
   - 分类损失从 81.36 下降至 0.06~0.10，收敛极为显著（>99.9%下降）
   - 回归损失从 1.91 下降至 0.22~0.30，收敛稳定

3. **轨迹预测损失（loss_traj + loss_traj_cls）**：
   - 轨迹回归从 7.55 下降至 0.28~0.53
   - 轨迹分类收敛至接近 0，表明模态选择已高度确定

4. **地图构建损失（loss_map_cls + loss_map_pts + loss_map_dir）**：
   - 地图分类从 1.85 下降至 0.10
   - 地图点回归从 5.94 下降至 0.59~0.64
   - 地图方向损失收敛至 0.005

5. **规划损失（Planning）**：
   - 固定时间规划 L1 损失：0.29~0.95
   - 固定距离规划 L1 损失：0.10~0.15
   - 规划损失是所有任务中绝对值最大的贡献者（权重高），反映了端到端规划的难度

6. **学习率**：采用 Cosine Annealing 策略，从 2e-5 逐步衰减至接近 0，训练末期学习率已降至最低。

### 5.4 模型权重保存

训练过程中每 3000 迭代保存一次 checkpoint，共保存了从 iter_3000 到 iter_225000 的所有权重文件。最终模型为 `latest.pth`（对应 iter_225000）。

### 5.5 训练损失曲线

训练损失曲线已保存为 `training_loss_curves.png`，包含以下 6 个子图：
- Total Loss（总损失）
- Detection Losses（检测分类 + 回归损失）
- Trajectory Losses（轨迹预测 + 分类损失）
- Map Losses（地图分类 + 点回归 + 方向损失）
- Planning Losses（规划固定时间 + 固定距离损失，最终 Decoder 层 d12）
- Learning Rate（学习率变化曲线）

---

## 6. 总结

本次使用 4 张 NVIDIA RTX 6000 GPU，在 Bench2Drive 数据集上完成了 DriveTransformer Large 模型的训练。模型采用 ResNet-50 骨干 + 768 维嵌入 + 12 层 Decoder 的 Large 配置，以 AdamW 优化器（lr=2e-5）和 Cosine Annealing 学习率策略训练 225,000 次迭代，总耗时约 4 天 8 小时。

训练结果显示所有任务（检测、轨迹预测、地图构建、规划）的损失均实现了显著收敛，模型整体训练过程稳定，未出现损失发散或异常波动。
