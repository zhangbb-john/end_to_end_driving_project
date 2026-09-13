## 1. Project3 报告（Chapter 4）

### 1.1. 目标
完成 Project-3 规划相关实现，覆盖：
- ego 输入特征预处理与编码
- agent/map/ego 注意力可见性规则
- 规划轨迹 refine 与 non-refine 输出
- 评测脚本 Python 解释器配置

### 1.2. 实现概述

#### 1.2.1. 2.1 ego 输入预处理与编码
对 `ego_lcf_feat`、`ego_his_trajs`、`ego_fut_cmd` 进行 squeeze/flatten 后拼接，送入 `self.ego_lcf_encoder()` 得到 `ego_query`。

#### 1.2.2. 2.2 注意力 mask（Project-3）
在 task self-attention 中，显式开放 agent/map/ego 相互可见的 mask 位置。

#### 1.2.3. 2.3 规划轨迹 refine 与 non-refine
refine 模式下对固定距离与固定时间分支回归结果做残差叠加；non-refine 直接赋值。

### 1.3. 修改文件
- start_eval.sh
- drivetransformer_head.py
- drivetransformer_layers.py

### 1.4. 3.1 具体改动（基于 commit 7bbb3473）
文件：start_eval.sh
原来是：
```bash
PYTHON="/home/slxy/.miniconda3/envs/drivetransformer/bin/python"
```
现在是：
```bash
PYTHON="python"
```

文件：drivetransformer_head.py
原来是：
```python
ego_query = torch.randn(1, self.embed_dims).to("cuda")
```
现在是：
```python
ego_lcf_feat_flat = ego_lcf_feat.squeeze(1)
ego_his_trajs_flat = ego_his_trajs.flatten(1)
ego_fut_cmd_flat = ego_fut_cmd.squeeze(1)
ego_traj = torch.cat([ego_lcf_feat_flat, ego_his_trajs_flat, ego_fut_cmd_flat], dim=-1)
ego_query = self.ego_lcf_encoder(ego_traj)
```

文件：drivetransformer_layers.py
原来是：
```python
pass
```
现在是：
```python
mask[agent_range, ego_range] = False
mask[map_range, ego_range] = False
mask[ego_range, agent_range] = False
mask[ego_range, map_range] = False
mask[ego_range, ego_range] = False
```

文件：drivetransformer_layers.py
原来是：
```python
pass
```
现在是：
```python
input_feat = ego_query + ego_pos_embed
ego_traj_ref_fix_dist_refine = ego_traj_branches_fix_dist[lid](input_feat).unsqueeze(-1)
ego_traj_ref_fix_dist = ego_traj_ref_fix_dist_refine + ego_traj_ref_fix_dist
ego_traj_ref_fix_time_refine = ego_traj_branches_fix_time[lid](input_feat).reshape(
	ego_query.shape[0], ego_query.shape[1], -1, 2
)
ego_traj_ref_fix_time = ego_traj_ref_fix_time_refine + ego_traj_ref_fix_time
```

文件：drivetransformer_layers.py
原来是：
```python
pass
```
现在是：
```python
ego_traj_ref_fix_dist = ego_traj_ref_fix_dist_refine
ego_traj_ref_fix_time = ego_traj_ref_fix_time_refine
```
### 1.5. 实验结果

评测结果 json 与闭环可视化视频均在 `asset/` 下。四条 route 各有两个版本：`edit_planner` 为补全规划模块后的结果，`original_noise` 为规划 query 仍是随机向量时的对照。

| Route | 补全后 | 随机向量对照 |
| --- | --- | --- |
| 2091 | [output_2091_edit_planner.mp4](asset/output_2091_edit_planner.mp4) | [output_2091_original_noise.mp4](asset/output_2091_original_noise.mp4) |
| 27494 | [output_27494_edit_planner.mp4](asset/output_27494_edit_planner.mp4) | [output_27494_original_noise.mp4](asset/output_27494_original_noise.mp4) |
| 17569 | [output_17569_edit_planner.mp4](asset/output_17569_edit_planner.mp4) | [output_17569_original_noise.mp4](asset/output_17569_original_noise.mp4) |
| 28198 | [output_28198_edit_planner.mp4](asset/output_28198_edit_planner.mp4) | [output_28198_original_noise.mp4](asset/output_28198_original_noise.mp4) |

#### 1.5.1. [asset/eval_bench2drive_dev4_edit_planner.json](asset/eval_bench2drive_dev4_edit_planner.json)
解读结果：整体状态 `Completed`，平均驾驶得分 76.5，平均路线完成度 100%，平均违规惩罚 0.765。

- RouteScenario_2091_rep0 — 状态: Completed；主要违规: 2 次车辆碰撞（collisions_vehicle）、多次速度相关违规（min_speed_infractions）；分数: composed=36.0, route=100, penalty=0.36。
- RouteScenario_27494_rep0 — 状态: Completed；主要违规: 以速度相关违规为主，无车辆碰撞；分数: composed=100.0, route=100, penalty=1.0。
- RouteScenario_17569_rep0 — 状态: Completed；主要违规: 以速度相关违规为主，无碰撞；分数: composed=100.0, route=100, penalty=1.0。
- RouteScenario_28198_rep0 — 状态: Completed；主要违规: 触发红灯违规（ran a red light），并有多条速度相关违规；分数: composed=70.0, route=100, penalty=0.7。


#### 1.5.2. [asset/eval_bench2drive_dev4_20260510_132849_noise_original.json](asset/eval_bench2drive_dev4_20260510_132849_noise_original.json)
解读结果：整体状态 `Failed`（若干 route 出现偏离或运行异常），平均驾驶得分 ≈19.70，平均路线完成度 51.36%，平均违规惩罚 ≈0.389。

- RouteScenario_2091_rep0 — 状态: Failed - Agent deviated from the route；主要违规: 碰撞静态挡栏/停止标志（collisions_layout）、越道（outside_route_lanes）、路线偏离（route_dev）、多次低速违规；分数: composed≈19.36, route≈50.25, penalty≈0.385。
- RouteScenario_27494_rep0 — 状态: Failed - Agent deviated from the route；主要违规: 与车辆碰撞（collisions_vehicle）、越道与路线偏离；分数: composed≈16.87, route≈54.45, penalty≈0.310。
- RouteScenario_17569_rep0 — 状态: Failed - TickRuntime（运行时中断）；主要违规: 碰撞护栏/车辆、越道；分数: composed≈11.04, route≈52.24, penalty≈0.211。
- RouteScenario_28198_rep0 — 状态: Failed - TickRuntime；主要违规: 碰撞交通信号灯（collisions_layout）、多条速度异常记录；分数: composed≈31.53, route≈48.5, penalty≈0.65。

### 1.6. 总结与分析
本次作业完成了 Project-3 的规划模块实现，并基于提交记录对关键文件进行了调整，实现要点如下：

- 将评测脚本 `start_eval.sh` 中的 Python 解释器切换为通用 `python`，便于跨环境运行。
- 在 `drivetransformer_head.py` 中，用实际的 ego 输入特征替代随机向量：对 `ego_lcf_feat` 做 `squeeze`、对 `ego_his_trajs` 做 `flatten`、对 `ego_fut_cmd` 做 `squeeze`，拼接后通过 `ego_lcf_encoder` 得到 `ego_query`，使规划输入基于真实观测。
- 在 `drivetransformer_layers.py` 中开放了 agent/map/ego 之间的 attention 可见性（将对应 mask 位置设为 False），并实现了规划分支的 refine（残差叠加）与 non-refine（直接回归）两种策略，保证轨迹维度匹配并支持逐层细化。

评测摘要：在新的planner版本中四条路全部完成且平均分较高，但仍存在个别碰撞和红灯违规；在带噪声planner的评测中，多条路出现路线偏离或运行中断，导致整体得分与完成度显著下降。此差异提示系统对输入质量与鲁棒性敏感。

 
