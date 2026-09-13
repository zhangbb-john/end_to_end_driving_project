## Project2 报告（Chapter 3）

### 1. 目标
基于 DriveTransformer 完成在线静态地图感知，覆盖：
- 在线建图 query 构建
- agent/map/ego 注意力 mask 规则
- map 点坐标的粗到精 refine

### 2. 实现概述

#### 2.1 Online Mapping Query 构建
用嵌入层权重替换随机初始化，并扩展 batch 维度。

步骤：
1) 读取 `self.map_query.weight` 与 `self.map_reference_points.weight`，形状为 (N, D)
2) 通过 `unsqueeze(0).expand(bs, -1, -1)` 扩展到 (bs, N, D)
3) 通过 `.to(device=img_feats.device, dtype=dtype)` 对齐到图像特征所在设备与精度

#### 2.2 注意力 Mask 规则（Project-2）
在 project-1 的基础上增加 agent 与 map 的双向可见：
- agent 关注 agent 和 map
- map 关注 agent 和 map
- ego 关注全部

通过对对应切片设置 `mask[...] = False` 完成。

#### 2.3 地图点坐标 refine
实现 map 点坐标的粗到精更新：
- refine：当前层回归输出作为残差，加到上一层坐标上（coarse-to-fine）
- non-refine：直接回归坐标

核心逻辑（与 §3.1 layers.py 改动一致）：
```python
if self.refine:
    map_pts_coord = map_reg_branches[lid](
        map_query + map_pts_pos_embed
    ).view(*map_pts_coord.shape) + map_pts_coord
else:
    map_pts_coord = map_reg_branches[lid](
        map_query + map_pts_pos_embed
    ).view(*map_pts_coord.shape)
```

### 3. 修改文件
- DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py
- DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py

### 3.1 具体改动（与 git status 一致）
#### drivetransformer_head.py
Online Mapping query 从随机初始化改为 embedding 权重扩展 batch 维度，并对齐到图像特征的设备与精度：
```python
# before
# map_query = torch.randn(bs, self.map_query.weight.shape[0], self.map_query.weight.shape[1]).to("cuda")
# map_reference_points = torch.randn(bs, self.map_reference_points.weight.shape[0], \
#                                      self.map_reference_points.weight.shape[1]).to("cuda")

# after
map_query = self.map_query.weight.unsqueeze(0).expand(bs, -1, -1).to(device=img_feats.device, dtype=dtype)
map_reference_points = self.map_reference_points.weight.unsqueeze(0).expand(bs, -1, -1).to(device=img_feats.device, dtype=dtype)
```

#### drivetransformer_layers.py
注意力 mask 增加 agent/map 双向可见：
```python
# project-2
mask[agent_range, map_range] = False
mask[map_range, agent_range] = False
mask[map_range, map_range] = False
```

map_pts_coord 的 refine 与 non-refine 回归：
```python
if self.refine:
    map_pts_coord = map_reg_branches[lid](
        map_query + map_pts_pos_embed
    ).view(*map_pts_coord.shape) + map_pts_coord
else:
    map_pts_coord = map_reg_branches[lid](
        map_query + map_pts_pos_embed
    ).view(*map_pts_coord.shape)
```

### 4. 输出
- 已生成 BEV 与 rgb_front 可视化视频：
  - [asset/demo_map.mp4](asset/demo_map.mp4) —— 完成 refine 后的静态地图感知效果
  - [asset/demo_map_original_noise.mp4](asset/demo_map_original_noise.mp4) —— 未 refine（原始噪声）的对照

### 5. 个人思考

#### 5.1 DriveTransformer 和 MapTR 的关系

读代码可以确认：**§2.3 的在线建图基本就是 MapTR 范式**，证据集中在几处：

- query 类型 `map_query_embed_type='instance_pts'`、每条 polyline 用**固定点数**表示（config 里 `map_fixed_ptsnum_per_pred_line=20`、`map_num_pts_per_vec`），这正是 MapTR「instance-level + point-level 分层 query、把地图元素建模成定长点序列」的做法。
- 匹配用 `MapHungarianAssigner3D` + `OrderedPtsL1Cost`（[config L409-414](../../../DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py#L409-L414)），并在 `assign` 时返回 `order_index`（[head L1166](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py#L1166)）。这就是 MapTR 的**置换等价（permutation-equivalent）匹配**——一条线正着画、反着画是同一条，匹配时在等价的点序中选代价最小的排列，避免方向歧义带来的伪监督。
- 损失是 `PtsL1Loss`（点回归）+ `PtsDirCosLoss`（相邻点方向余弦，约束形状/朝向），评测用 chamfer 距离（`tpfp_chamfer.py`），与 MapTR 的几何监督一致。

所以可以明确：DriveTransformer 没有重造地图分支，而是**把 MapTR 的矢量化建图整体接进统一 transformer**，让 map query 与 agent/ego query 在同一序列里通过注意力 mask 交互（§2.2）。区别只在于它不是独立网络，而是共享 backbone 与 self-attention 的一个任务头。

#### 5.2 为什么要 coarse-to-fine？什么情况下用？

§2.3 的 refine 分支本质是**用每一层的回归输出做残差，逐层修正上一层的坐标**（`pred = Δ_lid + pred_{lid-1}`），而不是每层都从零回归绝对坐标。这么做的好处：

1. **每层只学「小修正量 Δ」，比直接学绝对坐标容易**。残差的目标分布更集中在 0 附近、尺度更小，梯度更稳，深层 decoder 不容易发散——这和 DETR/Deformable-DETR 系列「iterative bounding box refinement」是同一套思路。
2. **前层给后层一个好的参考点**。地图点坐标会在每层重新编码成位置编码 `map_pts_pos_embed` 加进 query，坐标越准、PE 越准、全局注意力聚合出的特征越对，形成「坐标→PE→更准的坐标」的正反馈。（注：本实现的 sensor cross-attn 是标准全局注意力，坐标用作位置编码而非 deformable 采样位置，详见 §5.3。）
3. **天然匹配多层监督**。每层都出一版坐标、都算 loss，coarse-to-fine 让浅层负责大致定位、深层负责精修，分工明确。

什么情况下值得开 refine：**输出是带几何结构、可迭代逼近的连续量**（框、车道点、关键点），且有多层 decoder + 中间监督时收益最大；如果只有单层输出、或目标是纯分类/离散标签，refine 就没有意义。

#### 5.3 它和优化里「先离散后连续」是一回事吗？

**部分相通，但不是同一个机制。** 这里直接把三者并排对比：**(A) 先离散后连续优化、(B) multi-resolution（多重网格）优化、(C) MapTR 的 coarse-to-fine refine（本作业 §2.3）**。

三者**共享同一个直觉**：先求一个粗解，再在其邻域内精修，把难的全局问题拆成「先定位、后微调」，降低单步难度。差别在于「粗→精」靠什么实现、解决的是什么问题：

| 维度 | (A) 先离散后连续 | (B) multi-resolution / multigrid | (C) MapTR refine |
| --- | --- | --- | --- |
| 求解变量 | 离散阶段是整数/组合变量，连续阶段才切到实数 | 始终连续实数（网格节点场值） | 始终连续实数（点坐标） |
| 「尺度」来自 | 变量类型切换（离散↔连续） | 空间网格的疏密（spatial discretization） | 每层残差量级（注意力始终全局，无采样邻域变化） |
| 是否换表示 | 换：搜索空间/变量类型 | 换：粗网格↔细网格 | 不换：同一连续坐标空间，仅数值与注意力变化 |
| 主要解决 | **跳出非凸局部最优**（离散阶段定吸引域） | 加速收敛、消除不同频率误差 | 训练稳定 + 提供好初值（缩小每步学习目标） |

由此看清三件事：

1. **(A) 的「离散」是变量类型层面的离散**（0-1/整数松弛），它额外承担了**离散搜索、跳出局部最优**的职责；(B) 和 (C) 都没有这一层，全程连续。所以**把 (C) 等同于 (A) 会高估 refine——refine 不做组合搜索、不解决局部最优**。

2. **(B) 的「离散」只是空间域离散化**（网格疏密），求解量仍是连续实数，和 (A) 那种离散完全是两码事。正因如此，**(B) 比 (A) 更贴近 (C)**：两者都是「粗解作初值、连续精修」。

   > 举例：解二维泊松方程 $\nabla^2 u = f$（如热传导/静电势的稳态场），未知量 $u$ 是网格节点上的连续实数。multigrid 的做法是先在粗网格（如 $33\times33$）上迭代几步得到平滑的近似解 $u_{\text{coarse}}$——它快速消掉了误差里的低频（大尺度）成分；再把 $u_{\text{coarse}}$ **插值（prolongation）到细网格**（如 $129\times129$）作为初值继续迭代，专门消掉高频（细节）误差。整个过程 $u$ 始终是连续实数，「尺度」只来自网格疏密的切换，没有任何整数/组合变量——这正与 (C) 「浅层定大致位置、深层精修」的连续 coarse-to-fine 同构。
   >
   > 反例（属于 (A)）：旅行商 TSP 先用整数规划/贪心选出一条大致路线（**离散决策：访问顺序**），再对坐标做连续微调——这里的「粗」是离散组合搜索，承担了跳出大量局部解的职责，与 (B)/(C) 的连续插值/残差精修本质不同。

3. **(C) 连网格都不换**，那它的「不同尺度/精度」从哪来？——不在「空间网格疏密」，而在**每层残差量级**上。需要先澄清一个常见误解：

   > **本实现的 sensor cross-attn 不是 deformable attention，不在参考点附近采样。** 查 [drivetransformer_layers.py:32-138](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py#L32-L138) 的 `AttentionLayer`，它是**标准全局多头注意力**（`xops.memory_efficient_attention` 对**全部**图像 token 做 attention）。地图点坐标只以**位置编码** `map_pts_pos_embed` 的形式加进 query（每层用上一层坐标 `map_pts_coord.detach()` 重算一次，见 [L909](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py#L909)），**从不作为采样位置**。所以 MapTR 原版常配的「deformable 在参考点邻域采样、随坐标收敛而局部化」在这份代码里并不存在——之前若这么说是错的。

   - **真正的多尺度来源：残差量级随层递减**。每层做 `pred = Δ_lid + pred_{lid-1}`，浅层输出大幅度位移（把点从初始参考位置拉到大致正确区域 = 粗），深层只叠加越来越小的 Δ（局部微调 = 精）。「由粗到精」是修正步长在数值上自然衰减，不依赖换网格、也不依赖采样邻域变化。

     > **粗/精是「涌现」的，不是代码设置的。** 查 [head L487](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py#L487)：`map_reg_branches = _get_clones(map_reg_branch, num_pred)`，每层是同一个 branch 的独立 `deepcopy`——**层数、宽度、激活、xavier 初始化全部相同**，浅层与深层在代码上没有任何参数/结构差异，也**没有对 Δ 做任何显式缩放或特殊初始化**去强迫深层输出变小（branch 定义见 [L445-451](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py#L445-L451)，仅 LayerNorm+Linear+SiLU）。
     >
     > 那「深层只学小残差」从何而来？答案是**残差连接 + 每层都有监督**两者共同决定的：因为坐标按 `pred = Δ + pred_{lid-1}` 累加，且每层输出都参与 loss，浅层一旦把点拉到大致正确位置，**留给深层的真值与当前预测之差（即它需要回归的目标）本身就变小了**——深层学习的对象就是「上一层剩下的残差」。这是训练数据/优化驱动的结果，不是某个超参或层级配置写死的。
     >
     > （唯一的代码特例是 `map_reg_branches[-1]` 被替换成宽输出层 [L488](../../../DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_head.py#L488)，那是 prep 阶段一次性回归全部控制点用的，与逐层 refine 的粗/精机制无关。）
   - **坐标的作用是「喂位置编码」而非「定采样点」**：坐标越准 → 每层重算的 `map_pts_pos_embed` 越准 → 全局注意力的 query 表示越好 → 回归出更准的坐标，形成「坐标→PE→更准坐标」的正反馈。这是连续空间里的隐式精化，但**注意力始终是全局的，感受野不随层收缩**。
   - 每一层都会跑一次 sensor cross-attn 并重算 PE（refine 是逐层的），但每层都看全图，不存在「逐层聚焦到小邻域」的过程。

一句话总结：**(B) 用「换更密的网格」、(C) 用「逐层减小的残差」（注意力始终全局、并非靠采样邻域收缩）分别在连续空间里实现多尺度；(A) 则靠「离散↔连续切换」额外解决局部最优。(C) 与 (B) 近、与 (A) 远。**