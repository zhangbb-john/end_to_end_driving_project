# Project2 静态地图感知模型实现与可视化

## 1. 项目目标

实现基于DriveTransformer([https://arxiv.org/abs/2503.07656](https://arxiv.org/abs/2503.07656))的在线静态感知，主要理解以下几个部分：

- 在线建图Query构建；
- 静态地图注意力mask设置；
- 地图点坐标精细化预测；

最终，实现静态地图感知以及可视化。

## 2. 作业思路提示

完成本作业需要重点关注以下实现要点：

- 完成在线建图Query的初始化，并为其增加batch维度，适配批量计算；
- 基于注意力掩码矩阵，定义 agent/map/planning 的交互规则，实现指定主体间的注意力可见性控制；
- 利用粗到精残差优化策略，完成静态目标坐标的回归优化，提升坐标预测精度；
- 全程校验张量运算的合法性，确保维度匹配、设备一致。

## 3. 操作步骤

提示：（1）project-2的开发需要在完成project-1的基础之上进行；（2）可以通过在代码中搜索关键词"project-2"，快速定位需要补全的代码区域。

### 3.1 在线建图query构建

**代码路径：**
```
drivetransformer_head.py
```

掌握 PyTorch 中`nn.Embedding`层的权重调用方式，实现从嵌入层权重的`(N, D)`形状，广播扩展为适配批量计算的`(bs, N, D)`形状，替换当前随机数生成的临时代码，完成**在线建图（Online Mapping）** 模块的基础实现，为后续静态地图计算提供合法的查询特征与参考点特征。

1. `nn.Embedding`层的核心属性：`weight`为嵌入层的可学习权重矩阵，形状固定为`(N, D)`（`N`为嵌入词典大小，`D`为嵌入维度）；
2. PyTorch 张量的批量扩展：如何将二维张量`(N, D)`扩展为三维批量张量`(bs, N, D)`，保证批量计算时维度匹配；
3. 设备一致性：保证生成的张量与原嵌入层权重在同一计算设备（如 CUDA），避免设备不匹配错误。

根据提示替换以下代码：

```python
###################################################################
## Online Mapping
###################################################################
# project-2
# TODO-4
# map_query = nn.Embedding.weight (N, D) -> (bs, N, D)
# map_reference_points = nn.Embedding.weight (N, D) -> (bs, N, D)
# 替换此处代码
map_query = torch.randn(bs, self.map_query.weight.shape[0], self.map_query.weight.shape[1]).to("cuda")
map_reference_points = torch.randn(bs, self.map_reference_points.weight.shape[0], \
                                     self.map_reference_points.weight.shape[1]).to("cuda")

###################################################################
```

### 3.2 Attention Mask 适配

**代码路径：**
```
drivetransformer_layers.py
```

在完成`project-1`实现动态OD的注意力计算的基础上，实现静态地图的注意力mask适配，使得map query和agent query之间互相可见。

#### 核心知识点

1. 注意力掩码的语义：布尔型掩码矩阵`(total, total)`中，`mask[i,j]=True`表示第`i`个查询无法关注第`j`个键，`mask[i,j]=False`表示可关注；
2. 张量切片与索引：利用`slice`定义的区间对二维掩码矩阵进行区域赋值，实现**指定主体间的可见性控制**；
3. 多项目差异化规则：根据 project-1 和 project-2 的不同需求，设计不同的注意力交互逻辑，理解工程中 "模块化掩码设计" 的思想；
4. 设备与类型一致性：基于已创建的基础掩码（`bool`类型、指定设备）进行赋值，无需重新创建张量。

#### 前置条件与变量说明

代码中已完成基础准备工作，核心变量含义如下：

- `n_agent`：智能体（agent）查询数量，对应掩码**前 n_agent 行/列**；
- `n_map`：地图（map）查询数量，对应掩码**n_agent ~ n_agent+n_map 行/列**；
- `n_ego`：自车（ego）查询数量，固定为 1，对应掩码**最后 1 行/列**；
- `total`：总查询数（`n_agent + n_map + n_ego`），掩码矩阵形状为`(total, total)`；
- `mask`：基础掩码矩阵，初始为全`True`（所有位置均遮蔽，需按规则修改为`False`表示可见）；
- `agent_range/map_range/ego_range`：已定义的三类查询的切片区间，直接用于张量索引；
- 掩码规则：**先按 project-1 赋值，再基于 project-1 的结果按 project-2 规则覆盖赋值**。

```python
##############################################################
# project-1
# TODO-6 attention mask, ego query 
n_agent = agent_query_num
n_map = map_query_num
n_ego = 1
total = n_agent + n_map + n_ego
# 创建基础掩码（True表示需要mask）
mask = torch.ones(total, total, dtype=torch.bool, device=query.device)
# 定义任务区间
agent_range = slice(0, n_agent)
map_range = slice(n_agent, n_agent + n_map)
ego_range = slice(n_agent + n_map, total)
# 设置任务间不可见（mask=True）
# 任务内部可见（mask=False）
##############################################################
# project-1
# 允许agent看agent
# 替换此处代码
pass
# 允许map看agent, map
# 替换此处代码
pass
###############################################################
# project-2
# 允许agent看agent, map
# 替换此处代码
pass
# 允许map看agent, map
# 替换此处代码
pass
###############################################################
```

### 3.3 静态地图坐标点预测

**代码路径：**
```
drivetransformer_layers.py
```

#### 核心知识点

掌握**粗到精（coarse-to-fine）** 优化的核心思想，完成在线建图（Online Mapping）模块中地图点坐标（`map_pts_coord`）的计算逻辑；实现「原始坐标 + 隐式坐标」的叠加更新，理解精细化回归的工程实现方式，完成`refine`模式（精细化）和非`refine`模式（基础）下的坐标回归代码开发。

1. 粗到精优化：先通过网络回归隐式坐标，再加原始坐标，实现坐标的精细化表示，提升回归精度；
2. 网络分支调用：`map_reg_branches[lid]`为地图坐标回归的专用网络分支（`nn.Module`实例），输入特征张量可直接输出对应坐标；
3. 张量形状一致性：保证回归的隐式坐标与原始坐标形状完全一致，才能进行逐元素加法叠加；
4. 条件分支逻辑：根据`self.refine`布尔值，分别实现「精细化修正」和「基础回归」两种逻辑，保证代码兼容性。

```python
# online mapping
#########################################################################
# project-2 
if self.refine: # coarse-to-fine optimization
    # TODO-7 get map_pts_coord
    # input: map_query + map_pts_pos_embed
    # map_pts_coord_refine = map_reg_branches[lid](input)  # shape -> map_pts_coord.shape
    # map_pts_coord = map_pts_coord_refine + map_pts_coord
    pass
else:
    # map_pts_coord = map_reg_branches[lid](input)  # shape -> map_pts_coord.shape
    pass
#########################################################################
```

## 4. 作业提交说明

### 4.1 提交内容

- 相关改动代码的文件
- BEV视角和rgb_front的可视化视频

### 4.2 命名规则

压缩包命名：`用户名-Project2.zip`，压缩包大小建议小于100M。
