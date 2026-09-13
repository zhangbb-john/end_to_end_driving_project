# RTX 6000 Linux 配置命令（基于 notion.md 改编）
## 版本
PyTorch ≥ 2.7（即 cu128 wheel 所在版本线）已经**不再发布 Python 3.8 的 wheel**，
   官方支持范围是 Python 3.9 ~ 3.13。继续走下去 `pip install torch ... cu128` 大概率会报：
   ```
   ERROR: Could not find a version that satisfies the requirement torch
   ```
   或者只装到一个老版本（< 2.7），那个版本根本不支持 sm_120，等于白干。

### cmd1 重建 env 为 Python 3.10：
   ```bash
   conda deactivate
   conda env remove -n drivetransformer -y
   conda create -n drivetransformer python=3.10 -y
   conda activate drivetransformer
   python --version   # Python 3.10.x
   ```
### cmd2 cuda 还是118，强制130
```
   conda remove --force-remove cuda cuda-toolkit cuda-runtime cuda-nvcc \
    cuda-cudart cuda-cudart-dev cuda-libraries cuda-libraries-dev \
    cuda-nvrtc cuda-nvrtc-dev cuda-tools cuda-compiler -y 2>/dev/null

    conda list | awk '/^cuda/ {print $1}' | xargs -r conda remove -y --force-remove

    conda install -c nvidia \
    cuda-toolkit=13.0.* \
    -y
```
### cmd3 torch
```
# 换源（可选）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 cu130 最新稳定版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
# 匹配 cu130
pip install -U xformers --index-url https://download.pytorch.org/whl/cu130
import torch
print(torch.__version__)          # 2.11.0+cu130 之类
print(torch.version.cuda)         # 13.0
print(torch.cuda.is_available()) # True
print(torch.cuda.get_device_name(0)) # RTX 5090
```

**Step 4. 验证切换成功**

```bash
which nvcc          # 应指向 12.8 的 nvcc，而不是旧的 /usr/local/cuda-11.8/bin/nvcc
nvcc --version      # release 必须显示 12.8
python -c "import torch; print(torch.version.cuda)"   # 期望 12.8
```
### cmd4 依赖
```
## 7. 安装编译依赖
pip install ninja packaging

## 8. 安装项目核心依赖（重新编译 CUDA 扩展）
cd DriveTransformer
# 确保以 sm_120 重新编译 iou3d / roiaware_pool3d / mmcv._ext
export TORCH_CUDA_ARCH_LIST="12.0"
pip install -v -e .
```

## 报错记录
### torch 明明装好了，但执行 pip install -e . 时却说找不到 torch   torch 2.12.0+cu130 requires setuptools<82, but you have setuptools 82.0.1 which is incompatible.
```
# 1. 降级 setuptools 到 torch 2.12 cu130 支持的版本（关键！）
pip install setuptools==70.2.0 -U

# 2. 关闭 pip 隔离构建（让构建时能看到已装的 torch）
export PIP_NO_BUILD_ISOLATION=0

# 3. 再安装 -e. 就成功了
pip install -e .

# 4. 验证能导入
python -c "import torch; import drivetransformer; print('OK')"

```
#### 测试mmcv
```
import torch

# 测试 mmcv iou3d_cuda 算子 + CUDA 可用性
def check_cuda_ops():
    print("=" * 50)
    print("开始检查 CUDA 算子与环境...")
    print("=" * 50)

    # 1. 基础 CUDA 检查
    if torch.cuda.is_available():
        print(f"✅ PyTorch CUDA 可用")
        print(f"✅ 显卡型号: {torch.cuda.get_device_name(0)}")
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 版本: {torch.version.cuda}")
    else:
        print("❌ CUDA 不可用")
        return

    # 2. 测试 mmcv iou3d_cuda 算子
    try:
        from mmcv.ops.iou3d_det import iou3d_cuda
        print("✅ iou3d_cuda 算子加载成功！")

        # 3. 测试真实 CUDA 调用（验证算子/架构兼容）
        pts = torch.randn(10, 3).cuda()
        print("✅ CUDA 算子调用成功，sm_120 架构兼容正常！")

    except ImportError as e:
        print(f"❌ 算子导入失败: {e}")
    except Exception as e:
        print(f"❌ 算子运行失败: {e}")

    print("=" * 50)

if __name__ == "__main__":
    check_cuda_ops()
```
### ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
option1
```
conda activate drivetransformer
conda install -c conda-forge libstdcxx-ng -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

option2
```
conda install -y libstdcxx-ng
```
option3
```
conda activate drivetransformer
conda install -y libstdcxx-ng
conda install -y gcc=12.2.0

# 然后重新运行训练
bash adzoo/drivetransformer/dist_train.sh adzoo/drivetransformer/configs/drivetransformer_large.py 1
```
忘记那条管用了？

best
```
conda activate drivetransformer
conda install -c conda-forge libstdcxx-ng -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```
## RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy .tar format.
sed -i 's/torch.load(filename, map_location=map_location)/torch.load(filename, map_location=map_location, weights_only=False)/' /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/mmcv/utils/checkpoint.py


## AttributeError: module 'numpy' has no attribute 'bool'

sed -i 's/np.bool/bool/g' /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/datasets/pipelines/filter.py



## ImportError: cannot import name 'Iterable' from 'collections'

```
# 1. 确保版本正确
pip install motmetrics==1.1.3

# 2. 直接修复正确路径下的文件（这次 100% 能找到）
sed -i 's/from collections import OrderedDict, Iterable/from collections import OrderedDict\nfrom collections.abc import Iterable/' /root/miniconda3/envs/drivetransformer/lib/python3.10/site-packages/motmetrics/metrics.py

```

## RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy .tar format.

checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
## AttributeError: module 'numpy' has no attribute 'bool'

sed -i 's/np.bool/bool/g' /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/datasets/pipelines/filter.py
### NameError: name 'bool_' is not defined. Did you mean: 'bool'?
```sed -i 's/bool_/bool/g' /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/datasets/pipelines/filter.py
```

## PermissionError
 [Errno 13] Permission denied: 'adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_large/drivetransformer_large.py

 .../logs/train.05302001: Permission denied
```
#到root（不是ubuntu用户）

# 给 ubuntu 写入训练输出目录权限
chown -R ubuntu:ubuntu /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/adzoo/drivetransformer/work_dirs
chmod -R u+rwX,go+rX /root/autodl-tmp/end_to_end_driving/home_work/DriveTransformer/adzoo/drivetransformer/work_dirs

# 让 ubuntu 能用 conda
echo ". /root/miniconda3/etc/profile.d/conda.sh" >> /home/ubuntu/.bashrc
su - ubuntu -c "source ~/.bashrc"
```

## ValueError: Unexpected keyword arguments: temp_attn_masks
torch.utils.checkpoint 调用（见堆栈里 drivetransformer_layers.py 的 cp.checkpoint(self._forward, *args, **kwargs)），说明你传了 temp_attn_masks 但当前 PyTorch 的 checkpoint 不接受这个关键字参数。

### 根因

报错只在开启 `with_cp=True`（梯度检查点）后才出现，定位在 [drivetransformer_layers.py:853](../../../home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py#L853)：

```python
def forward(self, *args, **kwargs):          # L848
    if self.use_checkpoint and self.training:
        x = cp.checkpoint(self._forward, *args, **kwargs)   # L853  ← 报错处
    else:
        x = self._forward(*args, **kwargs)   # L855  ← 这条路径正常
```

调用链:上层 L958 用**关键字**方式传入 `layer(..., attn_mask, temp_attn_masks=temp_attn_masks)`，于是 `temp_attn_masks` 落进 `forward` 的 `**kwargs`，再被原样转发给 `cp.checkpoint`。

问题在于新版 PyTorch（torch>=2.x）`torch.utils.checkpoint.checkpoint` 的签名是:

```python
def checkpoint(function, *args, use_reentrant=..., context_fn=..., determinism_check=..., debug=..., **kwargs)
```

它会**先把传入的 kwargs 当成 checkpoint 自己的配置项解析**，遇到不认识的 `temp_attn_masks` 就抛 `ValueError: Unexpected keyword arguments`。也就是说，本该转发给 `_forward` 的业务关键字参数，被 checkpoint 误当成了它自己的参数。

对比 L855 的 else 分支为什么不报错:因为 `self._forward(*args, **kwargs)` 直接调用，而 `_forward` 的签名里明确有 `temp_attn_masks=None`（L711），自然能接住。问题只存在于 checkpoint 这条路径。

### 方案 A 原理:用闭包"吃掉"kwargs + 关闭 reentrant

核心思路:**不让业务 kwargs 经过 checkpoint 的参数解析**。做法是用一个 lambda 闭包把 `**kwargs` 捕获进去，对 `cp.checkpoint` 只暴露位置参数 `*args`，这样 checkpoint 的 kwargs 解析里就只剩它自己认识的配置项。

```python
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                lambda *inputs: self._forward(*inputs, **kwargs),
                *args,
                use_reentrant=False,
            )
        else:
            x = self._forward(*args, **kwargs)
```

两个关键点:

1. **`lambda *inputs: self._forward(*inputs, **kwargs)`**：`kwargs`（含 `temp_attn_masks`）通过闭包直接进入 `_forward`，绕开了 checkpoint 的参数解析 → 报错消失。`*args`（真正参与前向/梯度的张量）仍按位置传给 checkpoint，保证重算时它们能被正确追踪。

2. **`use_reentrant=False`**：新版 PyTorch 推荐值，必须显式设置。
   - 旧的 reentrant 实现对"非张量参数 / 关键字参数 / 输入中无 requires_grad 张量"支持很差，本模型的 `attn_masks`、`temp_attn_masks` 正是非梯度的 mask（张量/列表），用默认 reentrant 很可能触发另一个错。
   - `use_reentrant=False`（非重入实现）没有这些限制，能正确处理只读的 mask 参数，是当前官方推荐的稳定路径。

正确性说明:`temp_attn_masks` 是只读的注意力掩码，**不参与梯度**。它被闭包捕获、不被 checkpoint 追踪 RNG/版本，对结果没有影响;真正需要重算的张量都通过 `*args` 正常进入 checkpoint。因此该改法只解决参数透传问题，**不改变任何数学结果**，与 `with_cp` "对输出/梯度/曝光中性"的性质一致。

### 实施 plan

1. 改 [drivetransformer_layers.py:852-853](../../../home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py#L852)，把单行 `cp.checkpoint(self._forward, *args, **kwargs)` 替换为上面的 lambda + `use_reentrant=False` 写法。
2. 仅改这一处即可:三处 `with_cp=True` 共用同一个 `forward`，这里修好后全部生效。（其余 cp.checkpoint 调用 L177/L232/L465 只传位置参数，不受影响，无需改。）
3. 重新启动训练验证:
   ```bash
   bash adzoo/drivetransformer/dist_train.sh \
       adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py 4
   ```
   确认不再报 `temp_attn_masks` 错、前几十个 iter 正常打印 loss，并观察日志 `memory:` 峰值确认显存已下降。

### 具体改动:修改前 / 修改后

文件:[drivetransformer_layers.py](../../../home_work/DriveTransformer/adzoo/drivetransformer/mmdet3d_plugin/ours/drivetransformer_layers.py)，类 `DriveTransformerDecoderLayer.forward`。

**修改前(L848-857):**

```python
    def forward(self,
                *args,
                **kwargs
                ):
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(self._forward, *args, **kwargs)   # L853 报错:temp_attn_masks 被 checkpoint 当成自己的 kwarg

        else:
            x = self._forward(*args, **kwargs)
        return x
```

**修改后(L848-859):**

```python
    def forward(self,
                *args,
                **kwargs
                ):
        if self.use_checkpoint and self.training:
            # 用闭包把业务 kwargs(含 temp_attn_masks)捕获进去,只给 checkpoint 暴露位置参数;
            # use_reentrant=False 走非重入实现,正确支持非梯度的 mask 参数。
            x = cp.checkpoint(
                lambda *inputs: self._forward(*inputs, **kwargs),
                *args,
                use_reentrant=False,
            )
        else:
            x = self._forward(*args, **kwargs)
        return x
```

**改动点小结:**

| 项 | 修改前 | 修改后 |
|---|---|---|
| 传给 checkpoint 的函数 | `self._forward`(裸函数) | `lambda *inputs: self._forward(*inputs, **kwargs)`(闭包捕获 kwargs) |
| 传给 checkpoint 的参数 | `*args, **kwargs`(kwargs 被误解析) | 仅 `*args`(kwargs 已进闭包) |
| reentrant 模式 | 未指定(默认 True,易出错) | 显式 `use_reentrant=False` |
| `else` 分支(L855-856) | 不变 | 不变 |
| L711 的 `_forward` 签名 | 不变(本身已含 `temp_attn_masks=None`) | 不变 |

> 注意:这是源码补丁(改的是模型代码,不是 config),不会被 config 覆盖;`git stash`/切分支时记得带上这一处改动。

### 备选方案 B（不推荐）

把主 decoder layer（config L313）的 `with_cp` 改回 `False` 可绕过报错，但这层是 12 层 decoder、激活大头，关掉后显存收益最大的部分就没了，batch=4 很可能重新 OOM。仅在临时排查时用，正式训练应采用方案 A。
### A方案报措
RuntimeError: Expected to mark a variable ready only once.
Parameter at index 155 with name pts_bbox_head.map_prep_decoder.layers.0.ffns.0.postnorm.weight has been marked as ready twice.
 
#### 根因:DDP + 梯度检查点 + find_unused_parameters 三者冲突

这个错和上一个(`temp_attn_masks`)是连锁的——**正是上一步把 `with_cp` 打开后才出现的**。三个东西凑在一起:

1. **DDP 的梯度同步机制**:DistributedDataParallel 给每个参数挂一个 autograd hook,某参数的梯度算完(ready)就标记一次、触发 allreduce。**每个参数每次 backward 只应被标记 ready 一次**。
2. **梯度检查点(`use_reentrant=False`)**:反向传播时会**重算一遍前向**(这正是它省显存的代价)。重算会**重建被包裹区域的 autograd 子图**。
3. **`find_unused_parameters=True`**(本项目 config 设了,[train.py:302](../../../home_work/DriveTransformer/adzoo/drivetransformer/train.py#L302) 读取):为支持"某些参数某些 batch 不参与"的情况,DDP 会在前向结束后**额外遍历一次计算图**来找未使用参数。

冲突点:`find_unused_parameters` 的那次额外遍历 + checkpoint 重算重建的子图,导致同一个参数(报错里的 `map_prep_decoder...postnorm.weight`)的 ready hook **被触发了两次**,DDP 检测到"标记两次"就抛 `Expected to mark a variable ready only once`。报错参数落在 `map_prep_decoder` 上,正是我们 `with_cp=True` 包裹的 prep decoder。

#### 解决思路:启用 static_graph=True

PyTorch 官方对"DDP + 激活检查点"的推荐解法是给 DDP 设 **`static_graph=True`**。它的作用:

- **只在第一个 iteration 记录一次计算图,后续复用**,因此一个参数不会被重复标记 ready → 直接消除"marked ready twice"。
- 它**原生支持**:reentrant/非 reentrant 反向、同一模块被 checkpoint 多次、forward 之外的参数等过去不支持的情形。
- 它**自带未使用参数检测**(只在首个 iter 做一次),所以可以**同时把 `find_unused_parameters` 关掉**,既解决冲突又少一次每步遍历、略微提速。

改动位置:[train.py:301-307](../../../home_work/DriveTransformer/adzoo/drivetransformer/train.py#L301) 的 DDP 构造。

**修改前:**
```python
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=find_unused_parameters
                                        )
```

**修改后:**
```python
    if distributed:
        model = DistributedDataParallel(model.cuda(),
                                        device_ids=[torch.cuda.current_device()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=False,  # static_graph 自带未使用参数检测
                                        static_graph=True,             # 关键:解决 checkpoint 下 marked-ready-twice
                                        )
```

#### ⚠️ 使用 static_graph 的前提与验证

`static_graph=True` 的假设是:**每个 iteration 用到的参数集合和控制流保持一致(图是"静态"的)**。本模型是检测/规划多头 + 流式时序结构,需要确认它满足这个前提:

- 各注意力/FFN 模块在 `operation_order` 里是固定调用的,网络结构层面**参数集合每步一致**,所以通常满足 static 假设;Hungarian 匹配、GT 数量变化只影响 loss 计算、不改变哪些**模型参数**产生梯度。
- 但流式时序(`prev_exists`、`memory_len_frame`)的首帧 vs 后续帧若导致某些参数时用时不用,static_graph 会给出**错误梯度甚至崩溃**。这点需要实测验证。

**验证方法**:改完跑前 ~100 个 iter,确认:① 不再报 marked-ready-twice;② loss 正常下降、不 NaN;③ 跨越首帧/换 batch 边界时不崩。若崩了或 loss 异常,见下方备选。

#### 备选(若 static_graph 不适用)

- **备选 1**:保留 `find_unused_parameters=True`,但把 prep decoder 两处(config L258/L287)的 `with_cp` 改回 `False`,只在主 decoder(L313)保留 checkpoint。报错参数在 prep decoder,关掉它的 checkpoint 即可避开冲突,而主 decoder(12 层、激活大头)的省显存收益基本保留。代价:prep decoder 的激活重新常驻,显存略升,需复测峰值。
- **备选 2**:确认模型确实没有"未使用参数"时,直接 `find_unused_parameters=False` 且不开 static_graph。但本项目原作者设了 `True`,大概率有条件未使用参数,贸然关闭可能报另一个错(`Expected to have finished reduction...`),不推荐先试。
### A方案资源占用情况是否健康

实例面板快照(北京B区/908机,RTX PRO 6000 × 4卡,包年包月,2026-06-01 到期前 15 天释放):

| 指标 | 数值 | 判断 |
|---|---|---|
| 状态 | 运行中 / 正常 | ✅ |
| CPU | 11% | ✅ 偏低,正常(训练瓶颈在 GPU,CPU 只做数据加载/调度) |
| **内存(系统 RAM)** | **91%** | ⚠️ **偏高,需盯紧**(见下) |
| 系统盘 | 70.70% | 🟡 偏高,留意 |
| 数据盘 | 37.55% | ✅ 充足 |

**先说一个关键缺口**:这个面板**没有显示 GPU 利用率和显存占用**,而这俩才是判断训练是否健康的核心指标。光看面板不够,必须上机用 `nvidia-smi` 看。下面分指标说明。

#### 1. CPU 11% — 正常

训练的计算瓶颈在 GPU,CPU 主要负责 dataloader 取数和进程调度。11% 说明数据加载没成为瓶颈,健康。如果 CPU 长期跑满而 GPU 利用率低,才需要调 `workers_per_gpu`。

#### 2. 内存 91% — 偏高,是当前最该警惕的点

系统 RAM 到 91% 有真实 OOM 风险(被内核 OOM killer 杀进程 → 训练中断)。这个项目吃内存的几个来源:

- `workers_per_gpu=12` × 4 卡 = **48 个 dataloader worker 进程**,每个都持有数据缓存;
- config 里 `cache_lenth = batch_size+1` 的样本缓存 + 流式时序的历史帧;
- `b2d_map_infos.pkl` 有 **6.3GB**,加载进内存后常驻。

**判断**:91% 本身还没崩,但余量很小,长时间训练或遇到数据尖峰可能触顶。建议:
- 上机跑 `free -h` 和 `watch -n5 free -h` 看是否持续逼近 100% / swap 是否被大量占用;
- 若持续 >95%,优先把 `workers_per_gpu` 从 12 降到 8 或 6(48→32/24 个 worker,显著降内存,代价是数据加载略慢);
- 留意 `dmesg | grep -i oom` 有没有 OOM kill 记录。

#### 3. 系统盘 70.70% — 留意,别让 work_dirs 写爆

训练 checkpoint、日志都写在盘上。config 里 `checkpoint_config = dict(interval=3000)`,large 模型单个 ckpt 不小,200000 iter / 3000 ≈ **66 个 checkpoint**,累积可能几十上百 GB。系统盘已 70%,需确认 work_dirs 落在数据盘(37%,空间足)而非系统盘;否则训练中途写满盘会直接崩。建议核对 `WORK_DIR` 指向,并考虑只保留最近 N 个 ckpt(`max_keep_ckpts`)。

#### 4. 真正要补看的:GPU(面板没给)

上机执行,这才是判断 A 方案(开 checkpoint 后)是否健康的决定性指标:

```bash
watch -n 2 nvidia-smi
```

重点看三项:

| 看什么 | 健康标志 | 不健康的信号 |
|---|---|---|
| GPU-Util | 4 卡都稳定在 **85~100%** | 长期 <50% 或剧烈抖动 → 数据加载/同步成瓶颈 |
| 显存占用 | 远低于 97GB(开 cp 后预期大幅下降) | 接近 97GB → 还会 OOM,需降 batch |
| 4 卡是否均衡 | 利用率/显存四卡接近 | 某卡明显低 → DDP 负载不均或卡间通信问题 |

也可直接看训练日志里的 `memory:` 字段(单位 MB),对比开 cp 前(约 17933 ≈ 18GB)是否明显下降——下降了才说明梯度检查点真正生效。

#### 小结

- CPU/数据盘:健康,无需动作。
- **内存 91%:当前最大隐患**,上机确认趋势,必要时降 `workers_per_gpu`。
- 系统盘 70%:确认 ckpt 写在数据盘 + 限制保留数量。
- GPU:面板没给,**必须 `nvidia-smi` 实测**,确认 4 卡高利用率、显存有余量、日志 `memory:` 较开 cp 前下降——这三点都满足,A 方案才算健康。

#### 建议
workers_per_gpu改哪个文件

就改训练用的那个 config 文件:[drivetransformer_large.py](../../../home_work/DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py) 的 **L463**(在 `data = dict(...)` 里):

```python
data = dict(
    samples_per_gpu=batch_size,   # L462
    workers_per_gpu=12,           # L463 ← 改这里
    ...
```

**修改前 → 修改后**(内存 91% 偏高时):

```python
    workers_per_gpu=12,   # 12 × 4卡 = 48 个 dataloader worker 进程
```
改为:
```python
    workers_per_gpu=8,    # 8 × 4卡 = 32 个 worker,内存明显下降;若仍紧张降到 6
```

说明:
- `workers_per_gpu` 是**每张卡**的 dataloader worker 进程数,实际进程数 = `workers_per_gpu × 卡数`。12×4=48 个进程,每个都持有数据缓存,是内存吃紧的主因之一。
- 降到 8(32 进程)或 6(24 进程)能显著省内存,**代价是数据加载略慢**;只要 GPU 利用率仍保持高位(`nvidia-smi` 看 GPU-Util 不掉),就说明没拖慢训练,可放心降。
- 这是纯运行期参数,改完直接重启训练生效,**不需要重新编译/安装**。
- 注意:这一项不要去改 `_base_` 里的 dataset 基类配置,以当前 large config 里的 L463 为准(它会覆盖基类)。

> 配套排查:若降了 workers 内存仍逼近 100%,再查 `dmesg | grep -i oom` 是否已有 OOM kill,并用 `watch -n5 free -h` 看 swap 是否被大量占用。

# Testing 阶段：start_eval.sh 段错误(Segmentation fault)排查

## 现象

训练已经能跑通,但 testing(闭环评测)起不来。运行 `home_work/Bench2Drive/start_eval.sh` 时进程直接崩溃:

```
start_eval.sh: line 36:  6583 Segmentation fault      (core dumped) python leaderboard/leaderboard/leaderboard_evaluator.py --routes=... --port=30001 ...
```

日志(`start_eval_debug.log`)里只看到 import 阶段的几行 warning,然后立刻 `Segmentation fault (core dumped)`,程序连 `_setup_simulation`(启动 CARLA server)都没进。

## 定位过程:在 import 前后加打印

段错误是 C 层崩溃,不会抛 Python 异常、也没有 traceback,所以靠"分段打印 + flush"来缩小崩溃位置。在 `leaderboard/leaderboard/leaderboard_evaluator.py` 的每个关键 import 前后插入探针(注意经 `tee` 管道时缓冲会丢,必须 `flush=True`):

```python
import sys
print(f"[import-probe] python executable = {sys.executable}", flush=True)
print(f"[import-probe] python version = {sys.version}", flush=True)
print("[import-probe] BEFORE import carla", flush=True)
import carla
print(f"[import-probe] AFTER import carla -> {carla.__file__}", flush=True)
```

运行后输出停在:

```
[import-probe] python version = 3.10.20 (main, ...) [GCC 14.3.0]
[import-probe] BEFORE import carla
=== exit code = 139 ===          # 139 = 128 + SIGSEGV(11)
```

`AFTER import carla` 这一行从未打印 —— **崩溃点精确锁定在 `import carla`**。

进一步用最小命令复现,排除其它干扰:

```bash
python -c "import carla"   # 单独执行同样 exit 139,与 CARLA server 是否启动无关
```

## 根本原因:Python 解释器版本与 carla egg 的 ABI 不匹配

- 当前 conda 环境的 Python 是 **3.10.20**。
- 但 `PYTHONPATH` 里挂载的 carla 客户端库是为 **Python 3.7** 编译的二进制 egg:
  `carla-0.9.15-py3.7-linux-x86_64.egg`,内部是 `carla/libcarla.cpython-37m-x86_64-linux-gnu.so`(文件名里写死了 `cpython-37m`)。
- 这个 `.so` 不显式链接 `libpython`,而是引用一堆未定义的 `Py*` 符号(`PyDict_New`、`PyArg_ParseTupleAndKeywords` 等),由**加载它的解释器进程**在运行时提供。
- 用 3.10 解释器去加载 3.7 编译的扩展,**CPython 的 C-ABI(对象内存布局、引用计数宏、类型结构体字段偏移)对不上** —— 函数名虽然都在,但 `Py_INCREF` 等内联宏按 3.7 的固定偏移去读写 3.10 的 `PyObject`,踩坏内存,直接段错误。

**为什么之前 Python 3.8 能用同一个 egg?** 因为 CPython 普通扩展只保证**同一小版本**内 ABI 稳定。3.7↔3.8 的对象布局、调用约定几乎没动,3.7 的 `.so` 在 3.8 上属于"侥幸兼容";而 3.8→3.9→3.10 累积了破坏性改动(vectorcall、`tp_print` 移除、类型结构体重排等),漂移到 3.10 时彻底对不上。这个 egg 不是 `abi3`(限定 API)编译,没有跨版本兼容保护。

## 关键澄清:CARLA server 能跑 ≠ Python 3.10 可用

排查中一度误以为"`./CarlaUE4.sh` 能正常启动"说明 3.10 没问题。实际读代码确认:

- `CarlaUE4.sh` 只是个 shell 包装,最终执行原生二进制 `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping`。
- `ldd` 该二进制**不链接任何 libpython** —— 它是 UE4(C++)程序,启动时根本不加载 Python 解释器。
- server 内部的 libcarla 由 C++ 直接链接,绕过 Python 导入机制,所以"用哪个 Python"对它毫无意义。

也就是说:server(`CarlaUE4-Linux-Shipping`)和客户端 egg 用的是同一套 C++ 引擎代码,但编成两个产物 —— server 给 C++ 直接用、不挑 Python;客户端 egg 绑定到具体 Python 版本。server 能跑只能证明显卡/驱动/UE4 环境正常,无法证明客户端 `import carla` 能用。

## 可能的解决方案

按推荐程度排序:

1. **回到 Python 3.8 环境(推荐,最稳)** —— 与现有 `carla-0.9.15-py3.7` egg 验证过可用。
   ```bash
   conda create -n drivetransformer38 python=3.8
   conda activate drivetransformer38
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install -v -e .            # 重新编译 mmcv / iou3d / roiaware_pool3d 等 CUDA 扩展
   ```
   然后把 `start_eval.sh` 里的 `PYTHON="python"` 指向该 3.8 解释器。
   ⚠️ 注意:RTX PRO 6000 是 Blackwell 架构,CUDA 11.8 不一定支持其算力,需确认 torch/CUDA 版本与显卡兼容(这正是当初升到新环境的原因)。若 3.8 + cu118 无法驱动 Blackwell,则此方案不可行,走方案 2。

2. **给 Python 3.10 装匹配的 carla wheel** —— CARLA 0.9.15 官方提供 cp310 的 wheel:
   ```bash
   pip install carla==0.9.15
   ```
   装好后**删掉 `start_eval.sh` 中 `PYTHONPATH` 里的 py3.7 egg 路径**(`.../dist/carla-0.9.15-py3.7-linux-x86_64.egg`),避免旧 egg 被优先加载。
   ⚠️ 需同时确认 DriveTransformer / mmcv 的其它依赖在 3.10 下能正常编译运行(改动面较大,但能保留 Blackwell + 新 CUDA 的训练环境)。

3. **自行用 Python 3.10 重新编译 carla PythonAPI** —— 最后手段,需要 CARLA 源码并配置编译链,工作量最大,一般不必走到这一步。

> 排查小结:段错误无 traceback 时,"二分法插桩 + `flush=True` 打印 + 最小命令复现"是定位崩溃点最有效的手段。一旦锁定在某个 `import`,优先怀疑该扩展的 ABI / 版本匹配,而非业务代码。

# 调参
```
ubuntu@autodl-container-khhcc3aa8g-7a8a9f02:/home/slxy/zca/code/drivetransformer_private$ nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -l 1nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 17 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 8 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 27 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 5 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 3 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 73 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 53 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 62 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 55 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 45 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 43 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 44 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 0 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 10 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 13 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 58 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 44 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 42 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 46 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 55 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 42 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 61 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 0 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 3 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 0 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 13 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 12 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 56 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 93 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 17 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 71 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 74 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 84 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 80 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 78 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 1 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 3 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 19 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 46 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 75 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 35 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 26 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 39 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 43 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 52 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 53 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 41 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 53 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 53 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 57 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 59 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 89 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 82 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 81 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 0 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 12 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 0 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 44 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 53 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 44 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 44 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 50 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 16 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 57 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 33 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 33 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 47 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 24 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 65 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 39 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 54 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 42 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 11 %
0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 63 %
1, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 %
2, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 %
3, NVIDIA RTX PRO 6000 Blackwell Server Edition, 28741 MiB, 97887 MiB, 100 % ```

## 结论：偏低，建议上调

从上面的连续采样看，当前利用率**偏低，有明显提升空间**。

判断依据：

1. **利用率波动大、频繁掉到 0–20%**。多次出现 `0%`、`3%`、`10%`、`13%` 的采样点，且 4 张卡很不同步（常出现一张 100%、另一张 0% 的情况）。这是典型的 **GPU 在等数据/等同步**（dataloader、CPU 预处理瓶颈，或分布式 all-reduce 等待），而非算力被打满。
2. **显存只用了约 29%**：`28741 / 97887 MiB`。Blackwell 96G 的卡只用了不到 30G，batch size 还有很大余量。
3. 真正打满（80–100%）的时段只是间歇出现，平均大致落在 45–55%，并被频繁的低谷拉低。

建议上调方向（按性价比排序）：

- **加大 batch size（`samples_per_gpu`）**：显存富余很多，这是最直接的手段。注意按比例同步调整学习率。
- **增加 `workers_per_gpu` / 开 `persistent_workers` + `pin_memory`**：掉到 0% 多半是数据供给跟不上；Bench2Drive 的 6 路相机数据预处理很吃 CPU。需与前文的内存排查权衡：worker 进程越多内存占用越高，调到既不 OOM 又能喂满 GPU 为宜。
- **尽量离线 cache 预处理产物**：能提前算好的（如 `preprocess_bench2drive_drivetransformer.py` 的输出）不要在线做。
- 如使用 `prefetch_factor`，可适当调高。
```
