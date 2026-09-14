# Project4 端到端模型训练与闭环推理

## 1. 项目目标

本项目主要聚焦于完成 DriveTransformer 端到端自动驾驶模型的训练与闭环评测，主要包含以下几个部分：

1. Bench2Drive 数据集准备：下载 Bench2Drive 数据集并整理为 DriveTransformer 训练需要的格式；
2. 端到端模型的训练：根据个人算力配置修改模型并进行模型训练；
3. 进行闭环评测：完成所训练模型的闭环评测，查看性能指标。

## 2. 操作步骤

### 2.1 下载 Bench2Drive 数据集

推荐下载 Base-400G 版本（如果资源受限使用 Mini-4G 版本体验训练流程也可以）

https://github.com/Thinklab-SJTU/Bench2Drive#Dataset

### 2.2 整理数据格式为以下的目录结构

```text
		Bench2DriveZoo
		├── ...
		├── data/
		|   ├── bench2drive/
		|   |   ├── v1/                                          # Bench2Drive base
		|   |   |   ├── Accident_Town03_Route101_Weather23/
		|   |   |   ├── Accident_Town03_Route102_Weather20/
		|   |   |   └── ...
		|   |   └── maps/                                        # maps of Towns
		|   |       ├── Town01_HD_map.npz
		|   |       ├── Town02_HD_map.npz
		|   |       └── ...
		|   └── splits
		|           └── bench2drive_base_train_val_split.json    # trainval_split of Bench2Drive base
```

其中 Bench2DriveZoo 为 DriverTransformer 根目录，其余文件可以通过软链接的形式组成上述的目录结构，
`bench2drive_base_train_val_split.json` 为训练和验证集的切分文件，可以自行切分，也可以将所有数据用于训练，
则切分文件的内容如下：

```text
{
	"val": [
	]
}
```

### 2.3 数据集预处理

运行以下脚本（需要运行一段时间），注意修改其中的路径为自己的实际路径：

```text
cd /home/slxy/zca/code/drivetransformer_private/adzoo/drivetransformer/mmdet3d_plugin/datasets
python preprocess_bench2drive_drivetransformer.py --workers 16
```

### 2.4 训练模型

准备好数据以后即可通过以下指令启动训练：

```text
bash adzoo/drivetransformer/dist_train.sh adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py 8 #N_GPUS
```

这里需要根据你的设备情况修改 GPU 数量，此外 large 模型需要的显存较大，可以尝试通过修改配置文件中的
batchsize、网络层数、特征维度等方式降低显存占用。在调小模型以及 batchsize 后，可能会影响模型的训练效果，
大家能够正常训练得到自己的权重即可。

### 2.5 闭环评测

在 Bench2Drive 上进行闭环评测（修改评测脚本的 `--routes` 参数为 `bench2drive220.xml`），如果资源有限，
也可以在 `drivetransformer_bench2drive_dev10.xml` 上进行测评，评测结果在评测脚本中 `--checkpoint`
参数对应的文件中。

### 2.6 demo 展示

生成视频 demo 可以参考 `Bench2Drive/tools/generate_video.py`。以下 demo 所用模型是八卡 A800（单卡显存 80G），
使用 Bench2Drive base 数据集，训练 30 个 epoch 的可视化效果。算力和数据不足可能达不到很好的可视化结果，
能够跑通流程，理解模型训练和闭环评测即达到了学习目的。

![闭环评测 demo：八卡 A800、Bench2Drive base 数据集训练 30 epoch 的可视化效果](assets/lab4/demo_closed_loop.gif)

## 3. 作业提交说明

### 3.1 提交内容

1. 数据集目录结构截图；
2. 训练过程中打印 loss 截图；
3. 自己训练的模型配置文件；
4. 闭环推理结果以及部分可视化结果，分析自己所训模型的表现。

### 3.2 命名规则

压缩包命名：`用户名-Project4.zip`，压缩包大小建议小于 100M。
