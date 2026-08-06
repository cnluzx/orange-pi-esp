# DeepLabV3 Dice 0.7 模型说明

## 1. 模型概述

这是一个面向香橙派 5（Rockchip RK3588 NPU）的二分类语义分割模型，用于识别图像中的白色地面区域。

- 网络：DeepLabV3-MobileNetV2
- Backbone：MobileNetV2
- Output stride：16
- 类别数：2
  - `0`：背景
  - `1`：white floor
- 训练输入：`640x480`
- ONNX 输入：`[1, 3, 480, 640]`
- native ONNX/RKNN 输出：`[1, 2, 30, 40]`
- 目标硬件：香橙派 5 / RK3588 NPU
- 推理精度：RKNN FP16

模型不在 NPU 内部执行最终的 640x480 放大。板端拿到 30x40 logits 后，在 CPU 端进行 logits 放大和掩码后处理，这样可以降低 NPU 的内存带宽压力。

## 2. 训练配置

- 数据集：`prepared_dataset_latest`
- 数据量：train 414、valid 119、test 60
- 原始图像尺寸：640x480，训练过程中保持该比例和尺寸
- 训练迭代：2400 iterations
- Batch size：4
- 优化器：SGD，初始学习率 0.01，poly 学习率策略
- 数据增强：随机水平翻转
- AMP：启用 CUDA 混合精度
- 随机种子：1
- 损失函数：

```text
Loss = CrossEntropyLoss + 0.7 x ForegroundDiceLoss
```

Dice 项只针对 white floor 前景类别计算。

## 3. 测试表现

在 latest_dataset 的 60 张 test 图片上，native ONNX 使用 logits 双线性放大后再 argmax，结果如下：

| 指标 | Dice 0.7 |
|---|---:|
| Pixel Accuracy | 0.994071 |
| Background IoU | 0.988970 |
| White floor IoU | 0.987343 |
| Mean IoU | 0.988156 |
| White floor Dice | 0.993631 |

与 Dice 0.5 模型相比：

- Mean IoU：`0.982511 -> 0.988156`
- White floor IoU：`0.981330 -> 0.987343`
- White floor Dice：`0.990577 -> 0.993631`

## 4. 表现良好的原因

### 4.1 损失函数更关注前景完整性

白色地面通常是图像中的主要目标，但边缘、薄区域和局部缺口容易被像素级 CE 损失忽略。Dice 损失直接比较预测区域与标注区域的重叠程度，可以减少：

- 地面区域内部的小孔洞
- 边缘断裂
- 前景区域被侵蚀
- 类别不平衡导致的前景召回不足

将 Dice 权重从 0.5 提高到 0.7 后，模型对前景区域重叠的约束更强，因此白色地面 IoU 和 Dice 均有提升。

### 4.2 数据集与任务定义匹配

模型只处理“背景/白色地面”两个类别，任务复杂度低于多类别分割。数据集中的标注目标与实际部署目标一致，模型可以集中学习地面颜色、纹理、边界和透视形状。

### 4.3 MobileNetV2 适合边缘设备

MobileNetV2 参数量和计算量较低，DeepLabV3 的 ASPP 模块保留了较好的上下文建模能力，在精度和 RK3588 NPU 推理速度之间取得了平衡。

### 4.4 native 30x40 输出降低部署开销

模型只输出 30x40 的低分辨率 logits，避免在 NPU 内部执行大尺寸 Resize。最终放大和形态学处理放在 CPU 端完成，可减少大特征图读写，对香橙派 5 的实时性能更有利。

### 4.5 保持 640x480 输入比例

训练和部署输入比例一致，避免额外裁剪或拉伸造成的几何变形，有利于保持道路边界和底部轮廓的稳定性。

## 5. 部署建议

推荐使用以下文件：

```text
deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16.rknn
```

板端流程：

```text
摄像头图像 640x480
    -> RGB/归一化预处理
    -> RKNN FP16 NPU 推理
    -> 输出 2x30x40 logits
    -> CPU 双线性放大 logits 到 640x480
    -> argmax 得到二值掩码
    -> 可选 Close/Open、最大连通域和底部连续性处理
```

不要直接把 30x40 的类别索引使用最近邻放大作为唯一方案。优先放大两类 logits 后再 argmax，边界通常更平滑。

## 6. 文件位置

- Checkpoint：`../../checkpoints/latest_dataset_deeplabv3_640x480_dice07/best_deeplabv3_mobilenet_roboflow_voc_os16.pth`
- ONNX：`deeplabv3_mobilenet_roboflow_voc_640x480_native.onnx`
- RKNN FP16：`deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16.rknn`
- 训练脚本：`../../main.py`

## 7. 使用限制

当前指标是在已有 test split 上得到的，不能直接等同于所有道路、光照和摄像头条件下的实际效果。部署前仍建议补充以下 hard-negative 样本：

- 白墙、白色车辆和白色物体
- 强反光或过曝地面
- 非白色路面
- 无道路、路口和急转弯场景
- 雨天、阴天和夜间图像

另外，若 train/valid/test 来自连续视频帧，指标可能偏高。最终应使用独立视频片段或真实摄像头数据在香橙派 5 上验证精度、FPS 和内存占用。
