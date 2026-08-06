# latest_dataset Dice 0.7 实验结果

## 配置

- 模型：DeepLabV3-MobileNetV2，output stride 16
- 输入：`1x3x480x640`
- native 输出：`1x2x30x40`
- 数据集：`prepared_dataset_latest`（train 414、valid 119、test 60）
- 训练：2400 iterations，batch size 4，AMP，随机种子 1
- 损失：`CrossEntropy + 0.7 x foreground Dice`

## Test 对比

| 实验 | 背景 IoU | white floor IoU | mIoU | white floor Dice |
|---|---:|---:|---:|---:|
| Dice 0.5 | 0.983693 | 0.981330 | 0.982511 | 0.990577 |
| Dice 0.7 | 0.988970 | 0.987343 | 0.988156 | 0.993631 |

Dice 0.7 相比 Dice 0.5：mIoU 提升约 0.565 个百分点，white floor IoU 提升约 0.601 个百分点。

## 文件

- Checkpoint：`../../checkpoints/latest_dataset_deeplabv3_640x480_dice07/best_deeplabv3_mobilenet_roboflow_voc_os16.pth`
- ONNX：`deeplabv3_mobilenet_roboflow_voc_640x480_native.onnx`
- RKNN FP16：`deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16.rknn`

RKNN FP16 使用 RKNN Toolkit2 2.3.2 转换，目标平台 RK3588；转换过程成功完成。实际 FPS 仍需在香橙派 5 上用板端 runtime 测量。
