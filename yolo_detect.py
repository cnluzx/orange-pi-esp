import argparse
import time
import os
import sys
import numpy as np
from pathlib import Path

# OpenCV 自带的 Qt 插件目录不包含字体，显式使用系统字体目录。
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts")

import cv2

class YoloDetector:
    # last 1_6.pt 的训练类别名称按字符串排序保存在模型中。RKNN输出只
    # 保留类别索引，因此必须使用同一顺序还原业务ID，不能简单 index + 1。
    CLASS_INDEX_TO_BUSINESS_ID = (
        1, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 2, 20, 3, 4, 5, 6, 7, 8, 9,
    )

    def __init__(self, model_path=None, preset_classes=(1, 11), img_size=(640, 640),
                 show_window=False, save_debug=False):
        """
        初始化 RKNN 侦查视觉系统 (专为香橙派 5 Pro / RKNN3588 设计)
        :param model_path: .rknn 格式的模型路径
        :param preset_classes: 需要关注的类别 ID
        :param img_size: 模型要求的输入图像分辨率 (H, W)，常见为 (640, 640) 或 (320, 320)
        :param show_window: 是否显示检测窗口（生产环境建议False）
        :param save_debug: 是否保存调试图片
        """
        if model_path is None:
            model_path = str(
                Path(__file__).resolve().with_name("last 1_6_fp16_rk3588.rknn")
            )
        self.preset_classes = list(preset_classes)
        self.img_size = img_size
        self.show_window = show_window
        self.save_debug = save_debug
        self.debug_count = 0
        self.last_key = -1
        self.rknn = None
        self.pt_model = None
        self.model_path = str(Path(model_path).expanduser().resolve())

        if self.model_path.lower().endswith(".pt"):
            try:
                from ultralytics import YOLO

                self.pt_model = YOLO(self.model_path)
                print(f"[Vision] 成功加载 Ultralytics 模型: {self.model_path}")
            except Exception as e:
                print(f"[Vision Error] 加载 PT 模型失败: {e}")
            if self.save_debug:
                os.makedirs("logs/yolo_debug", exist_ok=True)
            return

        # ---- 初始化 RKNN NPU 运行时 ----
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError:
            RKNNLite = None

        if RKNNLite is not None and os.path.exists(self.model_path):
            try:
                self.rknn = RKNNLite()
                # 1. 加载 RKNN 模型到内存
                ret = self.rknn.load_rknn(self.model_path)
                if ret != 0:
                    print(f"[Vision Error] 加载 RKNN 模型失败，错误码: {ret}")
                    self.rknn = None
                else:
                    # 2. 初始化 NPU 硬件运行时环境
                    ret = self.rknn.init_runtime()
                    if ret != 0:
                        print(f"[Vision Error] 初始化 NPU 运行时失败，错误码: {ret}")
                        self.rknn = None
                    else:
                        print(f"[Vision] 成功加载 RKNN 硬件加速模型: {self.model_path}")
            except Exception as e:
                print(f"[Vision WARN] NPU 初始化发生异常: {e}，将自动切换至模拟检测模式。")
                self.rknn = None
        else:
            if RKNNLite is None:
                print("[Vision Info] 未检测到 rknnlite 运行库，当前运行在 [模拟检测模式] 下。")
            else:
                print(f"[Vision Info] 未找到 RKNN 权重文件 {self.model_path}，当前运行在 [模拟检测模式] 下。")
        
        # 创建调试目录
        if self.save_debug:
            os.makedirs("logs/yolo_debug", exist_ok=True)

    def detect_target(self, frame):
        if frame is None or frame.size == 0:
            print("[Vision WARN] 传入的画面为空，无法检测")
            return -1

        if self.pt_model is not None:
            try:
                results = self.pt_model.predict(
                    source=frame,
                    imgsz=self.img_size[0],
                    conf=0.5,
                    verbose=False,
                )
                result = results[0]
                detected_ids = []
                if result.boxes is not None:
                    for class_index in result.boxes.cls.cpu().tolist():
                        class_name = str(result.names[int(class_index)])
                        try:
                            detected_ids.append(int(class_name))
                        except ValueError:
                            print(f"[Vision WARN] 非数字类别名称: {class_name}")

                display_frame = result.plot()
                if self.show_window:
                    cv2.imshow("YOLO Target Detection", display_frame)
                    self.last_key = cv2.waitKey(1) & 0xFF
                if self.save_debug:
                    self.debug_count += 1
                    output_dir = Path("logs/yolo_debug")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(output_dir / f"pt_{int(time.time())}_{self.debug_count}.jpg"),
                        display_frame,
                    )

                for business_id in detected_ids:
                    if business_id in self.preset_classes:
                        print(f"[Vision] PT 检测命中目标，业务 ID: {business_id}")
                        return business_id
                return -1
            except Exception as e:
                print(f"[Vision ERROR] PT 推理发生错误: {e}")
                return -1

        if self.rknn is not None:
            try:
                # 与 Ultralytics 保持一致：等比例缩放后以114填充成方形。
                input_h, input_w = self.img_size
                frame_h, frame_w = frame.shape[:2]
                scale = min(input_w / frame_w, input_h / frame_h)
                resized_w = int(round(frame_w * scale))
                resized_h = int(round(frame_h * scale))
                resized = cv2.resize(frame, (resized_w, resized_h))
                pad_left = (input_w - resized_w) // 2
                pad_top = (input_h - resized_h) // 2
                input_bgr = np.full(
                    (input_h, input_w, 3), 114, dtype=np.uint8
                )
                input_bgr[
                    pad_top:pad_top + resized_h,
                    pad_left:pad_left + resized_w,
                ] = resized
                
                # 2. 转换为 RGB
                input_data = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)
                
                # 3. 【关键】确保输入是4维
                # 检查当前维度
                if len(input_data.shape) == 3:
                    # 添加 batch 维度: (H, W, C) -> (1, H, W, C)
                    input_data = np.expand_dims(input_data, axis=0)
                
                # 4. 【可选】检查模型期望的输入格式
                # 如果模型需要 NCHW (通道在前)，转换格式
                # 大多数 RKNN 模型使用 NHWC (通道在后)
                # 如果您不确定，可以尝试两种格式
                
                # 尝试 NHWC 格式 (默认)
                # input_data 已经是 (1, H, W, C)
                
                # 如果模型需要 NCHW，取消下面的注释
                # input_data = np.transpose(input_data, (0, 3, 1, 2))
                
                # 5. 送入 NPU 执行推理
                outputs = self.rknn.inference(inputs=[input_data])
                
                # 6. 后处理
                class_id = self.post_process(
                    outputs,
                    draw_frame=frame
                    if self.show_window or self.save_debug else None,
                    scale=scale,
                    pad_left=pad_left,
                    pad_top=pad_top,
                )
                
                # 7. 显示/保存
                if self.show_window:
                    cv2.imshow("YOLO NPU Detection View", frame)
                    self.last_key = cv2.waitKey(1) & 0xFF
                
                if self.save_debug:
                    self.debug_count += 1
                    timestamp = int(time.time())
                    filename = f"logs/yolo_debug/detect_{timestamp}_{self.debug_count}.jpg"
                    cv2.imwrite(filename, frame)
                    if class_id != -1:
                        cv2.putText(frame, f"Target: {class_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imwrite(filename.replace('.jpg', '_result.jpg'), frame)
                
                if class_id in self.preset_classes:
                    print(f"[Vision] NPU 检测命中目标，类别 ID: {class_id}")
                    return class_id
                
                return -1
                    
            except Exception as e:
                print(f"[Vision ERROR] NPU 推理发生错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 【重要】发生错误时，尝试重置摄像头
                # 但不要在这里重置，应该由上层处理
                return -1

        # ---- 情况 2：未加载 NPU 模型时的模拟模式 ----
        else:
            # 缩放到合适大小展示
            resized_bgr = cv2.resize(frame, self.img_size)
            h, w, _ = resized_bgr.shape
            
            # 在模拟模式下，直接返回预设的第一个类别ID（用于测试流程）
            # 实际使用时，这里应该替换为真实的模拟逻辑
            simulated_result = 1  # 假设检测到类别1
            
            if self.show_window:
                cv2.putText(resized_bgr, "Mock NPU Detecting...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(resized_bgr, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 0, 255), 2)
                cv2.putText(resized_bgr, f"Mock Target ID: {simulated_result}", (w // 4, h // 4 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("YOLO Mock Detection View", resized_bgr)
                self.last_key = cv2.waitKey(1) & 0xFF
            
            if self.save_debug:
                self.debug_count += 1
                timestamp = int(time.time())
                filename = f"logs/yolo_debug/mock_{timestamp}_{self.debug_count}.jpg"
                cv2.imwrite(filename, resized_bgr)
            
            print(f"[Vision Mock] 模拟检测完成，返回假定目标: {simulated_result}")
            
            if simulated_result in self.preset_classes:
                return simulated_result
            return -1

    def post_process(
        self, outputs, draw_frame=None, conf_threshold=0.5,
        nms_threshold=0.45, scale=1.0, pad_left=0, pad_top=0,
    ):
        """
        通用的 NumPy 向量化后处理（并可选在 draw_frame 上直接绘制出检测框）
        """
        if not outputs or len(outputs) == 0:
            return -1

        try:
            # 提取第一个输出张量
            output = outputs[0]
            output = np.squeeze(output)  # 降维 -> (channels, anchors)
            
            # 保证形状为 (num_anchors, 4 + num_classes)
            if output.shape[0] < output.shape[1]:
                output = np.transpose(output)
            
            # 拆分边界框坐标和分类置信度
            boxes = output[:, :4]
            scores = output[:, 4:]
            
            # 1. 向量化计算最大类别和得分
            class_ids = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)
            
            # 2. 粗筛低置信度数据
            mask = confidences > conf_threshold
            if not np.any(mask):
                return -1
            
            valid_boxes = boxes[mask]
            valid_confs = confidences[mask]
            valid_class_ids = class_ids[mask]
            
            # 3. 将 (center_x, center_y, w, h) 转换为 (left_x, top_y, w, h)
            x_left = valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0
            y_top = valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0
            widths = valid_boxes[:, 2]
            heights = valid_boxes[:, 3]
            
            nms_boxes = np.column_stack((x_left, y_top, widths, heights)).tolist()
            nms_scores = valid_confs.tolist()
            
            # 4. 调用 OpenCV NMS 过滤重叠框
            indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, nms_threshold)
            
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                
                # 【画框绘制逻辑】
                if draw_frame is not None:
                    for idx in indices:
                        # YOLO 张量使用 0-based 类别索引；项目其余模块和
                        # RESULT_MAP 使用 1-based 业务 ID。
                        cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[
                            int(valid_class_ids[idx])
                        ]
                        conf = float(valid_confs[idx])
                        
                        # 只绘制我们在 preset_classes 里注册的目标
                        if cls_id in self.preset_classes:
                            x = int((x_left[idx] - pad_left) / scale)
                            y = int((y_top[idx] - pad_top) / scale)
                            w = int(widths[idx] / scale)
                            h = int(heights[idx] / scale)
                            x = max(0, x)
                            y = max(0, y)
                            w = min(w, draw_frame.shape[1] - x)
                            h = min(h, draw_frame.shape[0] - y)
                            
                            # 绘制绿色的矩形检测框
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # 标注类别 ID 颜色和置信度数值
                            label = f"ID: {cls_id} ({conf:.2f})"
                            cv2.putText(draw_frame, label, (x, y - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 优先返回预设类别中的目标
                for idx in indices:
                    cls_id = self.CLASS_INDEX_TO_BUSINESS_ID[
                        int(valid_class_ids[idx])
                    ]
                    if cls_id in self.preset_classes:
                        return cls_id
                
                # 如果没有预设类别，返回第一个检测到的
                best_idx = indices[0]
                detected_class = self.CLASS_INDEX_TO_BUSINESS_ID[
                    int(valid_class_ids[best_idx])
                ]
                return detected_class

        except Exception as e:
            print(f"[Vision PostProcess Error] 后处理解析异常: {e}")
            print("提示：如果遇到维度不匹配错误，可能与您的模型导出/转换参数有关。")
            import traceback
            traceback.print_exc()
        
        return -1
    
    def release(self):
        """释放RKNN资源"""
        if self.rknn is not None:
            try:
                self.rknn.release()
                print("[Vision] RKNN资源已释放")
            except:
                pass
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()


def _parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def _ensure_pt_runtime(model_path):
    """直接运行脚本时，为 PT 模型切换到项目虚拟环境。"""
    resolved = Path(model_path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        return
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass

    venv_python = resolved.parent.parent / "myenv" / "bin" / "python"
    if not venv_python.is_file():
        raise RuntimeError(
            f"当前 Python 未安装 ultralytics，且找不到虚拟环境: {venv_python}"
        )
    print(f"[TEST] 切换到项目虚拟环境: {venv_python}", flush=True)
    os.execv(
        str(venv_python),
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


def main():
    parser = argparse.ArgumentParser(description="RKNN YOLO 目标识别测试")
    parser.add_argument("--camera", default="2", help="摄像头编号或设备路径")
    parser.add_argument("--model", default=None, help="RKNN 模型路径")
    parser.add_argument(
        "--frames", type=int, default=0,
        help="最多测试帧数，0 表示持续运行到按 Q 或 Esc",
    )
    parser.add_argument(
        "--classes",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20",
        help="关注的业务类别ID",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="不显示检测窗口"
    )
    parser.add_argument("--save-debug", action="store_true", help="保存调试图片")
    args = parser.parse_args()

    selected_model = args.model or str(
        Path(__file__).resolve().with_name("last 1_6_fp16_rk3588.rknn")
    )
    _ensure_pt_runtime(selected_model)

    preset_classes = tuple(
        int(value.strip()) for value in args.classes.split(",") if value.strip()
    )
    detector = YoloDetector(
        model_path=selected_model,
        preset_classes=preset_classes,
        show_window=not args.no_show,
        save_debug=args.save_debug,
    )
    if detector.rknn is None and detector.pt_model is None:
        detector.release()
        raise RuntimeError("RKNN 模型未加载，拒绝使用模拟结果进行模型测试")

    source = _parse_camera(args.camera)
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    camera = cv2.VideoCapture(source, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        detector.release()
        raise RuntimeError(f"无法打开摄像头: {source}")

    interrupted = False
    try:
        frame_index = 0
        while args.frames <= 0 or frame_index < args.frames:
            frame_index += 1
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                print("[TEST WARN] 摄像头取帧失败")
                continue
            result = detector.detect_target(frame)
            if result != -1 or frame_index == 1 or frame_index % 30 == 0:
                print(f"[TEST] frame={frame_index} result={result}")
            if detector.last_key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        camera.release()
        detector.release()
    if interrupted:
        print("[TEST] 用户中止")
    else:
        print("[TEST] 测试结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
