import time
import cv2
import os
import numpy as np

# 尝试导入板端的 RKNN-Lite 运行时
try:
    from rknnlite.api import RKNNLite  # type: ignore
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

class YoloDetector:
    def __init__(self, model_path="last_1_6.rknn", preset_classes=(1, 11), img_size=(640, 640),
                 show_window=False, save_debug=False):
        """
        初始化 RKNN 侦查视觉系统 (专为香橙派 5 Pro / RKNN3588 设计)
        :param model_path: .rknn 格式的模型路径
        :param preset_classes: 需要关注的类别 ID
        :param img_size: 模型要求的输入图像分辨率 (H, W)，常见为 (640, 640) 或 (320, 320)
        :param show_window: 是否显示检测窗口（生产环境建议False）
        :param save_debug: 是否保存调试图片
        """
        self.preset_classes = list(preset_classes)
        self.img_size = img_size
        self.show_window = show_window
        self.save_debug = save_debug
        self.debug_count = 0
        self.rknn = None
        
        # 兼容性处理：如果传入的是 .pt 路径且存在对应的 .rknn，则自动替换后缀
        if model_path.endswith(".pt"):
            model_path = model_path.replace(".pt", ".rknn")

        # ---- 初始化 RKNN NPU 运行时 ----
        if HAS_RKNN and os.path.exists(model_path):
            try:
                self.rknn = RKNNLite()
                # 1. 加载 RKNN 模型到内存
                ret = self.rknn.load_rknn(model_path)
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
                        print(f"[Vision] 成功加载 RKNN 硬件加速模型: {model_path}")
            except Exception as e:
                print(f"[Vision WARN] NPU 初始化发生异常: {e}，将自动切换至模拟检测模式。")
                self.rknn = None
        else:
            if not HAS_RKNN:
                print("[Vision Info] 未检测到 rknnlite 运行库，当前运行在 [模拟检测模式] 下。")
            else:
                print(f"[Vision Info] 未找到 RKNN 权重文件 {model_path}，当前运行在 [模拟检测模式] 下。")
        
        # 创建调试目录
        if self.save_debug:
            os.makedirs("logs/yolo_debug", exist_ok=True)

    def detect_target(self, frame):
        if frame is None or frame.size == 0:
            print("[Vision WARN] 传入的画面为空，无法检测")
            return -1

        if self.rknn is not None:
            try:
                # 1. 预处理：缩放
                resized_bgr = cv2.resize(frame, self.img_size)
                
                # 2. 转换为 RGB
                input_data = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
                
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
                class_id = self.post_process(outputs, draw_frame=resized_bgr if self.show_window else None)
                
                # 7. 显示/保存
                if self.show_window:
                    cv2.imshow("YOLO NPU Detection View", resized_bgr)
                    cv2.waitKey(1)
                
                if self.save_debug:
                    self.debug_count += 1
                    timestamp = int(time.time())
                    filename = f"logs/yolo_debug/detect_{timestamp}_{self.debug_count}.jpg"
                    cv2.imwrite(filename, resized_bgr)
                    if class_id != -1:
                        cv2.putText(resized_bgr, f"Target: {class_id}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imwrite(filename.replace('.jpg', '_result.jpg'), resized_bgr)
                
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
                cv2.waitKey(1)  # 改为1ms
            
            if self.save_debug:
                self.debug_count += 1
                timestamp = int(time.time())
                filename = f"logs/yolo_debug/mock_{timestamp}_{self.debug_count}.jpg"
                cv2.imwrite(filename, resized_bgr)
            
            print(f"[Vision Mock] 模拟检测完成，返回假定目标: {simulated_result}")
            
            if simulated_result in self.preset_classes:
                return simulated_result
            return -1

    def post_process(self, outputs, draw_frame=None, conf_threshold=0.5, nms_threshold=0.45):
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
                        cls_id = int(valid_class_ids[idx])
                        conf = float(valid_confs[idx])
                        
                        # 只绘制我们在 preset_classes 里注册的目标
                        if cls_id in self.preset_classes:
                            x = int(x_left[idx])
                            y = int(y_top[idx])
                            w = int(widths[idx])
                            h = int(heights[idx])
                            
                            # 绘制绿色的矩形检测框
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # 标注类别 ID 颜色和置信度数值
                            label = f"ID: {cls_id} ({conf:.2f})"
                            cv2.putText(draw_frame, label, (x, y - 8), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 优先返回预设类别中的目标
                for idx in indices:
                    cls_id = int(valid_class_ids[idx])
                    if cls_id in self.preset_classes:
                        return cls_id
                
                # 如果没有预设类别，返回第一个检测到的
                best_idx = indices[0]
                detected_class = int(valid_class_ids[best_idx])
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