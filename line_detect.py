"""Use a DeepLabV3 RKNN road mask for steering and crossroads detection."""

import argparse
import os
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from rknnlite.api import RKNNLite


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "test_code"
    / "deeplabv3_mobilenet_roboflow_voc_640x480_native_fp16(2).rknn"
)


class LineTracker:
    """Semantic road tracker with the original ``get_road_info`` interface."""

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        input_width: int = 640,
        input_height: int = 480,
        threshold: float = 0.6,
        road_class: int = 1,
        debug_mode: bool = True,
    ):
        self.roi_top_limit = 0.2
        self.roi_left_limit = 0.0
        self.roi_right_limit = 1.0

        self.input_width = int(os.environ.get("DEEPLAB_INPUT_WIDTH", input_width))
        self.input_height = int(os.environ.get("DEEPLAB_INPUT_HEIGHT", input_height))
        self.threshold = float(os.environ.get("DEEPLAB_THRESHOLD", threshold))
        self.road_class = int(os.environ.get("DEEPLAB_ROAD_CLASS", road_class))
        self.debug_mode = debug_mode and os.environ.get(
            "LINE_DEBUG", "1"
        ).lower() not in {"0", "false", "off", "no"}
        self.debug_frame_interval = max(
            1, int(os.environ.get("LINE_DEBUG_FRAME_INTERVAL", "3"))
        )
        self._debug_frame_count = 0

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("DeepLab threshold must be between 0 and 1")

        configured_model = model_path or os.environ.get("DEEPLAB_MODEL") or DEFAULT_MODEL
        self.model_path = Path(configured_model).expanduser().resolve()
        self.rknn: RKNNLite | None = self._load_runtime(self.model_path)
        self.output_shape_printed = False

        # Road-centre memory and smoothing. Coordinates are relative to the ROI.
        self.last_cx_roi = None
        self.last_offset = 0
        self.lost_frames = 0
        self.max_lost = 5
        self.use_memory = True
        self.center_smooth_alpha = 0.35
        self.max_center_jump_ratio = 0.08

        # Reject semantic blobs which are not connected to the near-field road.
        self.min_component_area = 300
        self.min_bottom_support_ratio = 0.02

        # Preserve the existing crossroads output used by main.py.
        self.cross_counter = 0
        self.cross_confirm_threshold = 2
        self.cross_width_ratio = 0.82
        self.cross_min_wide_rows = 4
        # Camera-0 measurements at N2 -> I1_1 are stable at 72.5% of ROI width.
        self.l_cross_width_ratio = float(
            os.environ.get("L_CROSS_WIDTH_RATIO", "0.69")
        )
        self.l_cross_min_wide_rows = max(
            1, int(os.environ.get("L_CROSS_MIN_WIDE_ROWS", "2"))
        )
        self.junction_mode = "NORMAL"
        self.last_key = 0

        self.last_inference_ms = 0.0
        self.last_foreground_percent = 0.0

        print(
            "[Tracker] DeepLabV3 semantic road tracker loaded | "
            f"model={self.model_path}"
        )
        print(
            f"[Tracker] input=RGB NHWC 1x{self.input_height}x"
            f"{self.input_width}x3 | road_class={self.road_class} | "
            f"threshold={self.threshold:.2f}"
        )

    @staticmethod
    def _load_runtime(model_path: Path) -> RKNNLite:
        if not model_path.is_file():
            raise FileNotFoundError(f"DeepLabV3 RKNN model not found: {model_path}")

        rknn = cast(RKNNLite, RKNNLite(verbose=False))
        if rknn.load_rknn(str(model_path)) != 0:
            rknn.release()
            raise RuntimeError(f"failed to load RKNN model: {model_path}")

        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", RKNNLite.NPU_CORE_AUTO)
        if rknn.init_runtime(core_mask=core_mask) != 0:
            rknn.release()
            raise RuntimeError("failed to initialize RKNN runtime")
        return rknn

    def _prepare_input(self, frame_bgr):
        resized = cv2.resize(
            frame_bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)

    def _decode_output(self, output):
        """Decode the same NCHW/NHWC output formats as test_deeplabv3.py."""
        array = np.asarray(output)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise RuntimeError(f"only batch=1 is supported, got {array.shape}")
            array = array[0]

        if array.ndim == 2:
            return array.astype(np.uint8), None
        if array.ndim != 3:
            raise RuntimeError(f"unsupported output shape: {array.shape}")

        if array.shape[0] <= 64:
            logits = array.astype(np.float32, copy=False)
            if logits.shape[0] == 2:
                delta = np.clip(logits[1] - logits[0], -30.0, 30.0)
                probability = 1.0 / (1.0 + np.exp(-delta))
                return (probability >= self.threshold).astype(np.uint8), probability
            return np.argmax(logits, axis=0).astype(np.uint8), None

        if array.shape[-1] <= 64:
            logits = array.astype(np.float32, copy=False)
            if logits.shape[-1] == 2:
                delta = np.clip(logits[..., 1] - logits[..., 0], -30.0, 30.0)
                probability = 1.0 / (1.0 + np.exp(-delta))
                return (probability >= self.threshold).astype(np.uint8), probability
            return np.argmax(logits, axis=-1).astype(np.uint8), None

        raise RuntimeError(f"cannot determine output layout: {array.shape}")

    def _lost_result(self, status="LINE"):
        self.lost_frames += 1
        if self.lost_frames > self.max_lost:
            self.last_cx_roi = None
        self.cross_counter = 0
        self.last_offset = int(self.last_offset * 0.75)
        return status, self.last_offset if status == "LINE" else 0

    def _select_near_road_component(self, road_mask):
        """Return the main semantic road component connected to the ROI bottom."""
        roi_h, roi_w = road_mask.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            road_mask, connectivity=8
        )
        if count <= 1:
            return None

        bottom_rows = max(4, int(roi_h * 0.10))
        min_bottom_support = max(
            8, int(roi_w * bottom_rows * self.min_bottom_support_ratio)
        )
        best_label = None
        best_score = -1.0

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_area:
                continue

            component_bottom = labels[roi_h - bottom_rows :, :] == label
            bottom_support = int(np.count_nonzero(component_bottom))
            if bottom_support < min_bottom_support:
                continue

            component_x = float(stats[label, cv2.CC_STAT_LEFT]) + (
                float(stats[label, cv2.CC_STAT_WIDTH]) * 0.5
            )
            if self.use_memory and self.last_cx_roi is not None:
                distance_penalty = abs(component_x - self.last_cx_roi) * 2.0
            else:
                distance_penalty = abs(component_x - roi_w * 0.5) * 0.5
            score = area + bottom_support * 4.0 - distance_penalty
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None
        return np.where(labels == best_label, 255, 0).astype(np.uint8)

    def _road_centroid(self, component_mask):
        """Calculate the near-field road centroid in ROI coordinates."""
        roi_h, _ = component_mask.shape
        near_mask = component_mask.copy()
        near_mask[: int(roi_h * 0.30), :] = 0
        moments = cv2.moments(near_mask, binaryImage=True)
        if moments["m00"] <= 0:
            moments = cv2.moments(component_mask, binaryImage=True)
        if moments["m00"] <= 0:
            return None
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    def set_junction_mode(self, mode: str) -> None:
        """Select the expected topology profile without bypassing vision."""
        normalized = str(mode).strip().upper()
        if normalized not in {"NORMAL", "L"}:
            raise ValueError(f"unsupported junction mode: {mode}")
        if normalized != self.junction_mode:
            self.junction_mode = normalized
            self.cross_counter = 0
            print(f"[Tracker] junction mode -> {self.junction_mode}")

    def _is_crossroad(self, component_mask):
        roi_h, roi_w = component_mask.shape
        scan_y_start = int(roi_h * 0.10)
        scan_y_end = int(roi_h * 0.50)
        wide_rows = 0
        if self.junction_mode == "L":
            width_ratio = self.l_cross_width_ratio
            min_wide_rows = self.l_cross_min_wide_rows
        else:
            width_ratio = self.cross_width_ratio
            min_wide_rows = self.cross_min_wide_rows

        for y in range(scan_y_start, scan_y_end, 4):
            is_road = (component_mask[y, :] > 0).astype(np.int8)
            diffs = np.diff(np.r_[0, is_road, 0])
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            if len(starts) and np.max(ends - starts) > int(
                roi_w * width_ratio
            ):
                wide_rows += 1
        return wide_rows >= min_wide_rows

    def _draw_debug(
        self,
        frame,
        component_mask,
        roi_start_row,
        roi_start_col,
        roi_end_col,
        global_cx,
        global_cy,
        offset,
    ):
        height = frame.shape[0]
        overlay = frame.copy()
        roi_overlay = overlay[
            roi_start_row:height, roi_start_col:roi_end_col
        ]
        roi_overlay[component_mask > 0] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, dst=frame)

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            path = max(contours, key=cv2.contourArea)
            path = path + (roi_start_col, roi_start_row)
            cv2.drawContours(frame, [path], -1, (0, 255, 0), 2)

        cv2.rectangle(
            frame,
            (roi_start_col, roi_start_row),
            (roi_end_col - 1, height - 1),
            (0, 0, 255),
            2,
        )
        # Highlight only the horizontal band used by _is_crossroad().
        junction_top = roi_start_row + int(component_mask.shape[0] * 0.10)
        junction_bottom = roi_start_row + int(component_mask.shape[0] * 0.50)
        cv2.line(
            frame,
            (roi_start_col, junction_top),
            (roi_end_col - 1, junction_top),
            (0, 165, 255),
            2,
        )
        cv2.line(
            frame,
            (roi_start_col, junction_bottom),
            (roi_end_col - 1, junction_bottom),
            (0, 165, 255),
            2,
        )
        active_ratio = (
            self.l_cross_width_ratio
            if self.junction_mode == "L"
            else self.cross_width_ratio
        )
        cv2.putText(
            frame,
            f"JUNCTION ROI | mode={self.junction_mode} "
            f"width>={active_ratio:.2f}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
        cv2.line(
            frame,
            (frame.shape[1] // 2, roi_start_row),
            (frame.shape[1] // 2, height - 1),
            (255, 255, 0),
            1,
        )
        cv2.circle(frame, (global_cx, global_cy), 6, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Offset: {offset} | infer: {self.last_inference_ms:.1f} ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Road: {self.last_foreground_percent:.1f}% | "
            f"Lost: {self.lost_frames}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Cross: {self.cross_counter}/{self.cross_confirm_threshold}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
        if self.cross_counter >= self.cross_confirm_threshold:
            cv2.putText(
                frame,
                "CROSS CONFIRMED!",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        try:
            cv2.imshow("Track View", frame)
            self.last_key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            # A headless launch must not interrupt steering just because no GUI
            # backend is available.
            print(f"[Tracker] debug window disabled: {exc}")
            self.debug_mode = False

    def get_road_info(self, frame):
        """Return ``(road_type, horizontal_error_pixels)`` for main.py."""
        if frame is None or frame.size == 0:
            return "WAITING", 0

        height, width = frame.shape[:2]
        center_x = width // 2
        roi_start_row = int(height * self.roi_top_limit)
        roi_start_col = int(width * self.roi_left_limit)
        roi_end_col = int(width * self.roi_right_limit)
        if roi_start_row >= height or roi_start_col >= roi_end_col:
            return "WAITING", 0

        try:
            self._debug_frame_count += 1
            started = time.perf_counter()
            model_input = self._prepare_input(frame)
            # release() 会将成员设为 None；使用局部引用可避免静态类型
            # 歧义，也避免清理线程在本次推理开始前释放运行时。
            rknn = self.rknn
            if rknn is None:
                raise RuntimeError("RKNN runtime has been released")
            outputs = rknn.inference(
                inputs=[model_input], data_format=["nhwc"]
            )
            self.last_inference_ms = (time.perf_counter() - started) * 1000.0
            if not outputs:
                raise RuntimeError("RKNN returned no output")

            if not self.output_shape_printed:
                output = np.asarray(outputs[0])
                print(f"[Tracker] RKNN output: shape={output.shape}, dtype={output.dtype}")
                self.output_shape_printed = True

            class_mask, _ = self._decode_output(outputs[0])
            if class_mask.shape != (height, width):
                class_mask = cv2.resize(
                    class_mask,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            roi_classes = class_mask[
                roi_start_row:height, roi_start_col:roi_end_col
            ]
            road_mask = np.where(
                roi_classes == self.road_class, 255, 0
            ).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            road_mask = cv2.morphologyEx(
                road_mask, cv2.MORPH_OPEN, kernel, iterations=1
            )
            road_mask = cv2.morphologyEx(
                road_mask, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            self.last_foreground_percent = float(np.mean(road_mask > 0) * 100.0)

            component_mask = self._select_near_road_component(road_mask)
            if component_mask is None:
                return self._lost_result("LINE")

            centroid = self._road_centroid(component_mask)
            if centroid is None:
                return self._lost_result("LINE")
            measured_cx, measured_cy = centroid

            if self.last_cx_roi is not None:
                max_jump = max(
                    8, int(component_mask.shape[1] * self.max_center_jump_ratio)
                )
                measured_cx = float(
                    np.clip(
                        measured_cx,
                        self.last_cx_roi - max_jump,
                        self.last_cx_roi + max_jump,
                    )
                )
                measured_cx = (
                    (1.0 - self.center_smooth_alpha) * self.last_cx_roi
                    + self.center_smooth_alpha * measured_cx
                )

            cx_roi = int(round(measured_cx))
            cy_roi = int(round(measured_cy))
            self.last_cx_roi = cx_roi
            self.lost_frames = 0

            global_cx = cx_roi + roi_start_col
            global_cy = cy_roi + roi_start_row
            offset = global_cx - center_x
            self.last_offset = offset

            if self._is_crossroad(component_mask):
                self.cross_counter += 1
            else:
                self.cross_counter = 0

            if self.debug_mode and self._debug_frame_count % self.debug_frame_interval == 0:
                self._draw_debug(
                    frame,
                    component_mask,
                    roi_start_row,
                    roi_start_col,
                    roi_end_col,
                    global_cx,
                    global_cy,
                    offset,
                )

            if self.cross_counter >= self.cross_confirm_threshold:
                return "CROSS", 0
            return "LINE", offset

        except Exception as exc:
            print(f"[Tracker] semantic road processing error: {exc}")
            return self._lost_result("WAITING")

    def release(self):
        """Release the RKNN runtime. Safe to call more than once."""
        rknn, self.rknn = self.rknn, None
        if rknn is not None:
            rknn.release()


def _parse_camera_source(value: object) -> int | str:
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def _parse_flip_code(value: object) -> int | None:
    value = str(value).strip().lower()
    if value in {"none", "off", "false", "no"}:
        return None
    code = int(value)
    if code not in (-1, 0, 1):
        raise ValueError("flip must be -1, 0, 1, or none")
    return code


def run_camera_test(
    camera_source: int | str = 0,
    flip_code: int | None = -1,
    model_path: str | os.PathLike | None = None,
    threshold: float = 0.6,
    road_class: int = 1,
) -> int:
    tracker = LineTracker(
        model_path=model_path,
        threshold=threshold,
        road_class=road_class,
    )
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    try:
        cap = cv2.VideoCapture(camera_source, backend)
    except Exception:
        cap = cv2.VideoCapture(camera_source)

    if not cap.isOpened():
        print(f"[ERROR] cannot open camera: {camera_source}")
        tracker.release()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\n>>> DeepLabV3 road-following test started")
    print(f">>> camera={camera_source} | flip={flip_code}")
    print(">>> press Q or Ctrl+C to exit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] camera frame unavailable", end="\r")
                continue
            if flip_code in (-1, 0, 1):
                frame = cv2.flip(frame, flip_code)

            status, offset = tracker.get_road_info(frame)
            print(
                f"status={status:<7} | offset={offset:<5} | "
                f"cross={tracker.cross_counter:<3} | "
                f"infer={tracker.last_inference_ms:>6.1f} ms",
                end="\r",
            )
            if tracker.last_key in (ord("q"), ord("Q"), 27):
                print("\n[SYS] test stopped")
                return 0
    except KeyboardInterrupt:
        print("\n[SYS] test stopped")
        return 0
    finally:
        cap.release()
        tracker.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DeepLabV3 RKNN semantic road-following test"
    )
    parser.add_argument(
        "--camera",
        default=os.environ.get("NAV_CAMERA", "0"),
        help="camera index or /dev/video path",
    )
    parser.add_argument(
        "--flip",
        default=os.environ.get("NAV_CAMERA_FLIP", "none"),
        help="-1, 0, 1, or none",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPLAB_MODEL", str(DEFAULT_MODEL)),
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--road-class", type=int, default=1)
    args = parser.parse_args()

    try:
        return run_camera_test(
            _parse_camera_source(args.camera),
            _parse_flip_code(args.flip),
            args.model,
            args.threshold,
            args.road_class,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
