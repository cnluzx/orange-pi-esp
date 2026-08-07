#!/usr/bin/env python3
"""高位摄像头涵洞横线由画面底部上移至顶部的检测测试。

跨帧跟踪一条明显长横线，只有它完整经历“画面底部 -> 画面顶部”后，才
向底盘发送急停和涵洞侦查指令 10。该文件不依赖成熟目录中的任何代码，
也可将 ``CulvertLineDetector`` 直接注入主程序预留的 culvert_detector。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_SERIAL = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"


@dataclass(frozen=True)
class CandidateLine:
    x_left: int
    y_left: int
    x_right: int
    y_right: int
    score: float

    @property
    def center_y(self) -> float:
        return (self.y_left + self.y_right) / 2.0


class CulvertLineDetector:
    """跨帧跟踪从画面底部向顶部移动的近水平长线。"""

    def __init__(
        self,
        min_track_frames: int = 3,
        max_missed_frames: int = 5,
        top_ratio: float = 0.35,
        bottom_ratio: float = 0.55,
        min_length_ratio: float = 0.25,
        min_travel_ratio: float = 0.18,
        max_horizontal_angle: float = 22.0,
    ) -> None:
        self.min_track_frames = max(2, min_track_frames)
        self.max_missed_frames = max(0, max_missed_frames)
        self.top_ratio = top_ratio
        self.bottom_ratio = bottom_ratio
        self.min_length_ratio = min_length_ratio
        self.min_travel_ratio = min_travel_ratio
        self.max_horizontal_angle = max_horizontal_angle
        self.last_candidates: list[CandidateLine] = []
        self.last_edges: np.ndarray | None = None
        self.raw_detected = False
        self.tracking = False
        self.start_y = 0.0
        self.last_y = 0.0
        self.best_y = 0.0
        self.track_hits = 0
        self.missed_frames = 0
        self.last_tracked: CandidateLine | None = None

    def reset(self) -> None:
        self.last_candidates = []
        self.raw_detected = False
        self._clear_track()

    def _clear_track(self) -> None:
        self.tracking = False
        self.start_y = 0.0
        self.last_y = 0.0
        self.best_y = 0.0
        self.track_hits = 0
        self.missed_frames = 0
        self.last_tracked = None

    def _extract_candidates(self, frame: np.ndarray) -> list[CandidateLine]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(gray))
        low = int(max(25, 0.66 * median))
        high = int(min(230, max(low + 30, 1.33 * median)))
        edges = cv2.Canny(gray, low, high)

        # 忽略画面最上、最下方的裁切边框，防止固定边框形成假横线。
        margin = max(3, int(height * 0.02))
        edges[:margin, :] = 0
        edges[height - margin :, :] = 0
        self.last_edges = edges

        # 涵洞横线在现场画面中是一条较宽、略模糊但几乎贯穿整幅画面的
        # 亮度边界。用纵向 Sobel 梯度的横向覆盖率补充 Hough，避免 Canny
        # 只抓到局部窗框、却漏掉真正的宽横线。
        projection_gray = cv2.GaussianBlur(gray, (9, 9), 0)
        vertical_gradient = np.abs(
            cv2.Sobel(projection_gray, cv2.CV_32F, 0, 1, ksize=3)
        )
        x_margin = max(4, int(width * 0.05))
        gradient_roi = vertical_gradient[:, x_margin : width - x_margin]
        row_strength = gradient_roi.mean(axis=1)
        row_coverage = (gradient_roi >= 25.0).mean(axis=1)
        valid_rows = np.flatnonzero(
            (row_strength >= 18.0) & (row_coverage >= 0.60)
        )
        projection_candidates: list[CandidateLine] = []
        if valid_rows.size:
            groups = np.split(valid_rows, np.where(np.diff(valid_rows) > 1)[0] + 1)
            for group in groups:
                y = int(group[np.argmax(row_strength[group])])
                if margin <= y < height - margin:
                    projection_candidates.append(
                        CandidateLine(
                            x_margin, y, width - x_margin - 1, y,
                            1.0 + float(row_strength[y]) / 100.0,
                        )
                    )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(24, int(width * 0.09)),
            minLineLength=max(50, int(width * self.min_length_ratio)),
            maxLineGap=max(12, int(width * 0.12)),
        )
        if lines is None:
            return projection_candidates

        candidates: list[CandidateLine] = list(projection_candidates)
        for x1, y1, x2, y2 in lines[:, 0]:
            if x1 <= x2:
                x_left, y_left, x_right, y_right = x1, y1, x2, y2
            else:
                x_left, y_left, x_right, y_right = x2, y2, x1, y1
            dx = x_right - x_left
            dy = abs(y_right - y_left)
            if dx <= 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if dx >= width * self.min_length_ratio and angle <= self.max_horizontal_angle:
                length = math.hypot(dx, dy)
                score = (length / width) * (1.0 - angle / 90.0)
                candidates.append(
                    CandidateLine(
                        int(x_left), int(y_left), int(x_right), int(y_right),
                        score,
                    )
                )

        # 粗横线的上下边缘会产生多条 Hough 线，按纵向位置合并。
        candidates.sort(key=lambda line: line.score, reverse=True)
        distinct: list[CandidateLine] = []
        merge_distance = max(8, int(height * 0.04))
        for candidate in candidates:
            if all(
                abs(candidate.center_y - kept.center_y) >= merge_distance
                for kept in distinct
            ):
                distinct.append(candidate)
        return distinct

    def inspect(self, frame: np.ndarray) -> tuple[bool, list[CandidateLine]]:
        if frame is None or frame.size == 0:
            return False, []
        height = frame.shape[0]
        candidates = self._extract_candidates(frame)
        self.last_candidates = candidates
        self.raw_detected = bool(candidates)

        if not self.tracking:
            bottom_lines = [
                line for line in candidates
                if line.center_y >= height * self.bottom_ratio
            ]
            if bottom_lines:
                tracked = max(bottom_lines, key=lambda line: line.center_y)
                self.tracking = True
                self.start_y = tracked.center_y
                self.last_y = tracked.center_y
                self.best_y = tracked.center_y
                self.track_hits = 1
                self.missed_frames = 0
                self.last_tracked = tracked
            return False, candidates

        # 每帧只允许有限位移及少量向下抖动，防止跳到另一条横线。
        max_step = height * 0.28
        downward_jitter = height * 0.06
        eligible = [
            line for line in candidates
            if abs(line.center_y - self.last_y) <= max_step
            and line.center_y <= self.last_y + downward_jitter
        ]
        if not eligible:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self._clear_track()
            return False, candidates

        tracked = min(eligible, key=lambda line: abs(line.center_y - self.last_y))
        self.last_tracked = tracked
        self.last_y = tracked.center_y
        self.best_y = min(self.best_y, tracked.center_y)
        self.track_hits += 1
        self.missed_frames = 0
        travel = self.start_y - self.best_y
        confirmed = bool(
            tracked.center_y <= height * self.top_ratio
            and travel >= height * self.min_travel_ratio
            and self.track_hits >= self.min_track_frames
        )
        return confirmed, candidates

    def __call__(self, frame: np.ndarray) -> bool:
        confirmed, _ = self.inspect(frame)
        return confirmed

    def annotate(self, frame: np.ndarray, confirmed: bool) -> np.ndarray:
        output = frame.copy()
        height, width = output.shape[:2]
        cv2.line(output, (0, int(height * self.top_ratio)),
                 (width - 1, int(height * self.top_ratio)), (255, 180, 0), 1)
        cv2.line(output, (0, int(height * self.bottom_ratio)),
                 (width - 1, int(height * self.bottom_ratio)), (255, 180, 0), 1)
        for line in self.last_candidates:
            cv2.line(output, (line.x_left, line.y_left),
                     (line.x_right, line.y_right), (0, 255, 0), 2)
        if self.last_tracked is not None:
            line = self.last_tracked
            cv2.line(output, (line.x_left, line.y_left),
                     (line.x_right, line.y_right), (0, 0, 255), 4)
        travel = max(0.0, self.start_y - self.best_y) if self.tracking else 0.0
        status = "CULVERT" if confirmed else ("TRACKING" if self.tracking else "WAIT BOTTOM")
        color = (0, 0, 255) if confirmed else (0, 220, 255)
        cv2.putText(output, f"{status} hits={self.track_hits} travel={travel:.0f}px",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return output


class ChassisLink:
    def __init__(self, port: str, baudrate: int = 115200, dry_run: bool = False):
        self.dry_run = dry_run
        self.serial = None
        if not dry_run:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("缺少 pyserial，请先安装 python3-serial") from exc
            self.serial = serial.Serial(port, baudrate, timeout=0)
            print(f"[SERIAL] connected: {port}")

    def send(self, command: str) -> None:
        print(f"[SERIAL TX] {command}")
        if self.serial is not None:
            self.serial.write(command.encode("ascii"))
            self.serial.flush()

    def trigger_culvert(self) -> None:
        # 指令 5 在下位机接收任务中具有最高优先级，可立即中断当前运动。
        self.send("@|5|0|0|0#")
        time.sleep(0.15)
        self.send("@|10|0|0|0#")

    def read_lines(self) -> Iterable[str]:
        if self.serial is None:
            return []
        messages: list[str] = []
        while self.serial.in_waiting:
            line = self.serial.readline().decode("utf-8", errors="ignore").strip()
            if line:
                messages.append(line)
        return messages

    def close(self) -> None:
        if self.serial is not None:
            self.serial.close()


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def configure_desktop_display(show: bool) -> None:
    """让 VS Code/SSH 终端启动的 OpenCV 窗口显示到本机桌面。"""
    if not show or os.environ.get("DISPLAY"):
        return
    runtime_dir = Path(f"/run/user/{os.getuid()}")
    auth_files = sorted(
        runtime_dir.glob(".mutter-Xwaylandauth.*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if Path("/tmp/.X11-unix/X0").exists() and auth_files:
        os.environ["DISPLAY"] = ":0"
        os.environ["XAUTHORITY"] = str(auth_files[0])
        os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
        print(f"[DISPLAY] using :0 ({auth_files[0]})", flush=True)
        return
    raise RuntimeError(
        "No desktop DISPLAY found. Run with --no-show or start from the local desktop."
    )


def open_camera(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
    cap = cv2.VideoCapture(source, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {source}")
    print(
        f"[CAMERA] opened source={source} "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"fps={cap.get(cv2.CAP_PROP_FPS):.1f}",
        flush=True,
    )
    return cap


def run_self_test() -> int:
    detector = CulvertLineDetector(min_track_frames=3)
    results = []
    for y in (220, 185, 150, 110, 70, 30):
        positive = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.line(positive, (15, y), (305, y - 4), (10, 10, 10), 8)
        results.append(detector(positive))
    if not results[-1]:
        print("[SELF TEST] FAIL: 未识别横线从底部到顶部的轨迹")
        return 1
    detector.reset()
    false_results = []
    for _ in range(8):
        negative = np.full((240, 320, 3), 220, dtype=np.uint8)
        cv2.line(negative, (15, 190), (305, 190), (10, 10, 10), 8)
        false_results.append(detector(negative))
    if any(false_results):
        print("[SELF TEST] FAIL: 静止横线被误识别")
        return 1
    print("[SELF TEST] PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="涵洞横线由底部上移至顶部的联动测试")
    parser.add_argument("--camera", default=os.environ.get("DETECT_CAMERA", "2"))
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--flip", choices=("none", "-1", "0", "1"), default="none",
                        help="画面翻转方式；高位摄像头默认不翻转")
    parser.add_argument("--min-track-frames", type=int, default=3)
    parser.add_argument("--max-missed-frames", type=int, default=5)
    parser.add_argument("--top-ratio", type=float, default=0.35,
                        help="横线到达画面高度比例，越大越宽松")
    parser.add_argument("--bottom-ratio", type=float, default=0.55,
                        help="开始跟踪的画面高度比例，越小越宽松")
    parser.add_argument("--min-travel-ratio", type=float, default=0.18,
                        help="横线最小上移距离占画面高度比例")
    parser.add_argument("--min-length-ratio", type=float, default=0.25,
                        help="横线最小长度占画面宽度比例")
    parser.add_argument("--max-horizontal-angle", type=float, default=22.0,
                        help="允许横线相对水平的最大角度")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示带检测标注的实时摄像头画面（默认开启，--no-show 关闭）",
    )
    parser.add_argument("--dry-run", action="store_true", help="不打开串口、不控制底盘")
    parser.add_argument(
        "--repeat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="收到 Y_END 后自动重新启用涵洞检测（默认开启，--no-repeat 关闭）",
    )
    parser.add_argument("--save-dir", default="", help="触发时保存标注图片的目录")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    configure_desktop_display(args.show)
    detector = CulvertLineDetector(
        min_track_frames=args.min_track_frames,
        max_missed_frames=args.max_missed_frames,
        top_ratio=args.top_ratio,
        bottom_ratio=args.bottom_ratio,
        min_travel_ratio=args.min_travel_ratio,
        min_length_ratio=args.min_length_ratio,
        max_horizontal_angle=args.max_horizontal_angle,
    )
    cap = open_camera(parse_source(args.camera), args.width, args.height)
    if args.show:
        cv2.namedWindow("Culvert detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Culvert detector", args.width * 2, args.height * 2)
        print("[DISPLAY] preview window created; press q or Esc to exit", flush=True)
    chassis = ChassisLink(args.serial, dry_run=args.dry_run)
    triggered = False
    line_detection_enabled = True
    flip_code = None if args.flip == "none" else int(args.flip)
    frame_count = 0
    first_frame_reported = False
    print("[SYS] 涵洞识别开始；横线从画面底部移动到顶部后将急停并启动嫌疑人扫描")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if not first_frame_reported:
                print(f"[CAMERA] first frame received: {frame.shape}", flush=True)
                first_frame_reported = True
            if flip_code in (-1, 0, 1):
                frame = cv2.flip(frame, flip_code)
            if line_detection_enabled:
                confirmed, _ = detector.inspect(frame)
            else:
                # 涵洞触发后由下位机云台和成熟上位机负责嫌疑人识别，
                # 此时不再让横线检测器参与决策。
                confirmed = False
                detector.raw_detected = False
            frame_count += 1

            if confirmed and not triggered:
                triggered = True
                line_detection_enabled = False
                print(f"[CULVERT] confirmed at frame {frame_count}; stopping chassis")
                chassis.trigger_culvert()
                annotated = detector.annotate(frame, True)
                if args.save_dir:
                    save_dir = Path(args.save_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    path = save_dir / f"culvert_{int(time.time())}.jpg"
                    cv2.imwrite(str(path), annotated)
                    print(f"[DEBUG] saved: {path}")

            for message in chassis.read_lines():
                print(f"[SERIAL RX] {message}")
                if message == "Y_START":
                    print("[SUSPECT] lower controller entered suspect-camera mode")
                elif message in {"Y_L", "Y_R"}:
                    print(f"[SUSPECT] gimbal position {message}; run suspect recognition now")
                elif message == "Y_END" and args.repeat:
                    detector.reset()
                    triggered = False
                    line_detection_enabled = True
                    print("[CULVERT] re-armed")

            if args.show:
                annotated = detector.annotate(frame, confirmed)
                if not line_detection_enabled:
                    cv2.putText(annotated, "SUSPECT MODE - line detection OFF",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (255, 80, 80), 2)
                cv2.imshow("Culvert detector", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\n[SYS] stopped")
    finally:
        cap.release()
        chassis.close()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
