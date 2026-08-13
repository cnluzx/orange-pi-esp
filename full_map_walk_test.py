"""无障碍、无涵洞的全地图行走测试。

该测试沿用 ``line_follow_test.py`` 的硬件调用方式：巡线模块直接读取
摄像头，底盘动作直接通过 ``ChassisController`` 发送，并在每个动作后轮询
下位机的 ``@`` 完成应答。地图规划只使用物理道路和 N1～N12 固定点，完成
条件为覆盖全部道路、经过所有固定点并返回出发区。
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass

import cv2

from Serial import ChassisController
from line_detect import LineTracker
from navigator import Navigator
from topo_map import TopoMap
from odometry import Odometry
from rfid import RFIDReader
try:
    from yolo_detect import YoloDetector
except Exception as exc:
    YoloDetector = None
    print(f"[YOLO WARN] 无法导入目标识别模块: {exc}")
try:
    from oled import OLEDDisplay
except Exception as exc:
    OLEDDisplay = None
    print(f"[OLED WARN] 无法导入显示模块: {exc}")
try:
    from broadcast import Broadcaster
except Exception as exc:
    Broadcaster = None
    print(f"[Audio WARN] 无法导入播报模块: {exc}")

_TEST_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_code")
if _TEST_CODE_DIR not in sys.path:
    sys.path.insert(0, _TEST_CODE_DIR)
from Culvert_detect import CulvertBrightnessDetector


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed L-shaped corners from the competition topology. J_TOP is handled
# by the dedicated START_ALIGN logic and is intentionally not in this set.
L_JUNCTION_NODES = frozenset({"N1", "N2", "N5", "N8", "N11", "N12"})

# OpenCV 在运行时提供该 C 扩展函数，但部分 cv2 类型存根没有声明它。
_video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _parse_camera_source(value: str) -> int | str:
    value = str(value).strip()
    try:
        return int(value)
    except ValueError:
        return value


def _parse_flip(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"none", "off", "false", "no"}:
        return None
    code = int(value)
    if code not in (-1, 0, 1):
        raise ValueError("flip 只能填写 None、-1、0 或 1")
    return code


@dataclass
class FullMapConfig:
    camera: int | str = _parse_camera_source(os.environ.get("NAV_CAMERA", "0"))
    flip: int | None = _parse_flip(os.environ.get("NAV_CAMERA_FLIP", "none"))
    serial_port: str = os.environ.get(
        "CHASSIS_PORT",
        "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0",
    )
    baudrate: int = 115200
    speed: int = int(os.environ.get("FULL_MAP_LINE_SPEED", "10"))
    kp: float = float(os.environ.get("FULL_MAP_KP", "0.35"))
    kd: float = float(os.environ.get("FULL_MAP_KD", "0.10"))
    max_angle: float = float(os.environ.get("FULL_MAP_MAX_ANGLE", "30"))
    action_timeout: float = float(os.environ.get("FULL_MAP_ACTION_TIMEOUT", "10"))
    camera_startup_timeout: float = float(
        os.environ.get("CAMERA_STARTUP_TIMEOUT", "10")
    )
    system_ready_timeout: float = float(
        os.environ.get("SYSTEM_READY_TIMEOUT", "20")
    )
    startup_min_road_fps: float = float(
        os.environ.get("STARTUP_MIN_ROAD_FPS", "10")
    )
    startup_stable_frames: int = max(
        2, int(os.environ.get("STARTUP_STABLE_FRAMES", "15"))
    )
    start_action_timeout: float = float(
        os.environ.get("START_ACTION_TIMEOUT", "8")
    )
    start_line_required_frames: int = max(
        1, int(os.environ.get("START_LINE_STABLE_FRAMES", "30"))
    )
    start_right_turn_angle: int = int(
        os.environ.get("START_RIGHT_TURN_ANGLE", "80")
    )
    start_forward_speed: int = int(os.environ.get("START_FORWARD_SPEED", "10"))
    start_forward_distance: int = int(
        os.environ.get("START_FORWARD_DISTANCE", "120")
    )
    return_start_speed: int = int(os.environ.get("RETURN_START_SPEED", "10"))
    return_start_distance: int = int(
        os.environ.get("RETURN_START_DISTANCE", "100")
    )
    return_start_timeout: float = float(
        os.environ.get("RETURN_START_TIMEOUT", "6")
    )
    cross_timeout: float = float(os.environ.get("FULL_MAP_CROSS_TIMEOUT", "30"))
    countdown: int = int(os.environ.get("FULL_MAP_COUNTDOWN", "3"))
    max_camera_misses: int = int(os.environ.get("FULL_MAP_MAX_CAMERA_MISSES", "10"))
    cross_cooldown: float = float(os.environ.get("FULL_MAP_CROSS_COOLDOWN", "1"))
    debug: bool = _env_bool("LINE_DEBUG", False)
    culvert_camera: int | str = _parse_camera_source(os.environ.get("CULVERT_CAMERA", "2"))
    culvert_flip: int | None = _parse_flip(os.environ.get("CULVERT_CAMERA_FLIP", "none"))
    rfid_port: str = os.environ.get("RFID_PORT", "/dev/ttyUSB0")
    rfid_baudrate: int = int(os.environ.get("RFID_BAUDRATE", "115200"))
    culvert_min_length_ratio: float = float(
        os.environ.get("CULVERT_MIN_LENGTH_RATIO", "0.35")
    )
    culvert_stop_delay: float = float(
        os.environ.get("CULVERT_STOP_DELAY", "0")
    )
    culvert_detect_interval: int = max(
        1, int(os.environ.get("CULVERT_DETECT_INTERVAL", "1"))
    )
    # 高位涵洞画面默认不弹窗，避免与底部巡线调试窗口争用 Qt 线程。
    culvert_debug: bool = _env_bool("CULVERT_DEBUG", False)
    culvert_scan_timeout: float = float(
        os.environ.get("CULVERT_SCAN_TIMEOUT", "20")
    )
    culvert_rearm_delay: float = float(
        os.environ.get("CULVERT_REARM_DELAY", "3")
    )


def build_test_navigator():
    """建立不含绿色任务点、但要求覆盖全部道路的导航器。"""
    topo_map = TopoMap()
    topo_map.build_competition_track(green_count=0)
    navigator = Navigator(topo_map)
    navigator.init_mission(
        start_node="START_AREA",
        mission_plan=topo_map.fixed_patrol_nodes,
        expected_green_count=int(os.environ.get("CULVERT_EXPECTED_COUNT", "8")),
        cover_all_edges=True,
    )
    # 将最终返航支路延后覆盖：第一次到达 N12 时原地掉头，继续巡检
    # 主地图；其他道路完成后才走 N12 → J_TOP → START_AREA。
    navigator.defer_coverage_edges([("N12", "J_TOP")])
    return topo_map, navigator


def calculate_plan():
    """离线推进规划器，得到本次测试会覆盖的完整节点路线。"""
    topo_map, navigator = build_test_navigator()
    route: list[str] = [str(navigator.current_node)]
    safety_limit = max(100, len(topo_map.nodes) * len(topo_map.edges) * 4)

    while not navigator.mission_complete and len(route) < safety_limit:
        segment = navigator.get_next_segment()
        if not segment:
            break
        reached = navigator.advance_path()
        if reached is None:
            break
        route.append(str(reached))

    return route, navigator.get_progress()


def print_plan() -> bool:
    route, progress = calculate_plan()
    print("[FULL MAP TEST] 无障碍、无涵洞任务")
    print("[FULL MAP TEST] 路线: " + " -> ".join(route))
    print(
        "[FULL MAP TEST] 道路覆盖: "
        f"{progress['covered_edges']}/{progress['required_edges']} "
        f"({progress['coverage_percent']:.1f}%)"
    )
    print(
        "[FULL MAP TEST] 固定点: "
        f"{progress['visited']}/{progress['total']}; "
        f"缺失={progress['missing']}"
    )
    print(f"[FULL MAP TEST] 返回出发区并完成: {progress['complete']}")
    return bool(progress["complete"])


class FullMapWalkTest:
    """使用 line_follow_test 同款接口运行地图巡线状态机。"""

    def __init__(self, args: FullMapConfig):
        self.args = args
        self.running = True
        self.stop_requested = False
        self.last_offset = 0
        self.last_action_error = None
        self.last_cross_time = 0.0
        self._last_line_log_time = 0.0
        self._road_frame_count = 0
        self._road_frame_log_time = time.monotonic()
        self._road_frame_log_count = 0
        self._road_frame_times = deque(maxlen=args.startup_stable_frames)
        self._road_fps_stable = threading.Event()
        self._startup_road_fps = 0.0
        self._camera_thread_started = threading.Event()
        self._culvert_camera_thread_started = threading.Event()
        self._road_thread_started = threading.Event()
        self._culvert_thread_started = threading.Event()
        self._rfid_thread_started = threading.Event()
        self._broadcast_thread_started = threading.Event()
        self.start_line_frames = 0
        self.cross_event_count = 0
        self.action_count = 0
        self.state = "START"
        self.odometry = Odometry()
        self._culvert_detector = CulvertBrightnessDetector(
            consistency_threshold=float(os.environ.get("CULVERT_BRIGHTNESS_THRESHOLD", "0.12")),
            edge_threshold=float(os.environ.get("CULVERT_EDGE_THRESHOLD", "0.012")),
            min_consecutive=3,
        )
        self._culvert_camera = None
        self._culvert_camera_lock = threading.Lock()
        self._culvert_camera_ready = threading.Event()
        self._culvert_camera_first_frame = threading.Event()
        self._culvert_camera_frame = None
        self._culvert_camera_sequence = 0
        self._culvert_lock = threading.Lock()
        self._culvert_triggered = threading.Event()
        self._culvert_active = threading.Event()
        self._culvert_road_lock = threading.Lock()
        self._culvert_road = None
        self._culvert_scanned_edges = set()
        self._culvert_rearm_at = 0.0
        self._npu_lock = threading.Lock()
        self.chassis = ChassisController(args.serial_port, args.baudrate)
        self.tracker = LineTracker(debug_mode=args.debug)
        self._yolo = None
        if YoloDetector is not None:
            try:
                self._yolo = YoloDetector(
                    preset_classes=tuple(range(1, 21)),
                    show_window=False,
                    save_debug=False,
                )
                if self._yolo.rknn is None:
                    self._yolo = None
                    print("[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试")
                else:
                    print("[YOLO] 目标模型已预加载")
            except Exception as exc:
                self._yolo = None
                print(f"[YOLO WARN] 目标模型预加载失败，涵洞扫描时将重试: {exc}")

        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self.camera = cv2.VideoCapture(args.camera, backend)
        except Exception:
            self.camera = cv2.VideoCapture(args.camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG"))
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Capture frames independently; the main loop always consumes the latest frame.
        self._camera_lock = threading.Lock()
        self._camera_ready = threading.Event()
        self._camera_frame = None
        self._camera_sequence = 0
        self._processed_camera_sequence = 0
        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._camera_thread.start()
        self._road_lock = threading.Lock()
        self._road_ready = threading.Event()
        self._road_result: tuple[str, int] | None = None
        self._road_result_sequence = 0
        self._consumed_road_sequence = 0

        # 高位摄像头独立采集，涵洞识别线程不会占用巡线摄像头帧。
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera, backend)
        except Exception:
            self._culvert_camera = cv2.VideoCapture(args.culvert_camera)
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._culvert_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self._culvert_camera.set(cv2.CAP_PROP_FPS, 15)
        self._culvert_camera.set(
            cv2.CAP_PROP_FOURCC, _video_writer_fourcc(*"MJPG")
        )
        self._culvert_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._culvert_camera_thread = threading.Thread(
            target=self._culvert_camera_capture_loop,
            name="culvert-camera-capture",
            daemon=True,
        )
        self._culvert_camera_thread.start()
        self._rfid_reader = RFIDReader(port=args.rfid_port, baudrate=args.rfid_baudrate)
        self._broadcast_queue: queue.Queue[tuple[str, int]] = queue.Queue()
        self._broadcaster = Broadcaster() if Broadcaster is not None else None
        self._oled_lock = threading.Lock()
        self._oled = None
        if OLEDDisplay is not None:
            try:
                self._oled = OLEDDisplay()
            except Exception as exc:
                print(f"[OLED WARN] 显示模块初始化失败: {exc}")

        self.topo_map, self.navigator = build_test_navigator()
        self.current_node = "START_AREA"
        self.previous_node = None
        self.current_path = None
        self.current_target_node = None
        self._sync_navigation_path()
        self._road_thread = threading.Thread(
            target=self._road_processing_loop, name="road-processing", daemon=True
        )
        self._culvert_thread = threading.Thread(target=self._culvert_loop, name="culvert-detect", daemon=True)
        self._rfid_thread = threading.Thread(target=self._rfid_loop, name="rfid-reader", daemon=True)
        self._broadcast_thread = threading.Thread(target=self._broadcast_loop, name="broadcast", daemon=True)
        self._culvert_thread.start()
        self._rfid_thread.start()
        self._road_thread.start()
        self._broadcast_thread.start()
        self._show_oled("全地图测试\n正在准备系统", duration=1200)

    def _show_oled(self, message: str, duration: int = 1200) -> None:
        """非阻塞显示运行状态；显示故障不影响底盘安全控制。"""
        if self._oled is None:
            return
        with self._oled_lock:
            try:
                self._oled.show_text(message, duration=duration)
            except Exception as exc:
                print(f"[OLED WARN] 显示失败: {exc}")

    def _camera_capture_loop(self) -> None:
        self._camera_thread_started.set()
        while self.running:
            ok, frame = self.camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.005)
                continue
            if self.args.flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.flip)
            with self._camera_lock:
                self._camera_frame = frame
                self._camera_sequence += 1
            self._camera_ready.set()

    def _culvert_camera_capture_loop(self) -> None:
        """持续清空高位UVC缓存，并保存带序号的最新帧。"""
        self._culvert_camera_thread_started.set()
        next_capture_at = 0.0
        while self.running:
            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(min(0.01, next_capture_at - now))
                continue
            camera = self._culvert_camera
            if camera is None or not camera.isOpened():
                time.sleep(0.05)
                continue
            ok, frame = camera.read()
            if not ok or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue
            with self._culvert_camera_lock:
                self._culvert_camera_frame = frame
                self._culvert_camera_sequence += 1
            self._culvert_camera_first_frame.set()
            self._culvert_camera_ready.set()
            # 高位摄像头始终保持15FPS，减少云台到位后的取帧延迟。
            next_capture_at = time.monotonic() + (1.0 / 15.0)

    def _ensure_yolo(self) -> bool:
        """确认预加载模型可用；启动时加载失败则在扫描阶段重试。"""
        if self._yolo is not None and self._yolo.rknn is not None:
            return True
        if YoloDetector is None:
            print("[YOLO ERROR] 目标识别模块不可用")
            return False
        try:
            self._yolo = YoloDetector(
                preset_classes=tuple(range(1, 21)),
                show_window=False,
                save_debug=False,
            )
            ready = self._yolo.rknn is not None
            print(f"[YOLO] 扫描阶段重试加载 {'成功' if ready else '失败'}")
            return ready
        except Exception as exc:
            self._yolo = None
            print(f"[YOLO ERROR] 目标模型按需加载失败: {exc}")
            return False

    def _release_yolo(self) -> None:
        yolo, self._yolo = self._yolo, None
        if yolo is not None:
            yolo.release()
            print("[YOLO] 目标模型资源已释放")

    def _latest_culvert_frame(
        self, after_sequence: int = -1, min_new_frames: int = 1,
        timeout: float = 0.8,
    ):
        """返回 after_sequence 之后至少 min_new_frames 的高位新帧。"""
        target_sequence = after_sequence + max(1, min_new_frames)
        deadline = time.monotonic() + timeout
        while self.running and time.monotonic() < deadline:
            with self._culvert_camera_lock:
                sequence = self._culvert_camera_sequence
                frame = self._culvert_camera_frame
                if sequence >= target_sequence and frame is not None:
                    return frame.copy(), sequence
                # 当前帧尚未达到目标序号，清除旧通知后等待采集线程发布
                # 真正的新帧。原逻辑未清除Event，会在两帧之间持续空转。
                self._culvert_camera_ready.clear()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                self._culvert_camera_ready.wait(min(0.1, remaining))
        return None, after_sequence

    def _road_processing_loop(self) -> None:
        """后台执行 RKNN 巡线推理，主线程只读取最新结果。"""
        self._road_thread_started.set()
        while self.running:
            if self._culvert_active.is_set():
                time.sleep(0.02)
                continue
            frame = self._latest_camera_frame(timeout=0.5)
            if frame is None:
                continue
            self.tracker.set_junction_mode(self._expected_junction_mode())
            with self._npu_lock:
                result = self.tracker.get_road_info(frame)
                with self._road_lock:
                    self._road_result = result
                    self._road_result_sequence += 1
                    self._road_ready.set()
                self._road_frame_count += 1
                self._road_frame_log_count += 1
                now = time.monotonic()
                self._road_frame_times.append(now)
                if len(self._road_frame_times) == self.args.startup_stable_frames:
                    stable_elapsed = self._road_frame_times[-1] - self._road_frame_times[0]
                    if stable_elapsed > 0:
                        self._startup_road_fps = (
                            (len(self._road_frame_times) - 1) / stable_elapsed
                        )
                        if self._startup_road_fps >= self.args.startup_min_road_fps:
                            self._road_fps_stable.set()
                if now - self._road_frame_log_time >= 1.0:
                    elapsed = max(0.001, now - self._road_frame_log_time)
                    fps = self._road_frame_log_count / elapsed
                    print(
                        f"[ROAD FRAME] total={self._road_frame_count} "
                        f"seq={self._road_result_sequence} status={result[0]} "
                        f"offset={result[1]} infer={self.tracker.last_inference_ms:.1f}ms "
                        f"fps={fps:.1f}"
                    )
                    self._road_frame_log_time = now
                    self._road_frame_log_count = 0

    def _culvert_loop(self) -> None:
        self._culvert_thread_started.set()
        frame_count = 0
        last_camera_sequence = -1
        debug_enabled = self.args.culvert_debug
        if debug_enabled:
            try:
                cv2.namedWindow("Culvert high camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Culvert high camera", 960, 540)
            except cv2.error as exc:
                print(f"[CULVERT] 调试窗口不可用: {exc}")
                debug_enabled = False
        while self.running:
            frame, last_camera_sequence = self._latest_culvert_frame(
                after_sequence=last_camera_sequence, timeout=0.5
            )
            if frame is None:
                continue
            if self.args.culvert_flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.culvert_flip)
            with self._culvert_road_lock:
                active_road = self._culvert_road
            if active_road is None:
                with self._culvert_lock:
                    self._culvert_detector.reset()
                continue
            if time.monotonic() < self._culvert_rearm_at:
                with self._culvert_lock:
                    self._culvert_detector.reset()
                time.sleep(0.05)
                continue
            frame_count += 1
            if frame_count % self.args.culvert_detect_interval:
                if debug_enabled:
                    try:
                        cv2.imshow("Culvert high camera", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            self.args.culvert_debug = False
                            debug_enabled = False
                    except cv2.error:
                        debug_enabled = False
                continue
            with self._culvert_lock:
                confirmed, candidates = self._culvert_detector.inspect(frame)
            if debug_enabled:
                try:
                    with self._culvert_lock:
                        annotated = self._culvert_detector.annotate(frame, confirmed)
                    cv2.imshow("Culvert high camera", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self.args.culvert_debug = False
                        debug_enabled = False
                except cv2.error:
                    debug_enabled = False
            if not confirmed or self._culvert_triggered.is_set():
                continue
            self._culvert_triggered.set()
            print(
                f"[CULVERT] 高位摄像头确认涵洞，道路={active_road[0]} -> {active_road[1]}，"
                f"候选线={len(candidates)}"
            )
            self._run_culvert_scan(*active_road)
        if self.args.culvert_debug:
            try:
                cv2.destroyWindow("Culvert high camera")
            except cv2.error:
                pass

    def _rfid_loop(self) -> None:
        self._rfid_thread_started.set()
        while self.running:
            try:
                card_id = self._rfid_reader.read_card()
                if card_id is not None:
                    self.odometry.calibrate_by_rfid(card_id)
                    self._broadcast_queue.put(("point", int(card_id)))
                    self._show_oled(f"到达固定点\nID: {card_id}")
                    print(
                        f"[RFID] 读取任务卡 ID={card_id}，仅用于播报，"
                        "不修改导航当前位置"
                    )
            except Exception as exc:
                print(f"[RFID] 读取线程异常: {exc}")
            time.sleep(0.02)

    def _broadcast_loop(self) -> None:
        self._broadcast_thread_started.set()
        while self.running or not self._broadcast_queue.empty():
            try:
                event_type, event_id = self._broadcast_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if self._broadcaster is not None:
                    if event_type == "point":
                        self._broadcaster.say_point(event_id)
                    elif event_type == "target":
                        self._broadcaster.say_target(event_id)
                    else:
                        print(f"[Audio WARN] 未知播报事件: {event_type}")
                else:
                    print(f"[Audio] 播报模块不可用，跳过 {event_type}:{event_id}")
            except Exception as exc:
                print(f"[Audio Error] {event_type}:{event_id} 播报失败: {exc}")
            finally:
                self._broadcast_queue.task_done()

    def _latest_camera_frame(self, timeout: float = 0.15):
        if not self._camera_ready.wait(timeout):
            return None
        deadline = time.monotonic() + timeout
        while self.running:
            with self._camera_lock:
                sequence = self._camera_sequence
                if sequence != self._processed_camera_sequence and self._camera_frame is not None:
                    frame = self._camera_frame.copy()
                    self._processed_camera_sequence = sequence
                    return frame
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)
        return None

    def validate_hardware(self) -> None:
        if not self.chassis.serial or not self.chassis.serial.is_open:
            raise RuntimeError(f"底盘串口不可用: {self.args.serial_port}")
        if not self.camera.isOpened():
            raise RuntimeError(f"巡线摄像头不可用: {self.args.camera}")
        if self._culvert_camera is None or not self._culvert_camera.isOpened():
            raise RuntimeError(f"高位涵洞摄像头不可用: {self.args.culvert_camera}")

    def _wait_until_system_ready(self) -> bool:
        """保持停车，直到所有模块、工作线程和巡线帧率均已就绪。"""
        checks = (
            ("巡线采集线程", self._camera_thread_started, lambda: self._camera_thread.is_alive()),
            ("高位采集线程", self._culvert_camera_thread_started, lambda: self._culvert_camera_thread.is_alive()),
            ("巡线推理线程", self._road_thread_started, lambda: self._road_thread.is_alive()),
            ("涵洞识别线程", self._culvert_thread_started, lambda: self._culvert_thread.is_alive()),
            ("RFID线程", self._rfid_thread_started, lambda: self._rfid_thread.is_alive()),
            ("播报线程", self._broadcast_thread_started, lambda: self._broadcast_thread.is_alive()),
            ("巡线摄像头画面", self._camera_ready, lambda: True),
            ("高位摄像头画面", self._culvert_camera_first_frame, lambda: True),
            ("巡线推理帧率", self._road_fps_stable, lambda: True),
        )
        modules = (
            ("RFID", self._rfid_reader is not None),
            ("播报", self._broadcaster is not None),
            ("OLED", self._oled is not None),
            ("YOLO", self._yolo is not None and self._yolo.rknn is not None),
        )
        deadline = time.monotonic() + self.args.system_ready_timeout
        last_pending = None
        self._show_oled("全地图测试\n等待系统就绪", duration=1200)
        while self.running and time.monotonic() < deadline:
            self.chassis.stop()
            pending = [name for name, ready in modules if not ready]
            pending.extend(
                name for name, event, alive in checks if not event.is_set() or not alive()
            )
            if not pending:
                print(
                    f"[STARTUP READY] 全部模块和线程已就绪；"
                    f"巡线最近{self.args.startup_stable_frames}帧="
                    f"{self._startup_road_fps:.1f} FPS"
                )
                self._show_oled("全地图测试\n系统已就绪", duration=1200)
                return True
            pending_text = ", ".join(pending)
            if pending_text != last_pending:
                print(f"[STARTUP WAIT] 等待: {pending_text}")
                last_pending = pending_text
            time.sleep(0.1)

        self.chassis.stop()
        pending = [name for name, ready in modules if not ready]
        pending.extend(
            name for name, event, alive in checks if not event.is_set() or not alive()
        )
        print(f"[STARTUP ERROR] 系统未全部就绪: {', '.join(pending) or '任务被中止'}")
        self._show_oled("启动失败\n系统未就绪", duration=4000)
        return False

    def stop(self, *_args: object) -> None:
        self.stop_requested = True
        self.running = False
        self.chassis.stop()

    def read_road(self) -> tuple[str, int] | None:
        # RKNN在双摄场景下单帧可能超过150ms；等待略放宽，避免把正常
        # 推理误判为连续丢帧。该等待不改变摄像头/推理线程的实际帧率。
        if not self._road_ready.wait(0.30):
            return None
        with self._road_lock:
            if self._road_result_sequence == self._consumed_road_sequence:
                self._road_ready.clear()
                return None
            result = self._road_result
            self._consumed_road_sequence = self._road_result_sequence
            # 事件只表示“存在尚未消费的新结果”。生产者和消费者都在
            # _road_lock 内修改它，避免新结果到达时被错误清除。
            self._road_ready.clear()
            if result is None:
                return None
            return result

    def _expected_junction_mode(self) -> str:
        """Use topology as a prior; vision must still confirm the junction."""
        if self.state == "START_ALIGN":
            return "NORMAL"
        if self.current_target_node in L_JUNCTION_NODES:
            return "L"
        return "NORMAL"

    def follow_once(self, offset: int) -> None:
        correction = self.args.kp * offset + self.args.kd * (
            offset - self.last_offset
        )
        self.last_offset = offset
        correction = max(-self.args.max_angle, min(self.args.max_angle, correction))
        self.chassis.drive_steer(self.args.speed, correction)

    def wait_for_completion(self, timeout: float | None = None) -> bool:
        """轮询 line_follow_test 使用的非阻塞串口读取接口。"""
        self.last_action_error = None
        deadline = time.monotonic() + (
            self.args.action_timeout if timeout is None else timeout
        )
        while self.running and time.monotonic() < deadline:
            message = self.chassis.read_message()
            if message and message.startswith("E:"):
                try:
                    left, right = message[2:].split(",", 1)
                    self.odometry.update_encoders(int(left), int(right))
                except (TypeError, ValueError):
                    print(f"[ODOMETRY] 忽略无法解析的编码器回报: {message}")
            if message and message.startswith("ERR:"):
                self.last_action_error = message
                print(f"[WARN] 下位机动作失败: {message}")
                return False
            # 兼容原版 ReconRobot 的监听逻辑：固件可能返回 ``@``，也
            # 可能返回带附加字段的完成行；两种都表示当前动作已结束。
            if message and "@" in message:
                return True
            time.sleep(0.01)
        return False

    def _run_action(self, name: str, command, timeout: float | None = None) -> bool:
        # 清除旧 ACK，避免上一条动作的完成消息被当前动作误收。
        self.action_count += 1
        action_id = self.action_count
        print(f"[ACTION #{action_id:03d} SEND] {name}")
        self.chassis.clear_pending_input()
        command()
        if self.wait_for_completion(timeout):
            print(f"[ACTION #{action_id:03d} DONE] {name}")
            return True
        if self.last_action_error:
            print(
                f"[ACTION #{action_id:03d} ERROR] {name} 被下位机拒绝: "
                f"{self.last_action_error}，立即停车"
            )
        else:
            print(f"[ACTION #{action_id:03d} TIMEOUT] {name}，立即停车")
        self.chassis.stop()
        return False

    def _set_culvert_road(self, from_node: str | None, to_node: str | None) -> None:
        """转向完成后更新当前道路，并决定是否启用涵洞识别。"""
        road = None
        if from_node and to_node and self.topo_map.is_configured_culvert_edge(from_node, to_node):
            edge = self.topo_map.edge_key(from_node, to_node)
            if edge not in self.topo_map.culvert_edges and edge not in self._culvert_scanned_edges:
                road = (from_node, to_node)
        with self._culvert_road_lock:
            changed = road != self._culvert_road
            self._culvert_road = road
        if changed:
            with self._culvert_lock:
                self._culvert_detector.reset()
            print(f"[CULVERT ROAD] 当前道路={road or '-'} 识别={'ON' if road else 'OFF'}")

    def _run_culvert_scan(self, from_node: str, to_node: str) -> None:
        """在当前普通道路确认涵洞后停车扫描并记录道路。"""
        edge = self.topo_map.edge_key(from_node, to_node)
        if edge in self._culvert_scanned_edges:
            print(f"[CULVERT] 道路 {from_node} -> {to_node} 已扫描，忽略重复触发")
            return
        self._culvert_scanned_edges.add(edge)
        self._culvert_active.set()
        scan_results = {"left": -1, "right": -1}
        try:
            # 涵洞一经确认立即停车，不再等待原先的0.5秒延迟。
            self.chassis.stop()
            self._show_oled(f"发现涵洞\n{from_node} -> {to_node}", duration=1500)
            self.navigator.mark_culvert_edge(from_node, to_node)
            self.chassis._send_command("@|5|0|0|0#")
            time.sleep(0.15)
            self._ensure_yolo()
            self.chassis._send_command("@|10|0|0|0#")
            captured_sides: set[str] = set()
            deadline = time.monotonic() + self.args.culvert_scan_timeout
            while self.running and time.monotonic() < deadline:
                message = self.chassis.read_message()
                if message:
                    message = message.strip()
                    print(f"[CULVERT SERIAL RX] {message}")
                    if message == "Y_START":
                        self._show_oled("云台扫描\n准备识别目标", duration=1500)
                    # 实车安装方向：Y_L(180度)朝小车右侧，Y_R(0度)朝小车左侧。
                    elif message == "Y_L" and "right" not in captured_sides:
                        captured_sides.add("right")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, frame_sequence = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=1, timeout=0.3
                        )
                        print(f"[YOLO SYNC] Y_L signal_seq={signal_sequence} capture_seq={frame_sequence}")
                        scan_results["right"] = self._detect_scan_target("右侧", "Y_L", scan_frame)
                    elif message == "Y_R" and "left" not in captured_sides:
                        captured_sides.add("left")
                        with self._culvert_camera_lock:
                            signal_sequence = self._culvert_camera_sequence
                        scan_frame, frame_sequence = self._latest_culvert_frame(
                            after_sequence=signal_sequence, min_new_frames=1, timeout=0.3
                        )
                        print(f"[YOLO SYNC] Y_R signal_seq={signal_sequence} capture_seq={frame_sequence}")
                        scan_results["left"] = self._detect_scan_target("左侧", "Y_R", scan_frame)
                    if message == "Y_END":
                        print(f"[CULVERT] 扫描完成，收到 Y_END，道路={from_node} -> {to_node}")
                        break
                time.sleep(0.02)
            else:
                print("[CULVERT WARN] 等待 Y_END 超时，恢复巡线")
            self._publish_scan_results(scan_results["left"], scan_results["right"])
        finally:
            with self._culvert_lock:
                self._culvert_detector.reset()
            self._culvert_rearm_at = time.monotonic() + max(0.0, self.args.culvert_rearm_delay)
            print(f"[CULVERT] 扫描结束，冷却 {self.args.culvert_rearm_delay:.1f}s，期间禁止重复触发")
            self._culvert_triggered.clear()
            self._culvert_active.clear()
            self._set_culvert_road(None, None)

    @staticmethod
    def _target_text(target_id: int) -> str:
        if target_id < 1:
            return "未识别"
        if target_id <= 10:
            return f"{target_id}号嫌疑人"
        return f"{target_id - 10}号刀具"

    def _capture_culvert_frame(self):
        """收到 Y_L/Y_R 后立即抓取第一张有效帧。

        下位机是在云台到位后才发送 Y_L/Y_R；收到信号后只等待第一张
        真正的新帧，避免使用信号前的缓存画面，也不额外等待多帧。
        """
        with self._culvert_camera_lock:
            sequence = self._culvert_camera_sequence
        frame, _ = self._latest_culvert_frame(
            after_sequence=sequence, min_new_frames=1, timeout=0.3
        )
        if frame is None:
            return None
        if self.args.culvert_flip in (-1, 0, 1):
            frame = cv2.flip(frame, self.args.culvert_flip)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        print(f"[YOLO FRAME] 信号后立即抓拍，清晰度={score:.1f}")
        return frame

    def _detect_scan_target(
        self, side_name: str, trigger_message: str = "", frame=None
    ) -> int:
        supplied_frame = frame is not None
        if frame is None:
            frame = self._capture_culvert_frame()
        if frame is None:
            print(f"[YOLO ERROR] {side_name}拍照失败")
            return -1
        else:
            frame = frame.copy()
            if supplied_frame and self.args.culvert_flip in (-1, 0, 1):
                frame = cv2.flip(frame, self.args.culvert_flip)
        print(
            f"[YOLO FRAME] {side_name}触发={trigger_message or '-'} "
            f"shape={frame.shape[1]}x{frame.shape[0]} "
            f"flip={self.args.culvert_flip} save=OFF"
        )
        if self._yolo is None or self._yolo.rknn is None:
            print(f"[YOLO ERROR] {side_name}目标模型不可用")
            return -1
        with self._npu_lock:
            target_id = int(self._yolo.detect_target(frame))
        text = self._target_text(target_id)
        print(f"[YOLO] {side_name}识别结果: {text} (business_id={target_id})")
        self._show_oled(f"{side_name}识别\n{text}", duration=1500)
        return target_id

    def _publish_scan_results(self, left_id: int, right_id: int) -> None:
        left_text = self._target_text(left_id)
        right_text = self._target_text(right_id)
        self._show_oled(
            f"左:{left_text}\n右:{right_text}", duration=2500
        )
        detected = []
        for target_id in (left_id, right_id):
            if target_id > 0 and target_id not in detected:
                detected.append(target_id)
        if detected:
            for target_id in detected:
                self._broadcast_queue.put(("target", target_id))
        else:
            self._broadcast_queue.put(("target", -1))

    def execute_cross_action(
        self,
        action: str,
        cross_id: int,
        at_node: str,
        next_node: str,
    ) -> bool:
        planned_action = action

        if action == "FORWARD":
            # 指令 7 会在下位机内阻塞等待 8000 个编码器脉冲，并可能触发
            # ERR:MOTION。直行通过路口只需恢复巡线直行；该指令无 ACK，
            # 后续主循环会持续发送巡线命令并驶离路口。
            self.action_count += 1
            action_id = self.action_count
            print(
                f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} "
                f"planned={planned_action} dispatched={action} "
                f"method=chassis.go_cross(speed={self.args.speed}, distance_multiplier=1)"
            )
            print(
                f"[ACTION #{action_id:03d} SEND] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD"
            )
            self.chassis.clear_pending_input()
            self.chassis.go_cross(speed=self.args.speed, distance_multiplier=1)
            if not self.wait_for_completion(self.args.action_timeout):
                print(
                    f"[ACTION #{action_id:03d} TIMEOUT] CROSS#{cross_id:03d} "
                    f"node={at_node} FORWARD，立即停车"
                )
                self.chassis.stop()
                return False
            print(
                f"[ACTION #{action_id:03d} DONE] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD（编码器闭环）"
            )
            time.sleep(0.3)
            self._set_culvert_road(at_node, next_node)
            return True

        commands = {
            "LEFT": (self.chassis.ccw, "chassis.ccw()"),
            "RIGHT": (self.chassis.cw, "chassis.cw()"),
            "U_TURN": (lambda: self.chassis.turn(180), "chassis.turn(180)"),
        }
        command_spec = commands.get(action)
        if command_spec is None:
            raise ValueError(f"未知路口动作: {action}")
        command, method = command_spec
        print(
            f"[CROSS #{cross_id:03d} DISPATCH] node={at_node} "
            f"planned={planned_action} dispatched={action} method={method}"
        )
        completed = self._run_action(
            f"CROSS#{cross_id:03d} node={at_node} {action} method={method}",
            command,
        )
        if completed:
            # 与原版 ReconRobot._execute_cross_action 保持相同的机械稳定时间，
            # 避免下一个路口命令紧跟在底盘完成 ACK 后发送。
            time.sleep(0.3)
            self._set_culvert_road(at_node, next_node)
        return completed

    def _sync_navigation_path(self) -> None:
        self.current_node = self.navigator.current_node
        self.previous_node = self.navigator.previous_node
        self.current_path = self.navigator.get_next_segment()
        self.current_target_node = (
            self.current_path[1]
            if self.current_path and len(self.current_path) > 1
            else None
        )

    def _confirm_next_node(self, cross_id: int | None = None) -> str | None:
        expected = self.current_target_node
        from_node = self.current_node
        reached = self.navigator.advance_path()
        self._sync_navigation_path()
        prefix = (
            f"[CROSS #{cross_id:03d} ARRIVAL]"
            if cross_id is not None
            else "[ARRIVAL]"
        )
        if reached:
            progress = self.navigator.get_progress()
            print(
                f"{prefix} from={from_node} expected={expected} reached={reached} "
                f"next={self.current_target_node or '-'}; "
                f"固定点 {progress['visited']}/{progress['total']}; "
                f"道路覆盖 {progress['covered_edges']}/"
                f"{progress['required_edges']}"
            )
        elif expected:
            print(
                f"{prefix} from={from_node} expected={expected} reached=None "
                "status=FAILED"
            )
        return reached

    def _run_start_actions(self) -> bool:
        print("\n[START_STATE] 全地图测试出发区固定动作...")
        self._show_oled("全地图测试\n正在出发", duration=1200)
        actions = (
            (
                f"出发区前进 speed={self.args.start_forward_speed} "
                f"distance={self.args.start_forward_distance}",
                lambda: self.chassis.distance(
                    val=self.args.start_forward_speed,
                    dis=self.args.start_forward_distance,
                ),
            ),
            (
                f"出发区右转 angle={self.args.start_right_turn_angle} "
                "method=chassis.turn_pure()",
                lambda: self.chassis.turn_pure(self.args.start_right_turn_angle),
            ),
        )
        for name, command in actions:
            if not self._run_action(name, command, self.args.start_action_timeout):
                return False
            time.sleep(0.2)

        self.start_line_frames = 0
        self.last_cross_time = 0.0
        self.state = "START_ALIGN"
        print(
            "[START_STATE] 前进和右转完成，进入姿态巡线；稳定后同步 J_TOP，"
            "J_TOP 转向由巡线完成，不再额外发送路口动作；"
            f"稳定帧数={self.args.start_line_required_frames}"
        )
        return True

    def _run_start_align(self, road_type: str, offset: int) -> None:
        self.follow_once(offset)
        if road_type not in {"LINE", "CROSS"}:
            return
        self.start_line_frames += 1
        if self.start_line_frames >= self.args.start_line_required_frames:
            # 实车会在该阶段依靠道路偏差自然完成 J_TOP 左转，且该过程不
            # 产生 CROSS 事件。这里同步逻辑节点，避免下一个实体路口再次
            # 被误认成 J_TOP，导致后续动作整体落后一站。
            reached = self._confirm_next_node()
            if reached != "J_TOP":
                print(
                    f"[START_SYNC ERROR] expected=J_TOP reached={reached}，停车"
                )
                self.chassis.stop()
                self.running = False
                return
            print(
                "[START_SYNC] logical_node=J_TOP next=N1; "
                "J_TOP 左转由视觉巡线完成，未发送额外路口动作"
            )
            print("[START_ALIGN] 姿态巡线稳定，开放后续路口识别")
            self._set_culvert_road("J_TOP", "N1")
            self.state = "FOLLOW_LINE"

    def _advance_virtual_targets(self, cross_id: int) -> bool:
        """推进 T1-T4 道路中点，但不为它们消耗视觉路口事件。"""
        while self.current_target_node:
            target = self.topo_map.get_node(self.current_target_node)
            if target is None or target.type != self.topo_map.NODE_TUNNEL:
                break

            expected = self.current_target_node
            reached = self.navigator.advance_path()
            self._sync_navigation_path()
            if reached != expected:
                print(
                    f"[NAVIGATION ERROR] 无法推进虚拟节点 {expected}: "
                    f"reached={reached}"
                )
                return False
            print(
                f"[CROSS #{cross_id:03d} VIRTUAL] reached={reached} "
                "consume_visual_cross=no; "
                f"下一个实体节点={self.current_target_node}"
            )
        return True

    def _handle_crossroad(self) -> bool:
        self.cross_event_count += 1
        cross_id = self.cross_event_count
        print(
            f"\n[CROSS #{cross_id:03d} DETECT] "
            f"logical_previous={self.previous_node or '-'} "
            f"logical_current={self.current_node} "
            f"target={self.current_target_node or '-'}"
        )
        # T1-T4 位于隧道路段中间，不是相机能够识别的实体路口。一次
        # CROSS 事件应先越过这些虚拟节点，再确认隧道另一端的实体节点。
        if not self._advance_virtual_targets(cross_id):
            return False

        expected = self.current_target_node
        if not expected:
            print("[NAVIGATION ERROR] 没有待确认的目标节点")
            return False

        reached = self._confirm_next_node(cross_id)
        if reached is None or reached != expected:
            return False

        arrived_from = self.previous_node
        if arrived_from is None:
            print(
                f"[NAVIGATION ERROR] 到达 {reached} 后缺少上一节点，"
                "无法计算路口动作"
            )
            self.chassis.stop()
            return False
        next_node = self.current_target_node
        if not next_node:
            return self.navigator.mission_complete

        action = self.topo_map.turn_action(arrived_from, reached, next_node)
        print(
            f"[CROSS #{cross_id:03d} PLAN] {arrived_from} -> {reached} -> "
            f"{next_node} action={action}"
        )
        self._set_culvert_road(None, None)
        if not self.execute_cross_action(action, cross_id, reached, next_node):
            return False

        if next_node == "START_AREA":
            # J_TOP 的转向完成后只走一小段回到出发区，随后结束任务。
            return self._run_return_start()
        self.state = "FOLLOW_LINE"
        return True

    def _run_return_start(self) -> bool:
        print(
            "[FINISH] J_TOP 转向完成，短距离返回出发区: "
            f"speed={self.args.start_forward_speed}, "
            f"distance={self.args.start_forward_distance}"
        )
        command = lambda: self.chassis.distance(
            val=self.args.start_forward_speed,
            dis=self.args.start_forward_distance,
        )
        if not self._run_action(
            f"最终返回出发区 speed={self.args.start_forward_speed} "
            f"distance={self.args.start_forward_distance}",
            command,
            self.args.start_action_timeout,
        ):
            return False
        reached = self._confirm_next_node()
        if reached != "START_AREA" or not self.navigator.mission_complete:
            print(
                f"[MISSION ERROR] 返回确认失败: reached={reached}, "
                f"complete={self.navigator.mission_complete}"
            )
            return False
        print("[MISSION] 全地图测试完成")
        self._show_oled("全地图完成\n已返回出发区", duration=4000)
        self.state = "STOP"
        return True

    def run_mission(self) -> int:
        if not self._wait_until_system_ready():
            return 2

        for remaining in range(self.args.countdown, 0, -1):
            self.chassis.stop()
            print(f"[READY] {remaining} 秒后启动...", end="\r", flush=True)
            time.sleep(1)
        print(" " * 50, end="\r")

        if not self._run_start_actions():
            self._show_oled("启动失败\n底盘动作异常", duration=4000)
            return 3

        self._show_oled("全地图测试\n巡检进行中", duration=1200)

        missed_frames = 0
        while self.running and self.state != "STOP":
            if self._culvert_active.is_set():
                # 涵洞扫描期间禁止主线程发送转向/巡线指令，避免抢占下位机任务。
                time.sleep(0.05)
                continue
            road = self.read_road()
            if road is None:
                missed_frames += 1
                if missed_frames >= self.args.max_camera_misses:
                    print("\n[ERROR] 摄像头连续取帧失败，停车")
                    self._show_oled("测试停止\n摄像头取帧失败", duration=4000)
                    return 2
                time.sleep(0.02)
                continue

            missed_frames = 0
            road_type, offset = road
            if self.state == "START_ALIGN":
                self._run_start_align(road_type, offset)
                continue

            if road_type == "CROSS":
                now = time.monotonic()
                if now - self.last_cross_time < self.args.cross_cooldown:
                    # 持续喂下位机失联保护，同时直行驶离刚处理过的路口。
                    self.chassis.drive_steer(self.args.speed, 0)
                    time.sleep(0.02)
                    continue
                self.chassis.drive_steer(0, 0)
                time.sleep(0.2)
                if not self._handle_crossroad():
                    return 4
                # 转向可能超过冷却时间，必须从动作完成后重新计时，避免同一
                # 路口的残留画面被当成下一个拓扑节点。
                self.last_cross_time = time.monotonic()
                continue

            self.follow_once(offset)
            now = time.monotonic()
            if now - self._last_line_log_time >= 0.25:
                print(
                    f"[LINE] 状态={road_type:<5} 偏差={offset:>4} "
                    f"目标={self.current_target_node or '-'} "
                    f"路口模式={self.tracker.junction_mode}",
                    end="\r",
                    flush=True,
                )
                self._last_line_log_time = now
            if self.tracker.last_key in (ord("q"), ord("Q"), 27):
                break
            time.sleep(0.02)

        return 0 if self.state == "STOP" and self.navigator.mission_complete else 1

    def close(self) -> None:
        self.running = False
        try:
            self.chassis.stop()
        finally:
            time.sleep(0.1)
            self.camera.release()
            if self._culvert_camera is not None:
                self._culvert_camera.release()
            if self._camera_thread.is_alive():
                self._camera_thread.join(timeout=1.0)
            if self._culvert_camera_thread.is_alive():
                self._culvert_camera_thread.join(timeout=1.0)
            if self._road_thread.is_alive():
                self._road_thread.join(timeout=1.0)
            if self._culvert_thread.is_alive():
                self._culvert_thread.join(timeout=1.0)
            if self._rfid_thread.is_alive():
                self._rfid_thread.join(timeout=1.0)
            if self._broadcast_thread.is_alive():
                self._broadcast_thread.join(timeout=1.0)
            self._rfid_reader.disconnect()
            self.tracker.release()
            if self._yolo is not None:
                self._release_yolo()
            cv2.destroyAllWindows()
            self.chassis.close()
            print("[SAFE] 电机已停车，摄像头、识别器和串口已释放")

    def run(self) -> int:
        try:
            self.validate_hardware()
            print(
                f"[READY] 摄像头={self.args.camera} 串口={self.args.serial_port} "
                f"巡线速度={self.args.speed}"
            )
            route, progress = calculate_plan()
            print(
                f"[FULL MAP TEST] 规划节点数={len(route)}，"
                f"需覆盖道路={progress['required_edges']}"
            )
            result = self.run_mission()
            if result != 0:
                if self.stop_requested:
                    self._show_oled("全地图测试\n用户已中止", duration=2500)
                else:
                    self._show_oled(
                        f"全地图测试停止\n错误码: {result}", duration=4000
                    )
            return result
        finally:
            self.close()


def run_live() -> int:
    robot = None
    try:
        robot = FullMapWalkTest(FullMapConfig())
        signal.signal(signal.SIGINT, robot.stop)
        signal.signal(signal.SIGTERM, robot.stop)
        return robot.run()
    except KeyboardInterrupt:
        print("\n[FULL MAP TEST] 用户中止")
        if robot is not None:
            robot._show_oled("全地图测试\n用户已中止", duration=2500)
        return 130
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # run() normally owns cleanup; this also handles constructor failures.
        if robot is not None and robot.running:
            robot.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只打印完整覆盖路线，不初始化摄像头、串口或识别器",
    )
    args = parser.parse_args()
    if args.plan_only:
        return 0 if print_plan() else 1

    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
