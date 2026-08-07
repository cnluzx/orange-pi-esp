"""无障碍、无涵洞的全地图行走测试。

该测试沿用 ``line_follow_test.py`` 的硬件调用方式：巡线模块直接读取
摄像头，底盘动作直接通过 ``ChassisController`` 发送，并在每个动作后轮询
下位机的 ``@`` 完成应答。地图规划只使用物理道路和 N1～N12 固定点，完成
条件为覆盖全部道路、经过所有固定点并返回出发区。
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import dataclass

import cv2

from Serial import ChassisController
from line_detect import LineTracker
from navigator import Navigator
from topo_map import TopoMap


SRC_DIR = os.path.dirname(os.path.abspath(__file__))


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
    speed: int = int(os.environ.get("FULL_MAP_LINE_SPEED", "8"))
    kp: float = float(os.environ.get("FULL_MAP_KP", "0.35"))
    kd: float = float(os.environ.get("FULL_MAP_KD", "0.10"))
    max_angle: float = float(os.environ.get("FULL_MAP_MAX_ANGLE", "30"))
    action_timeout: float = float(os.environ.get("FULL_MAP_ACTION_TIMEOUT", "10"))
    camera_startup_timeout: float = float(
        os.environ.get("CAMERA_STARTUP_TIMEOUT", "10")
    )
    start_action_timeout: float = float(
        os.environ.get("START_ACTION_TIMEOUT", "8")
    )
    start_line_required_frames: int = max(
        1, int(os.environ.get("START_LINE_STABLE_FRAMES", "30"))
    )
    start_right_turn_angle: int = int(
        os.environ.get("START_RIGHT_TURN_ANGLE", "75")
    )
    start_forward_speed: int = int(os.environ.get("START_FORWARD_SPEED", "10"))
    start_forward_distance: int = int(
        os.environ.get("START_FORWARD_DISTANCE", "130")
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
    debug: bool = _env_bool("LINE_DEBUG", True)


def build_test_navigator():
    """建立不含绿色任务点、但要求覆盖全部道路的导航器。"""
    topo_map = TopoMap()
    topo_map.build_competition_track(green_count=0)
    navigator = Navigator(topo_map)
    navigator.init_mission(
        start_node="START_AREA",
        mission_plan=topo_map.fixed_patrol_nodes,
        expected_green_count=0,
        cover_all_edges=True,
    )
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
        self.last_offset = 0
        self.last_action_error = None
        self.last_cross_time = 0.0
        self.start_line_frames = 0
        self.cross_event_count = 0
        self.action_count = 0
        self.state = "START"

        self.chassis = ChassisController(args.serial_port, args.baudrate)
        self.tracker = LineTracker(debug_mode=args.debug)

        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        try:
            self.camera = cv2.VideoCapture(args.camera, backend)
        except Exception:
            self.camera = cv2.VideoCapture(args.camera)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.topo_map, self.navigator = build_test_navigator()
        self.current_node = "START_AREA"
        self.previous_node = None
        self.current_path = None
        self.current_target_node = None
        self._sync_navigation_path()

    def validate_hardware(self) -> None:
        if not self.chassis.serial or not self.chassis.serial.is_open:
            raise RuntimeError(f"底盘串口不可用: {self.args.serial_port}")
        if not self.camera.isOpened():
            raise RuntimeError(f"巡线摄像头不可用: {self.args.camera}")

    def stop(self, *_args: object) -> None:
        self.running = False
        self.chassis.stop()

    def read_road(self) -> tuple[str, int] | None:
        ok, frame = self.camera.read()
        if not ok or frame is None or frame.size == 0:
            return None
        if self.args.flip in (-1, 0, 1):
            frame = cv2.flip(frame, self.args.flip)
        return self.tracker.get_road_info(frame)

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

    def execute_cross_action(
        self,
        action: str,
        cross_id: int,
        at_node: str,
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
                f"method=chassis.drive_steer(speed={self.args.speed}, angle=0)"
            )
            print(
                f"[ACTION #{action_id:03d} SEND] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD"
            )
            self.chassis.clear_pending_input()
            self.chassis.drive_steer(self.args.speed, 0)
            print(
                f"[ACTION #{action_id:03d} DONE] CROSS#{cross_id:03d} "
                f"node={at_node} FORWARD（非阻塞直行寻迹）"
            )
            time.sleep(0.3)
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
        if reached != expected:
            return False

        arrived_from = self.previous_node
        next_node = self.current_target_node
        if not next_node:
            return self.navigator.mission_complete

        action = self.topo_map.turn_action(arrived_from, reached, next_node)
        print(
            f"[CROSS #{cross_id:03d} PLAN] {arrived_from} -> {reached} -> "
            f"{next_node} action={action}"
        )
        if not self.execute_cross_action(action, cross_id, reached):
            return False

        if next_node == "START_AREA":
            self.state = "RETURN_START"
        else:
            self.state = "FOLLOW_LINE"
        return True

    def _run_return_start(self) -> bool:
        print(
            "[RETURN] 进入出发区: "
            f"speed={self.args.return_start_speed}, "
            f"distance={self.args.return_start_distance}"
        )
        command = lambda: self.chassis.distance(
            val=self.args.return_start_speed,
            dis=self.args.return_start_distance,
        )
        if not self._run_action(
            f"返回出发区 speed={self.args.return_start_speed} "
            f"distance={self.args.return_start_distance}",
            command,
            self.args.return_start_timeout,
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
        self.state = "STOP"
        return True

    def run_mission(self) -> int:
        deadline = time.monotonic() + self.args.camera_startup_timeout
        while self.running and time.monotonic() < deadline:
            if self.camera.isOpened():
                ok, frame = self.camera.read()
                if ok and frame is not None and frame.size:
                    # 将首帧放回摄像头流无法做到；下一轮正常读取即可。
                    break
            time.sleep(0.05)
        else:
            print("[ERROR] 巡线摄像头启动超时，立即停车")
            return 2

        if not self._run_start_actions():
            return 3

        missed_frames = 0
        while self.running and self.state != "STOP":
            road = self.read_road()
            if road is None:
                missed_frames += 1
                if missed_frames >= self.args.max_camera_misses:
                    print("\n[ERROR] 摄像头连续取帧失败，停车")
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
            print(
                f"[LINE] 状态={road_type:<5} 偏差={offset:>4} "
                f"目标={self.current_target_node or '-'}",
                end="\r",
                flush=True,
            )
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
            self.tracker.release()
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
            for remaining in range(self.args.countdown, 0, -1):
                print(f"[READY] {remaining} 秒后启动...", end="\r", flush=True)
                time.sleep(1)
            print(" " * 50, end="\r")
            return self.run_mission()
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
