# Description: 机械臂关节角示教/采集工具（纯只读，不下发任何控制帧）。
# 用途：手动拖动机械臂到目标姿态，按空格记录当前 6 关节角与末端位姿，
# 供后续分析（当前用于实测 Linker 灵巧手的平抓姿态与第 6 关节限位方向）。
# 典型流程：
#   1. 运行 arm/disable_arm.py 松掉机械臂使能（或保持任意状态均可，本工具只读）；
#   2. 手动拖动机械臂到想采集的姿态；
#   3. 在终端按空格采集一条（松开手、臂静止后再采，减少读数抖动）；
#   4. 换姿态继续采集，按 q 或 Esc（Ctrl-C 亦可）退出；
#   5. 分析 arm/hand-eye-data/joint_space_samples.jsonl（JSON Lines，可追加）。
#
# 安全性说明：本工具直接使用 C_PiperInterface_V2 连接 CAN（不实例化
# PiperBySDK，避免其构造函数中的使能/复位动作），且 ConnectPort 显式传
# piper_init=False——SDK 默认的 piper_init=True 会发送 3 条初始化查询帧，
# 传 False 后只启动接收线程。GetArmJointMsgs / GetArmGripperMsgs /
# GetArmEndPoseMsgs / GetArmStatus 均只读取 SDK 内部由反馈帧填充的缓存，
# 因此失能（disable）状态下同样可读，且全程零发送。
import datetime
import json
import os
import sys
import time
import termios
import tty

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from piper_sdk import C_PiperInterface_V2
from scipy.spatial.transform import Rotation as Rot

from arm.piper_ctrl_by_sdk import PiperBySDK  # 仅用其静态 arm_status2str，不实例化
from utils.config_getter import get_config_value

# 与 PiperBySDK 一致的反馈值缩放（0.001 度 / 0.001 毫米 -> 物理量）
FACTOR = 1000.0
JOINT_COUNT = 6
MAX_GRIPPER_ANGLE_DEG = 100
# 采集数据文件（JSON Lines，追加写入）
SAMPLE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand-eye-data", "joint_space_samples.jsonl"
)
# 读取反馈失败时的重试
READ_RETRY_TIMES = 3
READ_RETRY_INTERVAL_S = 0.2


def read_arm_state(piper: C_PiperInterface_V2) -> dict | None:
    """读取一次机械臂状态（关节角/夹爪/末端位姿/状态码），全部走反馈帧。

    数值换算与 PiperBySDK.get_arm_angles / get_arm_pose 保持一致。

    Returns:
        状态字典；连续读取失败返回 None。
    """
    for _ in range(READ_RETRY_TIMES + 1):
        try:
            joints = piper.GetArmJointMsgs()
            gripper = piper.GetArmGripperMsgs()
            end_pose = piper.GetArmEndPoseMsgs()
            status = piper.GetArmStatus().arm_status
            break
        except Exception:
            time.sleep(READ_RETRY_INTERVAL_S)
    else:
        return None

    js = joints.joint_state
    angles_deg = [
        js.joint_1 / FACTOR,
        js.joint_2 / FACTOR,
        js.joint_3 / FACTOR,
        js.joint_4 / FACTOR,
        js.joint_5 / FACTOR,
        js.joint_6 / FACTOR,
    ]
    gripper_deg = gripper.gripper_state.grippers_angle / FACTOR
    gripper_open_0to1 = float(
        np.clip(gripper_deg / MAX_GRIPPER_ANGLE_DEG, 0, 1).round(2)
    )
    ep = end_pose.end_pose
    pos_m = [
        ep.X_axis / FACTOR / 1000.0,
        ep.Y_axis / FACTOR / 1000.0,
        ep.Z_axis / FACTOR / 1000.0,
    ]
    # GetArmEndPoseMsgs 的欧拉角是 xyz 序（0.001 度），转成 zyx 序展示，
    # 与 PiperBySDK.get_ee_euler_zyx / get_arm_pose 的返回一致
    rot = Rot.from_euler(
        "xyz",
        np.array([ep.RX_axis, ep.RY_axis, ep.RZ_axis], dtype=np.float32) / FACTOR,
        degrees=True,
    )
    euler_zyx_deg = rot.as_euler("zyx", degrees=True)

    # SDK 缓存在收到首帧反馈前是零值，全零数据大概率是没连上/未上电
    if all(angle == 0.0 for angle in angles_deg) and all(p == 0.0 for p in pos_m):
        print("Warning: 反馈全为零，机械臂可能未上电或 CAN 未连通，本条丢弃")
        return None

    return {
        "joints_deg": angles_deg,
        "gripper_open_0to1": gripper_open_0to1,
        "pos_m": pos_m,
        "euler_zyx_deg": euler_zyx_deg.tolist(),
        "quat_xyzw": rot.as_quat().tolist(),
        "arm_status": f"0x{int(status.arm_status):02X}"
        f"({PiperBySDK.arm_status2str(status.arm_status)})",
        "motion_status": int(status.motion_status),
        "mode_feed": int(getattr(status, "mode_feed", -1)),
    }


def print_sample_block(index: int, record: dict) -> None:
    """打印一条采集记录，J6 单独高亮一行（当前重点分析第 6 关节）。"""
    print(f"---- 采集 #{index} @ {record['timestamp']} ----")
    print("  关节角(度):    ", [round(v, 2) for v in record["joints_deg"]])
    # ANSI 粗体黄色高亮 J6
    print(
        f"  \033[1;33mJ6 关节角(度): {record['joints_deg'][JOINT_COUNT - 1]:.2f}"
        "  <-- 当前重点分析\033[0m"
    )
    print("  末端位置(m):    ", [round(v, 3) for v in record["pos_m"]])
    print(
        "  末端姿态(度):   [RZ, RY, RX] =",
        [round(v, 2) for v in record["euler_zyx_deg"]],
    )
    print("  四元数(xyzw):   ", [round(v, 4) for v in record["quat_xyzw"]])
    print(
        "  状态:           arm=", record["arm_status"],
        " motion=0x%02X" % record["motion_status"],
        " mode_feed=", record["mode_feed"],
    )
    print(f"  已采集条数: {index}")
    sys.stdout.flush()


def count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    if not sys.stdin.isatty():
        print("错误: 本工具需要交互式终端（stdin 不是 tty），请直接在终端运行")
        sys.exit(1)

    print("=" * 62)
    print("机械臂关节角示教采集工具（只读，不下发任何控制帧/不使能/不移动）")
    print("  典型流程: 先运行 arm/disable_arm.py 松使能 -> 手动拖动机械臂")
    print("  -> 按空格采集一条 -> 换姿态继续 -> 按 q / Esc / Ctrl-C 退出")
    print(f"  数据文件(追加写入): {SAMPLE_FILE}")
    print("=" * 62)

    os.makedirs(os.path.dirname(SAMPLE_FILE), exist_ok=True)
    can_port = get_config_value("arm_port")
    print(f"连接机械臂 CAN 端口: {can_port} ...")
    piper = C_PiperInterface_V2(can_port)
    # piper_init=False: 连 SDK 默认的 3 条初始化查询帧都不发，仅启动接收线程
    piper.ConnectPort(piper_init=False)
    print("已连接（仅接收反馈帧）")

    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # 单键即读，Ctrl-C 仍以 KeyboardInterrupt 触发安全退出
    sample_count = 0
    try:
        with open(SAMPLE_FILE, "a", encoding="utf-8") as f:
            while True:
                key = os.read(fd, 1)
                if key in (b"q", b"Q", b"\x1b"):
                    print("\n退出采集")
                    break
                if key == b"\x03":  # 以防终端关闭了 ISIG
                    print("\nCtrl-C 退出")
                    break
                if key != b" ":
                    continue
                state = read_arm_state(piper)
                if state is None:
                    print("读取机械臂反馈失败，本条未记录，请重试")
                    continue
                now = datetime.datetime.now()
                record = {
                    "source": "probe_joint_angles",
                    "timestamp": now.isoformat(timespec="milliseconds"),
                    "epoch_s": round(time.time(), 3),
                    **state,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                sample_count += 1
                print_sample_block(sample_count, record)
    except KeyboardInterrupt:
        print("\nCtrl-C 退出")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        try:
            piper.DisconnectPort()
            print("机械臂 CAN 端口已断开")
        except Exception:
            pass

    print(f"本次采集 {sample_count} 条，文件累计 {count_jsonl_lines(SAMPLE_FILE)} 条")
    print(f"数据文件: {SAMPLE_FILE}")


if __name__ == "__main__":
    main()
