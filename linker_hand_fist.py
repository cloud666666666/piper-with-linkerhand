#!/usr/bin/env python3
"""
灵心巧手（LinkerHand）"握拳"单发脚本（实机验证用）。

用途：只下发一次握拳手势，用于快速确认通信链路（右手 + Modbus 串口或 CAN）
与握拳值是否符合预期；复用 arm/linker_hand.py 封装，因此自动吃 config：
- linker_hand_type（本机右手 → Modbus 从站 0x27 / CAN id 0x27）
- linker_hand_modbus（如 /dev/ttyUSB0 即走 RS485 Modbus，留空/None 走 CAN）
- linker_hand_fist_pose（覆盖握拳值；留空用内置默认）
脚本会打印本次实际下发的姿态值与来源（config 覆盖 or 内置默认）。

**待实机验证**：O6 右手握拳值官方 SDK 未定义（O6/L6 的 RIGHT_HAND 只定义了
"张开"），当前内置默认取官方通用示例值 [102, 18, 0, 0, 0, 0]；另有候选
（左手专用 [67, 151, 0, 0, 0, 0]、驱动 fist() 的全 0）。若手感/包络不对，
在 config.yaml 用 linker_hand_fist_pose 覆盖（优先微调第 1、2 个分量），
无需改代码；本脚本会回读关节位置，便于判断手指是否到位。

前提（Modbus 模式）：USB-485 转接器已插好且 config 里 linker_hand_modbus
填的是真实设备名（`ls /dev/serial/by-id/` 推荐，`ls /dev/ttyUSB*` / dmesg 辅助），
波特率 115200 8N1 由 SDK 固定；串口权限需 dialout 组或 sudo。

用法：
    python3 linker_hand_fist.py                # 握拳并保持 3 秒
    python3 linker_hand_fist.py --hold 5       # 保持 5 秒
    python3 linker_hand_fist.py --repeat 3     # 连发 3 次（每次间隔 --hold）
    python3 linker_hand_fist.py --hold 0       # 发完立刻退出

注意：只有显式运行本脚本才会发帧，import 本模块不会做任何通信。
构建封装时 LinkerHand 内部会先下发一次张开（封装既有行为），随后本脚本再
下发握拳，形成张开→握拳的过渡，属预期。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm.linker_hand import LinkerHand


def read_state(hand: LinkerHand) -> list | None:
    """读取当前各关节位置用于回显，失败返回 None（不中断流程）。"""
    try:
        state = hand.get_state()
    except Exception as e:
        print(f"  读取关节位置失败: {e}")
        return None
    if state is None or len(state) != hand.joint_count:
        print(f"  关节位置长度异常: {state}")
        return None
    return [int(v) for v in state]


def main():
    parser = argparse.ArgumentParser(description="灵心巧手握拳（单次下发，实机验证）")
    parser.add_argument("--hold", type=float, default=3.0, help="下发后保持秒数（默认 3）")
    parser.add_argument("--repeat", type=int, default=1, help="重复次数（默认 1）")
    args = parser.parse_args()
    repeats = max(1, args.repeat)

    hand = LinkerHand()
    source = (
        "config（linker_hand_fist_pose）"
        if hand.fist_pose_from_config
        else f"内置默认（{hand.hand_joint} 右手，SDK 未定义右手握拳、取官方通用示例值，待实机验证）"
    )
    print(f"握拳姿态: {[int(v) for v in hand.fist_pose]}  来源: {source}")
    print(f"通信链路: {'Modbus 串口' if hand.is_modbus else 'CAN'}")

    try:
        for index in range(repeats):
            pose = hand.close_fist()
            print(f"[{index + 1}/{repeats}] 已下发握拳: {pose}，保持 {args.hold:.1f}s")
            time.sleep(max(0.0, args.hold))
            state = read_state(hand)
            if state is not None:
                print(f"  回读关节位置: {state}")
    except KeyboardInterrupt:
        print("\nCtrl-C：中断本次测试")
    finally:
        hand.disconnect()
        print("已断开灵巧手连接")


if __name__ == "__main__":
    main()
