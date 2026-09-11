#!/usr/bin/env python3
"""
灵心巧手（LinkerHand）灵巧手最小化测试脚本
依次测试灵巧手的各手指关节 + 张开/握拳/包络抓取动作
用于排查灵巧手各关节是否有硬件/通信问题，不依赖 YOLO。
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm.linker_hand import LinkerHand

WAIT_AFTER_MOVE = 0.8   # 移动后等待时间（秒）
WAIT_AFTER_HOME = 0.5   # 归位后等待时间（秒）
ERROR_THRESHOLD = 30.0  # 关节值误差阈值（0~255）
TEST_OFFSET = 60        # 单关节测试偏移量（0~255）


def read_state(hand) -> list | None:
    """读取当前各关节位置，失败返回 None。"""
    try:
        state = hand.get_state()
    except Exception as e:
        print(f"  ❌ 读取关节位置失败: {e}")
        return None
    if state is None or len(state) != hand.joint_count:
        print(f"  ❌ 关节位置长度异常: {state}")
        return None
    values = [float(v) for v in state]
    # -1 表示手没有应答（未上电/未接线/接口号错误），必须判失败，
    # 否则目标值接近 0 的关节会被误判为通过
    if any(v < 0 for v in values):
        print(f"  ❌ 手无应答（关节读回 {[f'{v:.0f}' for v in values]}），"
              "请检查手电源、USB-485 转接器接线和 config 里的 linker_hand_modbus")
        return None
    return values


def test_joint(hand, idx: int) -> bool:
    """测试单个手指关节，返回是否全部通过。"""
    all_ok = True
    home = [float(v) for v in hand.open_pose]

    print(f"\n{'─' * 55}")
    print(f"🔧 关节{idx + 1}/{hand.joint_count}")
    print(f"{'─' * 55}")

    # 张开值往握拳方向偏移 TEST_OFFSET
    target = home.copy()
    target[idx] = max(0, min(255, home[idx] - TEST_OFFSET))

    print(f"  🎯 目标: joint{idx + 1} = {target[idx]:.0f} (张开值 {home[idx]:.0f} - {TEST_OFFSET})")
    try:
        hand.finger_move(target)
    except Exception as e:
        print(f"  ❌ 下发关节位置失败: {e}")
        return False
    time.sleep(WAIT_AFTER_MOVE)

    state = read_state(hand)
    if state is None:
        all_ok = False
    else:
        actual = state[idx]
        err = abs(actual - target[idx])
        status = "✅" if err < ERROR_THRESHOLD else "⚠️ 偏差较大"
        print(f"  📏 实际: joint{idx + 1} = {actual:.0f}  (误差 {err:.0f}) {status}")
        if err >= ERROR_THRESHOLD:
            all_ok = False

    # 回张开位
    hand.open_palm()
    time.sleep(WAIT_AFTER_HOME)

    status = "✅ 通过" if all_ok else "❌ 存在异常"
    print(f"  {status}")
    return all_ok


def test_gesture(hand, name: str, target_pose: list) -> bool:
    """执行一个手势并校验各关节是否到位，返回是否通过。"""
    print(f"\n{'─' * 55}")
    print(f"🔧 {name}")
    print(f"{'─' * 55}")
    print(f"  🎯 目标手势: {[f'{v:.0f}' for v in target_pose]}")
    try:
        hand.finger_move(target_pose)
    except Exception as e:
        print(f"  ❌ 下发手势失败: {e}")
        return False
    time.sleep(WAIT_AFTER_MOVE)

    state = read_state(hand)
    if state is None:
        return False
    errors = [abs(a - t) for a, t in zip(state, target_pose, strict=True)]
    max_err = max(errors)
    ok = max_err < ERROR_THRESHOLD
    print(f"  📏 实际: {[f'{v:.0f}' for v in state]}  (最大偏差 {max_err:.0f}) {'✅' if ok else '⚠️ 偏差较大'}")
    return ok


def main():
    print("=" * 55)
    print("  灵心巧手（LinkerHand）灵巧手测试")
    print("=" * 55)

    # ── 初始化 ──
    print("\n🔌 连接灵巧手...")
    try:
        hand = LinkerHand()
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return 1
    print(f"✅ 连接成功  型号: {hand.hand_joint}  关节数: {hand.joint_count}")

    results = {}

    try:
        # ── 张开归位 ──
        print("\n🏠 张开手掌（归位）...")
        hand.open_palm()
        time.sleep(1.0)
        state = read_state(hand)
        if state is None:
            print("❌ 手无应答，终止测试。请检查：手部 24V 电源、"
                  "USB-485 转接器是否插入本机（ls /dev/ttyUSB*）、"
                  "config.yaml 中 linker_hand_modbus 与左右手（左手 0x28/右手 0x27）是否正确")
            return 1
        print(f"   关节: {[f'{v:.0f}' for v in state]}")

        # ── 逐关节测试 ──
        for idx in range(hand.joint_count):
            results[f"关节{idx + 1}"] = test_joint(hand, idx)

        # ── 手势测试 ──
        results["张开手掌"] = test_gesture(hand, "张开手掌", hand.open_pose)
        results["完全握拳"] = test_gesture(hand, "完全握拳", hand.fist_pose)

        # ── 包络抓取测试（不要求到位，观察数值） ──
        grasp_pose = hand.grasp_by_size(0.03)
        print(f"\n{'─' * 55}")
        print("🔧 包络抓取（30mm 物体）")
        print(f"{'─' * 55}")
        print(f"  🎯 下发手势: {grasp_pose}")
        time.sleep(WAIT_AFTER_MOVE)
        state = read_state(hand)
        if state:
            print(f"  📏 实际: {[f'{v:.0f}' for v in state]}")
            results["包络抓取"] = True

        # ── set_gripper 插值测试 ──
        print(f"\n{'─' * 55}")
        print("🔧 set_gripper 插值（0.5）")
        print(f"{'─' * 55}")
        try:
            mid_pose = hand.set_gripper(0.5)
            print(f"  🎯 下发手势: {mid_pose}")
            time.sleep(WAIT_AFTER_MOVE)
            state = read_state(hand)
            if state:
                print(f"  📏 实际: {[f'{v:.0f}' for v in state]}")
                results["set_gripper插值"] = True
        except Exception as e:
            print(f"  ❌ 失败: {e}")

        # ── 传感器/状态读取测试 ──
        print(f"\n{'─' * 55}")
        print("🔧 状态与传感器读取")
        print(f"{'─' * 55}")
        try:
            print(f"  🔌 固件版本: {hand.api.get_embedded_version()}")
            print(f"  🔢 序列号: {hand.api.get_serial_number()}")
        except Exception as e:
            print(f"  ⚠️ 版本/序列号读取失败: {e}")
        try:
            print(f"  ⚡ 速度设置: {hand.get_speed()}")
        except Exception as e:
            print(f"  ⚠️ 速度读取失败: {e}")
        try:
            force = hand.get_force()
            print(f"  💪 力觉(法向/切向/方向/接近): {force}")
        except Exception as e:
            print(f"  ⚠️ 力觉读取失败: {e}")
        try:
            touch = hand.get_touch()
            print(f"  🖐️ 触觉(五指向和: {sum(touch[:5]) if touch else '无'}): {touch}")
        except Exception as e:
            print(f"  ⚠️ 触觉读取失败: {e}")
        try:
            temp = hand.get_temperature()
            print(f"  🌡️ 电机温度: {temp}")
        except Exception as e:
            print(f"  ⚠️ 温度读取失败: {e}")
        try:
            fault = hand.get_fault()
            print(f"  🚨 故障码(0为正常): {fault}")
        except Exception as e:
            print(f"  ⚠️ 故障码读取失败: {e}")

    finally:
        # ── 收尾 ──
        print(f"\n{'─' * 55}")
        print("🏠 张开手掌（归位）...")
        try:
            hand.open_palm()
            time.sleep(1)
        except Exception:
            pass
        print("🔌 断开连接...")
        try:
            hand.disconnect()
        except Exception:
            pass

    # ── 汇总 ──
    print("\n" + "=" * 55)
    print("  测试汇总")
    print("=" * 55)
    all_pass = True
    for name, ok in results.items():
        flag = "✅" if ok else "❌"
        print(f"  {flag}  {name}")
        if not ok:
            all_pass = False
    print(f"\n  结论: {'🎉 全部通过!' if all_pass else '⚠️ 存在异常关节，请检查!'}")
    print("=" * 55)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
