# Description: 灵心巧手（LinkerHand）灵巧手控制封装。
# 对官方 linkerhand-python-sdk（https://github.com/linker-bot/linkerhand-python-sdk）
# 的 LinkerHandApi 做一层封装：
# - 从 config.yaml 读取手型、通讯等配置；
# - 提供张开、握拳、按物体宽度包络抓取等高层动作；
# - 提供与夹爪语义一致的 set_gripper(open_0to1) 接口。
#
# 通讯方式按型号区分：O6 手为 Modbus RTU 串口（115200 8N1，需 pymodbus），
# 其余型号走 CAN。CAN 模式下 SDK 会读取其仓库内 LinkerHand/config/setting.yaml
# 中左右手、型号等配置，且 Linux 下需要用其中的 PASSWORD 以 sudo 打开 CAN 口。
import os
import sys
from typing import Any

import numpy as np

from utils.config_getter import get_config_value

# 各型号关节数量，与官方 position 与手指关节对照表一致
HAND_JOINT_COUNTS = {
    "O6": 6,
    "L6": 6,
    "L7": 7,
    "L10": 10,
    "G20": 20,
    "L20": 20,
    "L21": 25,
    "L25": 25,
}

# 内置默认手势（关节值范围 0~255），来源为官方示例/配置，可在 config.yaml 中覆盖。
# 本机装的是**右手**（config linker_hand_type: right → SDK CAN id = 0x27），
# 故 O6 默认值取右手/通用表；SDK 不做镜像，姿态值须由调用方按手别提供
# （见 LinkerHand/linker_hand_api.py:21-24 只按 hand_type 选 CAN id）。
# L10 张开：example/L10/gesture/linker_hand_open_palm.py
# O6 关节值语义：小=弯曲/靠掌心，大=伸直/远离掌心，与官方 relax()/fist() 一致
DEFAULT_OPEN_PALM_POSES = {
    # O6 张开：LinkerHand/config/O6_positions.yaml 的 RIGHT_HAND「张开」
    # = [255, 70, 255, 255, 255, 255]（左手版是 index1=179，勿混用）
    "O6": [255, 70, 255, 255, 255, 255],
    "L10": [255, 70, 255, 255, 255, 255, 255, 255, 255, 255],
}
# L10 握拳：example/L10/gesture/linker_hand_fist.py
# O6 握拳：**官方 yaml 未定义右手握拳**（O6/L6_positions.yaml 的 RIGHT_HAND
# 只有「张开」，L10 那样左右相同的手势表也没有握拳），故取官方通用示例
# example/O6/linker_hand_o6.py 的 POSE 表「握拳」= [102, 18, 0, 0, 0, 0]
# （该示例支持 --hand_type right，与手别无关）。
# 备选：左手专用值 [67, 151, 0, 0, 0, 0]（O6_positions.yaml LEFT_HAND 握拳）。
# **待实机验证**：右手握拳若手感/包络不对，优先微调 index0/index1
# （右手的张开 index1 = 70，而左手是 179，说明该自由度按手别取值），
# 用 config linker_hand_fist_pose 覆盖即可，无需改代码。
DEFAULT_FIST_POSES = {
    "O6": [102, 18, 0, 0, 0, 0],
    "L10": [80] * 10,
}


def _load_api_cls(sdk_path: str | None):
    """加载灵巧手 SDK，支持从 config 指定的仓库根目录导入。

    Args:
        sdk_path: linkerhand-python-sdk 仓库根目录；`None` 表示 SDK 已在
            python 环境中（已加入 PYTHONPATH）。
    """
    if sdk_path:
        sdk_path = os.path.expanduser(sdk_path)
        if not os.path.isdir(sdk_path):
            raise FileNotFoundError(f"灵巧手 SDK 路径不存在: {sdk_path}")
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
    from LinkerHand.linker_hand_api import LinkerHandApi

    return LinkerHandApi


class LinkerHand:
    """灵心巧手（LinkerHand）灵巧手控制封装。

    初始化时按 config.yaml 中的 `linker_hand_*` 配置项连接灵巧手，
    常用动作：

    - `open_palm()`：手掌张开（张开手势）
    - `close_fist()`：完全握拳（握拳手势）
    - `grasp_by_size(object_width_m)`：按物体宽度包络抓取
    - `check_grasped()`：通过触觉判定是否已抓到物体
    - `set_gripper(open_0to1)`：与夹爪语义一致的接口，
      1 为完全张开、0 为完全握拳，中间值插值
    """

    def __init__(self):
        hand_joint = str(get_config_value("linker_hand_joint")).upper()
        self.hand_joint = hand_joint
        if hand_joint not in HAND_JOINT_COUNTS:
            raise ValueError(
                f"不支持的灵巧手型号: {hand_joint}, "
                f"可选: {list(HAND_JOINT_COUNTS)}"
            )
        self.joint_count = HAND_JOINT_COUNTS[hand_joint]

        sdk_path = get_config_value(
            "linker_hand_sdk_path", "", raise_if_missing=False
        ) or None
        modbus = get_config_value(
            "linker_hand_modbus", "None", raise_if_missing=False
        ) or "None"
        # Modbus（RS485）模式：SDK 走 core/rs485/linker_hand_o6_rs485.py——
        # - 站号 = linker_hand_api.py 由 hand_type 推得（右手 0x27 / 左手 0x28），
        #   即本类传下去的 hand_type 同时决定 CAN id 与 Modbus 从站地址；
        # - 波特率 115200 由 SDK 在 api 构造里硬编码（无配置项）；
        # - 关节值/速度/力矩均为 0~255，**与 CAN 路径一致、无需换算**
        #   （驱动 set_joint_positions 显式校验 0-255），语义同为"小=弯曲"；
        # - 发帧：每帧一次 Modbus FC16 批量写 6 个角度寄存器，帧间隔 ≥30ms
        #   （FRAME_GAP），与 CAN 的"单帧 pose"一一对应，故上层 API 无差异；
        #   驱动另带便捷函数 relax()=全 255、fist()=全 0（可作为握拳的备选值）。
        # Modbus 模式下 can 参数被忽略（其余型号才走 CAN）。
        self.is_modbus = modbus != "None"
        api_cls = _load_api_cls(sdk_path)
        self.api = api_cls(
            hand_type=get_config_value("linker_hand_type"),
            hand_joint=hand_joint,
            modbus=modbus,
            can=get_config_value("linker_hand_can", "can0", raise_if_missing=False),
        )

        self.speed = get_config_value(
            "linker_hand_speed", [120, 250, 250, 250, 250], raise_if_missing=False
        )
        self.torque = get_config_value(
            "linker_hand_torque", [200] * 5, raise_if_missing=False
        )
        self.touch_threshold = float(
            get_config_value("linker_hand_touch_threshold", 0, raise_if_missing=False)
        )
        self.grasp_size_scale = float(
            get_config_value("linker_hand_grasp_size_scale", 2.0, raise_if_missing=False)
        )

        # 张开/握拳手势，config 中留空则使用内置默认；记录来源便于外部脚本打印
        open_pose_cfg = get_config_value(
            "linker_hand_open_palm_pose", [], raise_if_missing=False
        )
        self.open_pose = (
            open_pose_cfg
            or DEFAULT_OPEN_PALM_POSES.get(hand_joint)
            or [255] * self.joint_count
        )
        self.open_pose_from_config = bool(open_pose_cfg)
        fist_pose_cfg = get_config_value(
            "linker_hand_fist_pose", [], raise_if_missing=False
        )
        self.fist_pose = (
            fist_pose_cfg
            or DEFAULT_FIST_POSES.get(hand_joint)
            or [80] * self.joint_count
        )
        self.fist_pose_from_config = bool(fist_pose_cfg)
        if len(self.open_pose) != self.joint_count or len(self.fist_pose) != self.joint_count:
            raise ValueError(
                f"灵巧手手势关节数不匹配: {hand_joint} 需要 {self.joint_count} 个，"
                f"张开手势 {len(self.open_pose)} 个，握拳手势 {len(self.fist_pose)} 个"
            )
        if hand_joint not in DEFAULT_OPEN_PALM_POSES:
            print(
                f"Warning: {hand_joint} 无内置默认手势，使用通用手势"
                "（张开全 255 / 握拳全 80），请按实际手型在 config.yaml 中调整"
                " linker_hand_open_palm_pose / linker_hand_fist_pose。"
            )

        if hand_joint == "L7" and len(self.speed) != 7:
            raise ValueError(f"L7 的 linker_hand_speed 需要 7 个速度值，实际 {len(self.speed)} 个")
        if hand_joint in ("O6", "L6"):
            if len(self.speed) != 6:
                raise ValueError(
                    f"{hand_joint} 的 linker_hand_speed 需要 6 个速度值，实际 {len(self.speed)} 个"
                )
            if len(self.torque) != 6:
                raise ValueError(
                    f"{hand_joint} 的 linker_hand_torque 需要 6 个力矩值，实际 {len(self.torque)} 个"
                )

        self.api.set_speed(speed=self.speed)
        self.api.set_torque(torque=self.torque)
        self.open_palm()

    # ========== SDK 底层接口透传 ==========

    def finger_move(self, pose: list[float | int]) -> Any:
        """下发手指关节位置，关节值范围 0~255，长度须与型号关节数一致。"""
        return self.api.finger_move(pose=pose)

    def set_speed(self, speed: list[float | int]) -> Any:
        """设置手指运动速度，L7 需 7 个值，其他型号 5 个。"""
        return self.api.set_speed(speed=speed)

    def set_torque(self, torque: list[float | int]) -> Any:
        """设置手指最大力矩。"""
        return self.api.set_torque(torque=torque)

    def get_state(self) -> Any:
        """读取当前各关节位置。"""
        return self.api.get_state()

    def get_speed(self) -> Any:
        """读取当前各关节速度设置。"""
        return self.api.get_speed()

    def get_force(self) -> Any:
        """读取法向力、切向力等力觉数据（O6 等无该接口的型号返回 None）。"""
        try:
            return self.api.get_force()
        except AttributeError:
            return None

    def get_touch(self) -> Any:
        """读取五指触觉数据（无触觉传感器时为 -1）。"""
        return self.api.get_touch()

    def get_temperature(self) -> Any:
        """读取各电机温度。"""
        return self.api.get_temperature()

    def get_fault(self) -> Any:
        """读取电机故障码。"""
        return self.api.get_fault()

    # ========== 高层动作 ==========

    def open_palm(self) -> list[int]:
        """手掌张开。"""
        pose = [int(v) for v in self.open_pose]
        self.api.finger_move(pose=pose)
        return pose

    def close_fist(self) -> list[int]:
        """完全握拳。"""
        pose = [int(v) for v in self.fist_pose]
        self.api.finger_move(pose=pose)
        return pose

    def grasp_by_size(self, object_width_m: float) -> list[int]:
        """按物体宽度包络抓取。

        参照官方 example/L10/grab/dynamic_grasping.py：四指收紧，
        拇指根部开度随物体宽度增大。其余型号在张开与握拳手势之间按宽度插值，
        插值方向为物体越大手指收得越少。

        Args:
            object_width_m: 物体近似宽度，单位米。

        Returns:
            实际下发的关节位置列表。
        """
        mm = max(0.0, float(object_width_m)) * 1000 * self.grasp_size_scale
        if self.hand_joint == "L10":
            # 官方动态抓取公式：拇指根部 60 + mm*2，四指根部 25 + mm*2
            pose = [255, 70, 255, 255, 255, 255, 255, 255, 255, 120]
            pose[0] = int(min(255, 60 + mm))
            pose[1] = 60
            pose[2] = pose[3] = pose[4] = pose[5] = int(min(255, 25 + mm))
            pose[9] = 58
        else:
            # 通用插值：物体越大越接近张开手势
            fraction = float(np.clip(1 - mm / 255, 0, 1))
            pose = [
                int(round(o + (f - o) * fraction))
                for o, f in zip(self.open_pose, self.fist_pose, strict=True)
            ]
        self.api.finger_move(pose=pose)
        return pose

    def check_grasped(self) -> bool:
        """通过触觉判定是否已抓到物体。

        五个手指触觉值之和达到 `linker_hand_touch_threshold` 认为已抓到；
        阈值为 0 时不检查，恒认为成功。

        Returns:
            是否判定为已抓到物体。
        """
        if self.touch_threshold <= 0:
            return True
        touch = self.get_touch()
        if not touch:
            return True
        fingers = [v for v in touch[:5] if isinstance(v, (int, float))]
        if len(fingers) < 5:
            return True
        if any(v < 0 for v in fingers):
            print("Warning: 触觉数据无效（可能未安装触觉传感器），跳过抓取判定")
            return True
        touch_sum = sum(fingers)
        print(f"触觉总和: {touch_sum}, 阈值: {self.touch_threshold}")
        return touch_sum >= self.touch_threshold

    def set_gripper(self, open_0to1: float) -> list[int]:
        """与夹爪语义一致的接口：1 为完全张开，0 为完全握拳，中间值插值。

        Args:
            open_0to1: 开合程度，范围为 `[0, 1]`，越大越开。

        Returns:
            实际下发的关节位置列表。
        """
        open_0to1 = float(np.clip(open_0to1, 0, 1))
        pose = [
            int(round(o + (f - o) * (1 - open_0to1)))
            for o, f in zip(self.open_pose, self.fist_pose, strict=True)
        ]
        self.api.finger_move(pose=pose)
        return pose

    def disconnect(self):
        """关闭串口/CAN 等清理。"""
        if self.is_modbus:
            # Modbus 模式直接关闭 pymodbus 串口客户端
            try:
                self.api.hand.close()
            except Exception:
                pass
            return
        try:
            self.api.close_can()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    hand = LinkerHand()
    print("张开手掌")
    hand.open_palm()
    time.sleep(2)
    print("抓取 30mm 宽物体")
    hand.grasp_by_size(0.03)
    time.sleep(2)
    print("完全握拳")
    hand.close_fist()
    time.sleep(2)
    hand.open_palm()
    hand.disconnect()
