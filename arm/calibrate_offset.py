# Description: 设置零点，即获取一组数据，让不同本体上零点的物理状态相同
# （本仓库只支持 Piper）
#   通过 .venv/lib/python3.10/site-packages/piper_sdk/demo/V2/piper_set_joint_zero.py
#   脚本设置统一零点：要严格设置标准的零点位（J1零点校准刻度对齐为零点，J2J3当机械臂
#   失能后自然垂下放置即为零点，J6轴线与J4共线状态下J5为零点），以保证不同的本体零点
#   相同，指定相同的位姿会有相同表现。
#   Piper 下此脚本仅显示当前的 x,y,z,RZ,RY,RX 方便调试，非必需。
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arm.arm_base import Arm
import time
from utils.config_getter import get_config_value

def main():
    arm_backend = get_config_value("arm_backend", "real", raise_if_missing=False)
    if arm_backend == "sim":
        print("offset 标定仅适用于真机，arm_backend=sim 时无需也无法标定，已退出")
        return

    arm = Arm()
    arm.disable_torque()

    try:
        while True:
            print("=" * 10)
            try:
                pos, rot = arm.get_arm_pose()
                if pos:
                    for pos_i in pos:
                        print(f"  - {pos_i:.2f}")
                if rot:
                    for rot_i in rot:
                        print(f"  - {rot_i:.2f}")
            except Exception:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        arm.disconnect_arm()


if __name__ == "__main__":
    main()
