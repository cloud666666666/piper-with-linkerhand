# Description: Piper + Linker 灵巧手"水平包络抓取"姿态适配。
# 末端从夹爪换成灵巧手后，父类 PiperBySDK 默认的垂直朝下姿态
# （DEFAULT_DOWN_EULER_DEG_ZYX = [0, 180, 0]）不再适用：
# 抓取时掌面放平（约平行桌面），手水平地从侧面滑过去包络抓取。
# 2026-09-10 示教（arm/probe_joint_angles.py）结论：钉死 J5（固件命令
# 限位内取 -69.0，示教 -74.85 命令不可达）与 J6（6.75）两根腕关节，
# J5 差额 4.95° 由 J2/J3 抬小臂软补偿。
# move_to_flat 自带 roll-aware IK（位置 + 全姿态测地代价 + 多起点搜索），
# 解出关节角后调父类 set_arm_angles 下发，不透传父类 move_to。原因：
# 父类 _ik_cost_function 的姿态项 = tilt（工具 z）+ yaw（zyx 欧拉第一
# 分量），在 RY≈90 奇异邻域 yaw 对绕工具轴的 roll 不敏感，而平抓掌面的
# 平度主要由该 roll 决定；改用全姿态测地项 + 多起点后（四元数读法已修正，
# 见 2026-09-11 说明），对真平掌的残差：走廊中心 [0.30,0,0.17] 3.87°@3.3mm、
# z=0.16 处 2.17°@1.9mm（40 起点 Pareto）。
# 钉死机制不变：临时把实例属性 self.joint_bounds 中被钉关节的上下界都
# 改成钉死角度（本类 minimize 同样用 bounds=self.joint_bounds），求解后
# finally 恢复原 bounds。不修改父类任何代码。
#
# 相关 config 键（读不到时用代码内默认值，缺键不会崩）：
# - linker_hand_flat_euler_deg_zyx       示教基准姿态 [RZ, RY, RX]，默认
#                                        [-142.249, 81.015, 149.475]；rot_rad
#                                        将该姿态绕世界 z 旋转（只改朝向）
# - linker_hand_pin_joints               钉死关节 [[关节号(1起算), 度], ...]，
#                                        默认 [[5, -69.0], [6, 6.75]]
# - linker_hand_flat_approach_backoff_m  水平滑入前沿接近方向的退距，默认 0.10
# - linker_hand_flat_grasp_height_offset 平抓时手掌中心高于桌面的偏移，默认 0.01
#                                        （已不参与高度计算，见下）
# - linker_hand_flat_grasp_height_m      平抓工作高度（绝对值），默认 0.11
import sys
import os
import time


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from arm.piper_ctrl_by_sdk import PiperBySDK
from arm.arm_base import StepCallback
from utils.config_getter import get_config_value


class PiperFlatHandSDK(PiperBySDK):
    """Piper 机械臂"平抓姿态"控制实现（Linker 灵巧手水平包络抓取）。

    与父类 PiperBySDK 的区别仅在目标姿态与 IK：
    - 父类 move_to 未显式传欧拉角时默认垂直朝下，姿态代价对 roll 不敏感；
    - 本类 move_to_flat 默认用示教基准平抓姿态（掌面约平行桌面），自带
      位置 + 全姿态测地的代价与多起点搜索（roll-aware IK），IK 求解期间
      示教选定的腕关节（默认 J5/J6）被钉死在示教角度。
    rot_rad 语义：将示教基准姿态绕世界 z 轴旋转 rot_rad（只改朝向，
    保持 RY/RX 姿态分量不变，不再是"替换欧拉 RZ 分量"）。
    """

    # 示教基准姿态 [RZ, RY, RX]（度）：2026-09-10 示教样本2 关节角
    # （J5 取 -69.0，其余不变）的 kinpy FK zyx 欧拉；2026-09-11 三候选
    # （缩短版/真平掌/本值）对比实测后保留本值：对真平掌残差 0.88°@pos 5.0mm
    # （缩短版 0.73°@5.7mm 位置略超、真平掌 1.65°@1.4mm 残差略大）。
    # 注意 kinpy 的 Transform.rot 是 scalar-first 四元数 (w,x,y,z)，读姿态
    # 必须用 rot_mat / matrix()，不能用 scipy 的 R.from_quat（期望 xyzw，
    # 混用会让姿态差 165°+，2026-09-11 已修正此读法 bug）。
    # 与真平掌（J5=-74.85 示教姿态）相差 4.95°（J5 收回量），由 roll-aware
    # IK 用 J2/J3 软补偿。示教扫动掌面漂移 7~13°，实机可微调三分量。
    DEFAULT_FLAT_EULER_DEG_ZYX = [-142.249, 81.015, 149.475]
    # 默认钉死关节 [[关节号(1起算), 角度(度)], ...]，2026-09-10 示教：
    # J5 顶住限位、J6 自由小滚转 6.75。真机实测固件 J5 命令限位 ≈±70，
    # 示教读数 -74.85 仅断使能手拖可达、命令不可达（固件停在 -70.01 并
    # 到位超时）。2026-09-11 Pareto 分析后取 -69.0（放松 1° 换平度
    # 2.17°→0.88°、位置 1.9→5.0mm，均在抓取容差内）；若要物理顶死限位
    # 可改回 -69.9，残差 2.2°~3.9°。
    DEFAULT_PIN_JOINTS = [[5, -69.0], [6, 6.75]]
    # 平抓 IK 旋转容差（测地，弧度）：全姿态测地项天然覆盖父类 yaw 项在
    # RY≈90 奇异邻域失明的 roll 分量；量级与父类 tilt 容差（5°）一致
    FLAT_ROT_TOLERANCE_RAD = np.deg2rad(5.0)
    # 平抓 IK 位置项权重（米）：不用父类的 10mm——2026-09-11 实测同一起点集
    # 下 10mm 权重收敛到"对真平掌 0.88° @ pos 5.02mm"（越 5mm 判据线），
    # 8mm 权重收敛到 1.23° @ 3.41mm（≤1.5° 且 ≤5mm，10/10 起点一致）
    FLAT_IK_POS_WEIGHT_M = 0.008
    # 2026-09-10 示教样本2 关节角（度）：多起点 IK 的暖启动基准。
    # J5 的 -74.85 是手拖读数（命令不可达），作起点时会被钉死 bounds 裁剪
    DEFAULT_FLAT_TEACH_JOINTS = [-3.867, 90.286, -7.659, 0.764, -74.85, 6.748]
    # 多起点数：当前关节角 + 示教基准 + 8 个示教邻域随机扰动（固定种子）
    IK_MULTISTART_COUNT = 10

    def __init__(
        self,
        move_mode_end_pose: bool = False,
        debug_mode: bool = False,
        move_speed: int = 100,
        pin_wrist_on_start: bool | None = None,
    ):
        """初始化参数与父类一致（外加启动固定手腕开关），读取平抓相关 config 键。

        启动行为：父类初始化完成（使能 + 复位回 home）后，若开关打开则调用
        `pin_wrist_in_place()` 原地把被钉关节（默认 J5/J6）转到钉死值，
        使整个使用过程保持平抓构型，避免"先到位置再翻腕"。开关默认从
        config `linker_hand_pin_on_start` 读取（缺省 true）；若 home 姿态
        原地翻腕有碰周边风险，可在 config 里设 false，改为先移动到安全点
        再手动调用 `pin_wrist_in_place()`。

        Args:
            move_mode_end_pose: 是否默认使用末端位姿控制模式。
            debug_mode: 是否使用调试模式；调试模式下超时更长、夹爪力更保守。
            move_speed: 运动速度百分比，范围 1-100，默认 100（全速）。
            pin_wrist_on_start: 启动时是否原地固定手腕；`None` 表示读 config。
        """
        super().__init__(
            move_mode_end_pose=move_mode_end_pose,
            debug_mode=debug_mode,
            move_speed=move_speed,
        )

        # 示教基准平抓欧拉角（度）。缺键/留空/格式错误时退回代码内默认值。
        flat_euler = get_config_value(
            "linker_hand_flat_euler_deg_zyx",
            self.DEFAULT_FLAT_EULER_DEG_ZYX,
            raise_if_missing=False,
        )
        try:
            flat_euler = [float(v) for v in flat_euler]
            if len(flat_euler) != 3:
                raise ValueError
        except (TypeError, ValueError):
            print(
                f"Warning: linker_hand_flat_euler_deg_zyx 配置无效({flat_euler})，"
                f"使用默认 {self.DEFAULT_FLAT_EULER_DEG_ZYX}"
            )
            flat_euler = list(self.DEFAULT_FLAT_EULER_DEG_ZYX)
        self.flat_euler_deg_zyx = flat_euler

        # 钉死关节列表（示教腕关节）。格式 [[关节号(1起算), 角度(度)], ...]，
        # 缺键/留空/格式错误时退回代码内默认示教值。
        pin_joints = get_config_value(
            "linker_hand_pin_joints", None, raise_if_missing=False
        )
        if pin_joints is None:
            pin_joints = [list(pair) for pair in self.DEFAULT_PIN_JOINTS]
        try:
            pin_joints = [(int(pair[0]), float(pair[1])) for pair in pin_joints]
            if not pin_joints:
                raise ValueError
            if any(
                not 1 <= joint_no <= self.JOINT_COUNT
                for joint_no, _ in pin_joints
            ):
                raise ValueError
            if len({joint_no for joint_no, _ in pin_joints}) != len(pin_joints):
                raise ValueError  # 关节号重复
        except (TypeError, ValueError, IndexError):
            print(
                f"Warning: linker_hand_pin_joints 配置无效({pin_joints})，"
                f"使用默认 {self.DEFAULT_PIN_JOINTS}"
            )
            pin_joints = [tuple(pair) for pair in self.DEFAULT_PIN_JOINTS]
        # 超出 URDF 限位只警告、不裁剪：真正的硬保护是机械臂固件自身的
        # 限位检查（0x04），示教值来自真机实际可达姿态。
        for joint_no, angle_deg in pin_joints:
            lower_deg = float(np.rad2deg(self.joint_bounds[joint_no - 1][0]))
            upper_deg = float(np.rad2deg(self.joint_bounds[joint_no - 1][1]))
            # 0.1 度容差：避免边界值（如 -69.9 对 URDF 标称 -69.89）的舍入差误报
            if not lower_deg - 0.1 <= angle_deg <= upper_deg + 0.1:
                print(
                    f"Warning: 钉死关节 J{joint_no} 的角度 {angle_deg} 度超出 "
                    f"URDF 标称限位 [{lower_deg:.1f}, {upper_deg:.1f}] 度，"
                    "按配置值使用不裁剪（若运行时报 0x04 请在 config 中收回）"
                )
        self.pin_joints = pin_joints

        # 水平滑入前，接近起点沿接近方向离目标的退距（米）
        self.flat_approach_backoff_m = float(
            get_config_value(
                "linker_hand_flat_approach_backoff_m", 0.10, raise_if_missing=False
            )
        )
        # 平抓时手掌中心高于桌面的偏移量（米）。**已不再参与平抓高度计算**
        # （平抓工作高度统一由 linker_hand_flat_grasp_height_m 绝对值决定，
        # 见下），保留该键仅为兼容旧脚本/参考。
        self.flat_grasp_height_offset = float(
            get_config_value(
                "linker_hand_flat_grasp_height_offset",
                0.01,
                raise_if_missing=False,
            )
        )
        # 平抓工作高度（米，绝对值，唯一来源）：
        # 0.11 = 原 default_desktop_height(0.15) + offset(0.01) - 0.05；
        # 2026-09-11 实机在 test 模式验证该高度合适，抓取同步统一到该值。
        # 说明：default_desktop_height(0.15) 与当前平抓场景实际可用高度不符，
        # 故用绝对值；日后若重测桌面高度可改回"相对桌面+偏移"的语义。
        self.flat_grasp_height_m = float(
            get_config_value(
                "linker_hand_flat_grasp_height_m", 0.11, raise_if_missing=False
            )
        )

        # 启动即原地固定手腕（默认开，见类/方法文档；home 姿态翻腕有碰撞
        # 风险时在 config 里设 linker_hand_pin_on_start: false）
        if pin_wrist_on_start is None:
            pin_wrist_on_start = bool(
                get_config_value(
                    "linker_hand_pin_on_start", True, raise_if_missing=False
                )
            )
        self.pin_wrist_on_start = bool(pin_wrist_on_start)
        if self.pin_wrist_on_start:
            print(
                "=" * 62
                + f"\n启动即固定手腕：原地把 "
                f"{[(f'J{jn}', v) for jn, v in self.pin_joints]} 设为平抓构型"
                "（linker_hand_pin_on_start，可在 config 关闭）\n" + "=" * 62
            )
            self.pin_wrist_in_place()

    def pin_wrist_in_place(self, tolerance_deg: float = 0.5) -> bool:
        """原地把被钉关节转到钉死值（其余关节不动），保持平抓构型。

        只改 `self.pin_joints` 中的关节，`set_arm_angles` 下发并打印前后角度；
        若全部被钉关节都已在 `tolerance_deg` 容差内则跳过（打印说明）。

        Args:
            tolerance_deg: 判定"已在位"的角度容差，单位为度。

        Returns:
            是否成功（已在位跳过也算成功）。
        """
        current_angles_deg, _ = self.get_arm_angles()
        if current_angles_deg is None:
            print("Warning: 读取关节角失败，跳过原地固定手腕")
            return False
        target_angles_deg = [float(angle) for angle in current_angles_deg]
        for joint_no, angle_deg in self.pin_joints:
            target_angles_deg[joint_no - 1] = float(angle_deg)

        pending = [
            (joint_no, float(current_angles_deg[joint_no - 1]), float(angle_deg))
            for joint_no, angle_deg in self.pin_joints
            if abs(float(current_angles_deg[joint_no - 1]) - float(angle_deg))
            > tolerance_deg
        ]
        if not pending:
            current_pins = [
                f"J{jn}: {float(current_angles_deg[jn - 1]):.2f}°"
                for jn, _ in self.pin_joints
            ]
            print(
                f"手腕已在平抓构型（{'  '.join(current_pins)}），"
                f"容差 {tolerance_deg}°，跳过原地固定"
            )
            return True

        print(
            "原地固定手腕: "
            + "  ".join(
                f"J{jn}: {cur:.2f}° -> {target:.2f}°" for jn, cur, target in pending
            )
        )
        ok = self.set_arm_angles(target_angles_deg)
        after_angles_deg, _ = self.get_arm_angles()
        if after_angles_deg is not None:
            print(
                "  固定后: "
                + "  ".join(
                    f"J{jn}: {float(after_angles_deg[jn - 1]):.2f}°"
                    for jn, _ in self.pin_joints
                )
            )
        return bool(ok)

    # ========== 平抓 roll-aware IK ==========

    def _flat_ik_cost(
        self, joint_angles, target_pos, target_rot, chain
    ) -> float:
        """平抓 IK 代价：位置误差 + 全姿态测地误差。

        为何不沿用父类 Arm._ik_cost_function：其姿态项拆成 tilt（工具 z
        轴对齐）+ yaw（zyx 欧拉第一分量），在 RY≈90 的奇异邻域 yaw 对绕
        工具轴的 roll 不敏感，而 roll 是平抓掌面平度的主要自由度，且单
        起点 SLSQP 容易落入 roll 错误分支。改用全姿态测地 + 多起点后，
        对真平掌的残差可达 3.87°@3.3mm（走廊中心 [0.30,0,0.17]；z=0.16
        处 2.17°@1.9mm，40 起点 Pareto），腕部补偿主要来自 J2/J3，
        J4 几乎不动。
        """
        fk = chain.forward_kinematics(joint_angles)
        pos_err = np.linalg.norm(fk.pos - target_pos)
        # 全姿态测地误差（弧度），天然覆盖 tilt / yaw / roll 三个分量。
        # 注意：kinpy 的 Transform.rot 是 scalar-first 四元数 (w,x,y,z)，
        # scipy 的 R.from_quat 期望 (x,y,z,w)——必须走 rot_mat，混用会让
        # 姿态差 165°+（2026-09-11 修正）。
        rot_err = (R.from_matrix(fk.rot_mat).inv() * target_rot).magnitude()
        return (pos_err / self.FLAT_IK_POS_WEIGHT_M) ** 2 + (
            rot_err / self.FLAT_ROT_TOLERANCE_RAD
        ) ** 2

    def _solve_flat_ik(self, pos, target_rot) -> list[float] | None:
        """多起点求解平抓 IK（钉死关节经 self.joint_bounds 生效）。

        起点：当前关节角（与父类 move_to 一致，保证连续性）、示教样本2
        关节角、以及 IK_MULTISTART_COUNT-2 个示教邻域随机扰动（固定种子，
        结果可复现）。SLSQP 的 lb==ub 定界保证被钉关节恒为钉死值；起点
        先夹回 bounds（钉死关节自动落到钉死值），避免 scipy 裁剪告警。

        Args:
            pos: 目标位置 `[x, y, z]`，单位为米。
            target_rot: 目标姿态（scipy Rotation）。

        Returns:
            关节角列表（度）；全部起点求解失败返回 None。
        """
        current_angles_deg, _ = self.get_arm_angles()
        if current_angles_deg is None:
            current_angles_deg = [0.0] * self.JOINT_COUNT
        starts = [list(current_angles_deg), list(self.DEFAULT_FLAT_TEACH_JOINTS)]
        rng = np.random.default_rng(0)
        teach = np.array(self.DEFAULT_FLAT_TEACH_JOINTS, dtype=float)
        for _ in range(self.IK_MULTISTART_COUNT - len(starts)):
            starts.append((teach + rng.uniform(-20.0, 20.0, self.JOINT_COUNT)).tolist())

        lower = np.array([b[0] for b in self.joint_bounds])
        upper = np.array([b[1] for b in self.joint_bounds])
        target_pos = np.asarray(pos, dtype=float)

        best_cost = None
        best_x = None
        for start_deg in starts:
            # 起点夹回 bounds；被钉关节的 bounds 为 (钉死值, 钉死值)，
            # 夹剪后起点自然落在钉死值上
            x0 = np.clip(np.deg2rad(np.array(start_deg, dtype=float)), lower, upper)
            try:
                result = minimize(
                    self._flat_ik_cost,
                    x0=x0,
                    args=(target_pos, target_rot, self.chain),
                    method="SLSQP",
                    bounds=self.joint_bounds,
                )
            except Exception:
                continue
            if result.success and (best_cost is None or result.fun < best_cost):
                best_cost = result.fun
                best_x = result.x
        if best_x is None:
            return None
        return np.rad2deg(best_x).tolist()

    def safe_disable(
        self,
        re_enable: bool = False,
        disconnect_port: bool = False,
        settle_s: float = 0.5,
    ) -> bool:
        """按 arm/disable_arm.py 的安全流程收尾：摆到安全姿态后再失能。

        `arm/disable_arm.py` 走的是父类 `disconnect_arm()`（piper_ctrl_by_sdk.py
        401-414）：`reset()`（清错 + 回 home）-> `move_to_home(safe_pos=True)`
        （关节姿态 [0,0,0,0,25,0]）-> 等待 0.5s -> `disable_torque()` ->
        `DisconnectPort()`。本方法复用同一安全姿态，只是把最后两步拆开：
        - **失能必须在安全姿态下进行**：平抓钉死构型（J5=-69°、折叠+平掌）
          突然失能会因重力下坠，故此处按 disable_arm 的 J5=+25° 姿态执行
          （与保持平抓手腕的需求冲突时以安全姿态为准）。
        - `re_enable=True`：先重新使能（用于采集结束后已失能/被手拖过的
          情况），再走安全姿态；`disconnect_port=True`：收尾关闭 CAN 端口
          （等价父类 disconnect_arm 的最后一步，但不触发它对已失能机械臂
          发送回零指令后的无效到位等待）。

        Args:
            re_enable: 是否先重新使能（当前已失能/被手拖过时用）。
            disconnect_port: 失能后是否关闭 CAN 端口。
            settle_s: 摆到安全姿态后、失能前的等待秒数（与父类一致 0.5s）。

        Returns:
            安全姿态是否确认到位（未到位仍会失能，但会打印警告）。
        """
        if re_enable:
            print("重新使能机械臂，准备摆到可安全失能的姿态（请注意周围安全）...")
            self.enable_torque()
        print(
            "移动到可安全失能的姿态（关节 J5=+25°，与 arm/disable_arm.py 一致）..."
        )
        ok = self.move_to_home(safe_pos=True)
        if not ok:
            print("Warning: 安全姿态未确认到位，请扶稳机械臂后再失能")
        time.sleep(settle_s)
        self.disable_torque()
        if disconnect_port:
            self.piper.DisconnectPort()
            print("Arm disconnected")
        return ok

    def move_to_flat(
        self,
        pos: list[float],
        rot_rad: float | int | None = None,
        euler_angles_deg_zyx: list[float] | None = None,
        step_callback: StepCallback | None = None,
        gripper_open_0to1: float | int | None = None,
    ) -> bool:
        """以平抓姿态（示教腕关节钉死、掌面约平行桌面）移动末端到目标位置。

        与父类 move_to 的区别：
        - IK 用本类 roll-aware 代价（位置 + 全姿态测地）多起点求解，
          随后调父类 set_arm_angles 下发，保留插值/到位等待/超时语义
          （绕开父类 move_to 的原因见 _flat_ik_cost 文档）；
        - 未显式传 euler_angles_deg_zyx 时，目标姿态为示教基准姿态
          （self.flat_euler_deg_zyx），rot_rad 将其绕世界 z 轴旋转
          （只改朝向，RY/RX 姿态分量保持示教值）；
        - IK 求解期间 self.pin_joints 中各关节的 bounds 被临时改为
          (示教角度, 示教角度)，钉死在示教值，求解结束后恢复原 bounds。

        Args:
            pos: 目标位置 `[x, y, z]`，单位为米。
            rot_rad: 末端绕世界 z 轴的相对旋转角，单位为弧度；作用于
                示教基准姿态之上，未显式传 euler_angles_deg_zyx 时生效。
            euler_angles_deg_zyx: 显式指定的平抓目标欧拉角 `[RZ, RY, RX]`（度）。
            step_callback: 插值步回调。
            gripper_open_0to1: 夹爪开合程度（换成灵巧手后通常不传）。

        Returns:
            移动是否成功。
        """
        if len(pos) != 3:
            raise ValueError("位置参数格式错误，应该是[x, y, z]")

        # 构建目标旋转：显式欧拉直接用；否则示教基准姿态绕世界 z 转 rot_rad
        if euler_angles_deg_zyx is not None:
            target_rot = R.from_euler("zyx", euler_angles_deg_zyx, degrees=True)
        else:
            target_rot = R.from_euler("zyx", self.flat_euler_deg_zyx, degrees=True)
            if rot_rad is not None:
                target_rot = R.from_rotvec([0.0, 0.0, float(rot_rad)]) * target_rot

        # 临时钉死示教关节：把这些关节的 bounds 上下界都改成该角度，
        # 本类 minimize 用的同样是实例属性 bounds=self.joint_bounds，
        # 因此 IK 求解期间这些关节固定在示教值，finally 恢复全部原 bounds。
        original_bounds = {}
        for joint_no, angle_deg in self.pin_joints:
            index = joint_no - 1
            original_bounds[index] = self.joint_bounds[index]
            pin_rad = float(np.deg2rad(angle_deg))
            self.joint_bounds[index] = (pin_rad, pin_rad)
        try:
            angles_deg = self._solve_flat_ik(pos, target_rot)
            if angles_deg is None:
                print("平抓逆运动学不收敛，无法到达指定位置")
                return False
            return self.set_arm_angles(
                angles_deg,
                gripper_open_0to1=gripper_open_0to1,
                step_callback=step_callback,
            )
        finally:
            for index, bound in original_bounds.items():
                self.joint_bounds[index] = bound


if __name__ == "__main__":
    # 自测：实例化（启动已原地固定 J5/J6 为平抓构型）-> 直接走平抓走廊
    # 中心测试点，验证钉死关节与 J2/J3 软补偿量 -> 回默认姿态。
    # 会动真机，请确认周围安全后再运行！
    import time

    arm: PiperFlatHandSDK = PiperFlatHandSDK(debug_mode=False)
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    print("末端位姿:", np.array(arm.get_arm_pose()).round(2).tolist())
    print("示教基准姿态(度):", arm.flat_euler_deg_zyx)
    print("钉死关节:", [(f"J{joint_no}", angle) for joint_no, angle in arm.pin_joints])
    print(
        "平抓工作高度(米):", arm.flat_grasp_height_m,
        "（linker_hand_flat_grasp_height_m，绝对值；桌面默认",
        arm.desktop_height, "）",
    )

    # 启动时手腕已是平抓构型，直接走平抓测试点（不再先垂直移动到位置再翻腕）：
    # 走廊中心 = desktop + 高度偏移（默认 0.01 → z=0.16），方位角 0° 对应
    # 朝向 = 示教基准绕世界 z 转 +4.2°
    flat_pos = [0.3, 0.0, arm.flat_grasp_height_m]
    res = arm.move_to_flat(flat_pos, rot_rad=np.deg2rad(4.2))
    time.sleep(2)
    print(f"move_to_flat 返回: {res}  测试点: {flat_pos}")
    angles_deg, _ = arm.get_arm_angles()
    if angles_deg is not None:
        for joint_no, target_deg in arm.pin_joints:
            print(
                f"J{joint_no}: 实际 {angles_deg[joint_no - 1]:.2f} 度"
                f" / 钉死目标 {target_deg} 度"
            )
        for name, index in (("J2", 1), ("J3", 2), ("J4", 3)):
            delta = angles_deg[index] - arm.DEFAULT_FLAT_TEACH_JOINTS[index]
            print(
                f"{name}: 实际 {angles_deg[index]:.2f} 度"
                f" / 示教样本2 {arm.DEFAULT_FLAT_TEACH_JOINTS[index]:.2f} 度"
                f" (Δ{delta:+.2f} 度，软补偿量)"
            )
    print("平抓后末端位姿:", np.array(arm.get_arm_pose()).round(2).tolist())

    # 回默认姿态
    arm.move_to_home()
    time.sleep(1)
    print("关节角度:", arm.get_arm_angles())
    arm.disconnect_arm()
