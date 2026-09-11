# Description: 平抓姿态 2D 手眼标定（Linker 灵巧手水平包络抓取专用）。
# 与垂直模式 arm/calibrate_handeye_2d.py（已提交，保持不动）的区别：
# - 垂直模式假设"法兰 xy == 末端 xy"（夹爪垂直桌面），标定结果存
#   arm/hand-eye-data/2d_homography.npy，供 arm_base.pixel2pos 使用；
# - 平抓模式下手掌沿工具 z 前伸出臂长 L，且偏移方向 = 接近朝向 =
#   抓取点方位角 + φ0（config linker_hand_flat_heading_offset_deg），
#   随点位旋转 —— "像素 -> 法兰"不再是单应性，需联合拟合
#   H（像素 -> 掌心）+ L（臂长），实现见 arm/flat_handeye.py；
#   结果存 arm/hand-eye-data/2d_homography_flat.npz，供
#   classification/catch_with_linker_hand_flat.py 使用。
# - 本脚本不读写垂直模式的任何标定文件。
#
# 使用方法（平抓标定，mode=calibrate）：
# 0. PiperFlatHandSDK 启动时会先回 home 再原地固定 J5/J6（config
#    linker_hand_pin_on_start），因此拖动起点就是平抓构型；脚本启动时会
#    打印一条手腕状态自检（见 report_pin_state），若该开关被关掉会提示用
#    arm.pin_wrist_in_place() 手动固定。
#    本脚本所有"动力移动"（仅测试模式）都走 move_to_flat，全程保持平抓；
#    采集阶段靠断使能手拖、不涉及动力移动，也不存在"先到位置再转姿态"的中转。
# 1. 运行本脚本：先回零并使能、原地固定手腕，随后**摆到安全姿态
#    （关节 J5=+25°，与 arm/disable_arm.py 的安全流程一致）再失能**供手拖，
#    避免折叠+平掌构型突然失能因重力下坠；采集结束后同样会重新使能摆回
#    安全姿态、再失能并断开 CAN 端口；
# 2. 手动拖动机械臂摆出平抓姿态（J5 停在手拖止点 ≈-74.9 或电动钉值 -69.0
#    均可，J6 ≈ 6.75°——均不作卡口）；**校验只卡一项：工具 z 离水平面 ≤11°**
#    （掌面须贴平，掌心才会落在桌面平面、单应性才成立），J5/J6 只打印诊断；
#    工具 z 超差时会在**实时画面**上提示并等键：按 y 仍记录 / 其他键跳过
#    （5 秒超时自动跳过）——确认走 utils/cv2_display 的 poll_key 通道，终端
#    stdin 在远程画面模式下收不到输入；config
#    linker_hand_flat_pose_check_enabled 置 false 可完全关闭该校验；
# 3. 让"掌心（抓取接触点）"对准画面里的目标点（鼠标点选），不要用指尖或法兰；
#    建议在掌心贴一个小标记，便于目视对齐；
# 4. 按空格记录（同时记录像素点与当拍法兰位姿），换目标点重复 ≥6 条；点位要
#    铺开方位角范围（建议跨度 ≥±20°，否则臂长 L 与单应平移项不可分、L 不可辨识）；
# 5. 按 ESC 结束采集 -> 自动拟合，终端打印 H、L、RMS 与"纯单应 L=0"对照残差；
#    拟合器会在以下情况醒目报警：点数 <8（不可靠）、臂长 L 命中搜索边界
#    （0/0.25m，即 L 不可辨识，建议补点/铺开方位角/查坏点），并打印最差
#    2 个点的序号与残差；若确认某点是坏点，可剔除后重拟合：
#        python3 arm/calibrate_arm_hand.py --mode fit --drop-points 2,5
#    （序号 1 起算、按采集顺序；剔除后的点集会写回原始 npy；剔除后 <6 点拒绝）
#    注意：n <15 时最终单应用全点最小二乘（小样本用 RANSAC 会挑最小集精确
#    拟合、掩盖坏点），n ≥15 才用 RANSAC；
# 6. 结果存 arm/hand-eye-data/2d_homography_flat.npz；mode=test 点击画面可
#    验证映射（会真的移动机械臂，走平抓姿态）。
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from camera.camera_api import Camera
from utils.config_getter import get_config_value
from utils.cv2_display import (
    show_image,
    poll_key,
    set_mouse_callback,
    destroy_all_windows,
)
from arm.piper_ctrl_by_sdk_flat_hand import PiperFlatHandSDK
from arm.flat_handeye import fit_flat_mapping, load_mapping, save_mapping
import argparse
import cv2
import numpy as np
import time
import threading


# ---- 平抓姿态校验（2026-09-11 按用户意见简化：只卡工具 z 水平度）----
# 不在 J5/J6 上设卡口，物理依据（均为本仓库离线实验结论）：
# - 平抓 IK 只解 J1~J4，J5/J6 在抓取时恒为钉值，标定记录的是"法兰 xy"，
#   掌心位置由法兰 + 臂长 L 推出，与 J5/J6 的微小差异无关；
# - J6 是纯绕工具 z 的滚转、掌心就在该轴上，对掌心位置影响严格为 0；
# - J5 在两族之间（手拖止点 ≈-74.9 / 电动钉值 -69.0）掌心位置只差 ≈2mm
#   （工具 z 夹角 5.90°：水平投影因子 L·Δcos≈1.0mm + 法兰位移≈0.9mm，
#   L=0.1m），与标定 RMS（1~3mm）同量级，且固件命令限位不允许消除该差。
# 唯一硬条件 = 掌面处于平抓族（工具 z 近水平）：掌面贴平掌心才落在桌面
# 平面附近，单应（像素 -> 桌面平面）才成立。
# 该族实测工具 z 离水平面：J5=-76.5→1.13°、-74.9→2.73°、-69.0→8.63°、
# -67.5→10.13°，统一取 11° 上限。
FLAT_TOOL_Z_HORIZONTAL_TOL_DEG = 11.0
# 以下两个参考值仅用于诊断打印，不参与判定
FLAT_J5_REF_DEG = -69.0  # 电动钉值（断使能手拖时通常停在机械止点 ≈-74.9）
FLAT_J6_REF_DEG = 6.75  # 电动钉值
# 姿态超差时"按 y 仍记录"的画面确认超时（秒）；确认走 cv2_display 的
# poll_key 通道（远程画面模式终端 stdin 收不到输入）
FLAT_POSE_CONFIRM_TIMEOUT_S = 5.0
# --mode fit --drop-points 剔除后允许拟合的最少点数（不足则拒绝）
MIN_POINTS_AFTER_DROP = 6
# 方位角跨度过小时 L 不可辨识的提示阈值（度）
AZIMUTH_SPREAD_WARN_DEG = 20.0

TEACHING_POINT_LABELS = tuple(str(i + 1) for i in range(16))


def report_pin_state(arm: PiperFlatHandSDK) -> None:
    """启动后打印手腕是否已固定在钉值附近；未固定时提示手动固定方法。

    PiperFlatHandSDK 初始化时按 config `linker_hand_pin_on_start` 原地固定
    被钉关节（默认 J5/J6）；本函数只做只读检查与提示，不下发任何运动。
    """
    angles_deg, _ = arm.get_arm_angles()
    if angles_deg is None:
        print("Warning: 读取关节角失败，无法确认手腕是否已固定")
        return
    state = "  ".join(
        f"J{joint_no}={float(angles_deg[joint_no - 1]):.2f}°(钉值 {target}°)"
        for joint_no, target in arm.pin_joints
    )
    near = all(
        abs(float(angles_deg[joint_no - 1]) - float(target)) <= 0.5
        for joint_no, target in arm.pin_joints
    )
    if near:
        print(f"启动手腕状态: 已在平抓构型（{state}），全程保持平抓姿态")
    elif arm.pin_wrist_on_start:
        print(
            f"Warning: 启动固定后手腕与钉值仍有偏差（{state}），"
            "请检查是否被障碍物挡住或使能异常"
        )
    else:
        print(
            f"提示: linker_hand_pin_on_start 已关闭，当前 {state}；"
            "可先移动到安全点后调用 arm.pin_wrist_in_place() 手动固定手腕，"
            "否则平抓移动会在移动过程中翻腕"
        )


def check_flat_pose(arm: PiperFlatHandSDK) -> tuple[bool, str]:
    """校验掌面是否处于平抓族（唯一判据：工具 z 离水平面 ≤ 上限）。

    不卡 J5/J6（依据见文件头常量区注释：求解器只解 J1~J4、J6 对掌心位置
    影响严格为 0、J5 两族差 ~2mm 可忽略），J5/J6 只打印实测值作诊断。
    config `linker_hand_flat_pose_check_enabled` 为 false 时完全跳过判定
    （仍打印诊断，不提示不等待）。

    本函数**不做任何等待/交互**：超差时的画面提示与按键确认由调用方（采集
    循环）用 `confirm_record_on_screen` 完成（终端 stdin 在远程画面模式下
    收不到输入，所有确认必须走 cv2_display 的 poll_key 通道）。

    Returns:
        `(ok, info)`：`ok` 为是否可直接记录；`info` 为一行诊断信息
        （超差时含原因，供画面提示使用）。
    """
    enabled = bool(
        get_config_value(
            "linker_hand_flat_pose_check_enabled", True, raise_if_missing=False
        )
    )
    angles_deg, _ = arm.get_arm_angles()
    if angles_deg is None:
        return False, "读取关节角失败"
    fk = arm.chain.forward_kinematics(np.deg2rad(angles_deg))
    tool_z = fk.rot_mat[:, 2]
    horizontality_deg = float(np.rad2deg(np.arcsin(np.clip(abs(tool_z[2]), -1.0, 1.0))))
    ok = horizontality_deg <= FLAT_TOOL_Z_HORIZONTAL_TOL_DEG
    # J5/J6 只作诊断信息，不参与判定
    print(
        f"  腕部信息: J5={float(angles_deg[4]):.2f}° J6={float(angles_deg[5]):.2f}°"
        f"（参考: 电动钉值 J5={FLAT_J5_REF_DEG}°/J6={FLAT_J6_REF_DEG}°，"
        "断使能手拖时 J5 常停在机械止点 ≈-74.9°，均不作卡口）"
    )
    print(
        f"  工具z水平度={horizontality_deg:.2f}°"
        f"(≤{FLAT_TOOL_Z_HORIZONTAL_TOL_DEG}°) -> {'OK' if ok else '超差'}"
        + ("" if enabled else "（姿态判定已关闭，不拦截）")
    )
    if not enabled:
        return True, f"姿态判定已关闭（工具z={horizontality_deg:.1f}°）"
    if ok:
        return True, f"掌面在平抓族（工具z={horizontality_deg:.1f}°）"
    return False, (
        f"掌面不在平抓族: 工具z={horizontality_deg:.1f}° > "
        f"{FLAT_TOOL_Z_HORIZONTAL_TOL_DEG}°（掌心会离开桌面平面）"
    )


def confirm_record_on_screen(
    cam: Camera,
    window_name: str,
    message: str,
    timeout_s: float = FLAT_POSE_CONFIRM_TIMEOUT_S,
) -> str:
    """在实时画面上提示并等待按键确认（走 cv2_display 的 poll_key 通道）。

    远程画面模式下终端 stdin 收不到输入，故确认一律画在实时画面上、
    用 poll_key 取键（与采集循环同一套键通道）。

    Args:
        cam: 相机（用于持续取帧刷新画面）。
        window_name: 显示窗口名（与采集循环一致）。
        message: 超差原因等提示文字。
        timeout_s: 超时秒数，超时按"跳过"处理。

    Returns:
        `"yes"` 按 y（记录该条）；`"skip"` 其他键或超时（跳过该条）；
        `"exit"` 按 ESC（与采集循环的 ESC 退出语义一致）。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        frames = cam.get_frames()
        frame = frames.get("color") if frames else None
        if frame is not None:
            image_to_show = frame.copy()
            cv2.putText(
                image_to_show,
                message,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image_to_show,
                f"按 y 仍记录 / 其他键跳过（{max(0, int(deadline - time.time()))} 秒后自动跳过）",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            show_image(window_name, image_to_show)
        key = poll_key(1)
        if key == -1:
            continue
        if key == 27:
            return "exit"
        return "yes" if key == ord("y") else "skip"
    return "skip"


def pack_flange_xy(end_pos: list[float] | None) -> np.ndarray | None:
    """从 get_arm_pose 的位置取法兰 xy（平抓标定用）。"""
    if end_pos is None:
        return None
    return np.array(end_pos[:2], dtype=np.float32)


def detect_chessboard_points(
    color_image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined_corners.reshape(-1, 2).astype(np.float32)


def index_to_row_col(index: int, pattern_size: tuple[int, int]) -> tuple[int, int]:
    cols, _ = pattern_size
    row = index // cols
    col = index % cols
    return row, col


def select_teaching_corner_indices(pattern_size: tuple[int, int]) -> np.ndarray:
    cols, rows = pattern_size
    if cols < 2 or rows < 2:
        raise ValueError("棋盘格内角点行列数至少都要大于等于2")

    top_row = 0
    bottom_row = rows - 1
    left_col = 0
    right_col = cols - 1
    mid_row = rows // 2
    mid_col = cols // 2

    ordered_points = [
        (top_row, left_col),
        (top_row, mid_col),
        (top_row, right_col),
        (mid_row, right_col),
        (bottom_row, right_col),
        (bottom_row, mid_col),
        (bottom_row, left_col),
        (mid_row, left_col),
        (mid_row, mid_col),
    ]

    unique_points: list[tuple[int, int]] = []
    for point in ordered_points:
        if point not in unique_points:
            unique_points.append(point)

    return np.array([row * cols + col for row, col in unique_points], dtype=np.int32)


def draw_reference_points(
    color_image: np.ndarray,
    corners: np.ndarray,
    reference_indices: np.ndarray,
    active_reference_idx: int | None = None,
) -> np.ndarray:
    image_to_show = color_image.copy()
    for x, y in corners:
        cv2.circle(image_to_show, (int(x), int(y)), 4, (0, 255, 0), -1)
    for label_idx, corner_index in enumerate(reference_indices):
        x, y = corners[int(corner_index)]
        color = (0, 255, 255) if label_idx == active_reference_idx else (0, 0, 255)
        cv2.circle(image_to_show, (int(x), int(y)), 10, color, 2)
        label = TEACHING_POINT_LABELS[label_idx]
        cv2.putText(
            image_to_show,
            label,
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    return image_to_show


def collect_image_pose(
    arm: PiperFlatHandSDK, image_points_path: str, flange_xy_path: str
):
    """手工采集：点选画面目标点 -> 掌心对准 -> 空格记录（平抓模式）。

    每条记录前校验掌面是否处于平抓族（只卡工具 z 水平度；J5/J6 仅打印诊断）；
    工具 z 超差时会在实时画面上提示并等键：按 y 仍记录 / 其他键跳过 / 5 秒
    超时自动跳过（确认走 cv2_display 的 poll_key 通道，终端 stdin 在远程
    画面模式下收不到输入）。
    """
    image_points: list[tuple[int, int]] = []
    flange_points: list[np.ndarray] = []
    selected_point: tuple[int, int] | None = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            print(f"选择图片点: ({x}, {y})，将掌心（抓取接触点）对准该点后按空格记录。")

    window_name = "Camera"
    set_mouse_callback(window_name, mouse_callback)

    cam = Camera(color=True, depth=False, undistort=True)
    print(
        "平抓标定采集：鼠标点选目标点 -> 掌心对准 -> 空格记录 -> ESC 结束。\n"
        "  注意: 机械臂当前已失能并停在安全姿态（J5=+25°），请手动拖动到"
        "下面的平抓姿态后再开始采集。\n"
        "  校验条件: 只卡工具 z 离水平面 ≤11°（掌面贴平，掌心才落在桌面平面）；"
        "J5/J6 不设卡（手拖止点≈-74.9 与电动钉值 -69.0 均可、J6≈6.75），仅打印诊断；"
        "超差时画面会提示按 y 仍记录/其他键跳过\n"
        "  提示: 点位请铺开方位角范围（跨度建议 ≥±20°），否则臂长 L 不可辨识"
    )
    try:
        while True:
            try:
                frames = cam.get_frames()
                color_image = frames.get("color")
                if color_image is None:
                    print("failed to get color image")
                    continue

                image_to_show = color_image.copy()
                if selected_point is not None:
                    cv2.circle(image_to_show, selected_point, 5, (0, 0, 255), -1)
                cv2.putText(
                    image_to_show,
                    f"pairs: {len(image_points)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                show_image(window_name, image_to_show)

                key = poll_key(1)
                if key == 27:
                    break
                if key != ord(" "):
                    continue
                if selected_point is None:
                    print("请先点击图片点，再按空格记录。")
                    continue
                pose_ok, pose_info = check_flat_pose(arm)
                if not pose_ok:
                    # 超差：在实时画面上提示并等键（终端 stdin 在远程画面
                    # 模式下收不到输入）
                    decision = confirm_record_on_screen(cam, window_name, pose_info)
                    if decision != "yes":
                        print(f"本条已跳过（{pose_info}）")
                        if decision == "exit":
                            break
                        continue
                    print(f"已按 y 确认记录（超差: {pose_info}）")

                end_pos, end_rot_deg_zyx = arm.get_arm_pose()
                flange_xy = pack_flange_xy(end_pos)
                if flange_xy is None:
                    print("获取机械臂末端位姿失败")
                    continue

                image_points.append(selected_point)
                flange_points.append(flange_xy)
                print(
                    f"已记录第 {len(image_points)} 条: 像素 {selected_point} -> "
                    f"法兰 xy ({flange_xy[0]:.4f}, {flange_xy[1]:.4f}) m"
                )
                selected_point = None
            except KeyboardInterrupt:
                break
    finally:
        destroy_all_windows()
        cam.close()

    np.save(image_points_path, np.array(image_points, dtype=np.float32))
    np.save(flange_xy_path, np.array(flange_points, dtype=np.float32))
    print(f"采集完成，共 {len(image_points)} 条；已保存像素点与法兰点。")
    return image_points_path, flange_xy_path


def teach_board_reference_points(
    arm: PiperFlatHandSDK,
    cam: Camera,
    pattern_size: tuple[int, int],
    reference_indices: np.ndarray,
    round_index: int,
    capture_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """棋盘格半自动采集：锁定棋盘后按提示依次把掌心移到各角点并记录。"""
    window_name = "Camera"
    locked_corners: np.ndarray | None = None
    flange_points: list[np.ndarray] = []
    current_step = 0
    print(
        f"请先固定棋盘格。第 {round_index + 1}/{capture_count} 轮：按回车锁定当前棋盘位置，"
        "随后按提示依次把掌心（抓取接触点）对准各个角点。"
    )

    while current_step < len(reference_indices):
        frames = cam.get_frames()
        color_image = frames.get("color")
        if color_image is None:
            print("failed to get color image")
            continue

        corners = detect_chessboard_points(color_image, pattern_size)
        image_to_show = color_image.copy()

        if locked_corners is None:
            if corners is not None:
                image_to_show = draw_reference_points(
                    color_image, corners, reference_indices, None
                )
            cv2.putText(
                image_to_show,
                "Press Enter to lock current chessboard",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image_to_show,
                "Need full chessboard visible before locking",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            show_image(window_name, image_to_show)

            key = poll_key(1)
            if key == 27:
                raise KeyboardInterrupt
            if key != 13:
                continue
            if corners is None:
                print("当前未检测到完整棋盘格，无法锁定。")
                continue

            locked_corners = corners.copy()
            print("已锁定棋盘，接下来按提示依次把掌心对准各角点。")
            continue

        image_to_show = draw_reference_points(
            color_image, locked_corners, reference_indices, current_step
        )
        current_corner_index = int(reference_indices[current_step])
        row, col = index_to_row_col(current_corner_index, pattern_size)
        cv2.putText(
            image_to_show,
            f"Move palm to point {current_step + 1} then press Space",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            image_to_show,
            f"Target row={row}, col={col}. Enter unlocks chessboard.",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        show_image(window_name, image_to_show)

        key = poll_key(1)
        if key == 27:
            raise KeyboardInterrupt
        if key == 13:
            locked_corners = None
            current_step = 0
            flange_points.clear()
            print("已解锁棋盘，请重新调整棋盘并按回车锁定。")
            continue
        if key != ord(" "):
            continue
        pose_ok, pose_info = check_flat_pose(arm)
        if not pose_ok:
            # 超差：在实时画面上提示并等键（同手工采集流程，走 poll_key 通道）
            decision = confirm_record_on_screen(cam, window_name, pose_info)
            if decision != "yes":
                print(f"本条已跳过（{pose_info}）")
                if decision == "exit":
                    raise KeyboardInterrupt
                continue
            print(f"已按 y 确认记录（超差: {pose_info}）")

        end_pos, _ = arm.get_arm_pose()
        flange_xy = pack_flange_xy(end_pos)
        if flange_xy is None:
            print("获取机械臂末端位姿失败")
            continue

        flange_points.append(flange_xy)
        print(
            f"已记录点 {current_step + 1} (row={row}, col={col}) -> "
            f"法兰 xy ({flange_xy[0]:.4f}, {flange_xy[1]:.4f}) m"
        )
        current_step += 1

    assert locked_corners is not None
    image_points = locked_corners[reference_indices].astype(np.float32)
    return image_points, np.array(flange_points, dtype=np.float32)


def collect_board_correspondences(
    arm: PiperFlatHandSDK,
    image_points_path: str,
    flange_xy_path: str,
    pattern_size: tuple[int, int],
    capture_count: int,
):
    """棋盘格半自动采集多轮并拼接；只采集不拟合（拟合见 fit_and_save）。"""
    cam = Camera(color=True, depth=False, undistort=True)
    try:
        reference_indices = select_teaching_corner_indices(pattern_size)
        all_image_points: list[np.ndarray] = []
        all_flange_points: list[np.ndarray] = []

        for round_index in range(capture_count):
            image_points, flange_points = teach_board_reference_points(
                arm,
                cam,
                pattern_size,
                reference_indices,
                round_index,
                capture_count,
            )
            all_image_points.append(image_points)
            all_flange_points.append(flange_points)

        merged_image_points = np.concatenate(all_image_points, axis=0).astype(
            np.float32
        )
        merged_flange_points = np.concatenate(all_flange_points, axis=0).astype(
            np.float32
        )
        np.save(image_points_path, merged_image_points)
        np.save(flange_xy_path, merged_flange_points)
        return image_points_path, flange_xy_path
    finally:
        destroy_all_windows()
        cam.close()


def fit_and_save(
    image_points_path: str, flange_xy_path: str, mapping_path: str
) -> str:
    """联合拟合 H（像素->掌心）与 L（臂长），打印对照并保存 npz。"""
    image_points = np.load(image_points_path).astype(np.float64)
    flange_xy = np.load(flange_xy_path).astype(np.float64)
    if image_points.shape[0] != flange_xy.shape[0]:
        raise ValueError("像素点与法兰点数量不匹配")
    heading_offset_deg = float(
        get_config_value(
            "linker_hand_flat_heading_offset_deg", 4.2, raise_if_missing=False
        )
    )
    H, L_m, stats = fit_flat_mapping(image_points, flange_xy, heading_offset_deg)
    save_mapping(
        mapping_path, H, L_m, heading_offset_deg, stats, image_points, flange_xy
    )
    print(f"使用朝向偏移 φ0 = {heading_offset_deg:.1f}°（linker_hand_flat_heading_offset_deg）")
    print("单应矩阵 H（像素 -> 掌心，单位米）:")
    print(np.array2string(H, precision=8))
    print(
        f"臂长 L = {L_m * 1000:.1f} mm  联合模型 RMS = {stats['rms_m'] * 1000:.2f} mm"
        f"（内点 {stats['n_inliers']}/{stats['n']}，{stats.get('fit_method', '?')}）"
    )
    print(
        f"纯单应 L=0 对照 RMS = {stats['rms_m_l0'] * 1000:.2f} mm"
        "（明显更大说明平抓必须用 H+L 联合模型）"
    )
    print(f"逐点残差(mm): {np.round(stats['residuals_m'] * 1000, 2).tolist()}")
    if stats.get("worst_points"):
        worst = "，".join(str(i) for i, _ in stats["worst_points"])
        print(
            f"提示: 若判定上述最差点为坏点，可剔除后重拟合（1 起算、按采集顺序）："
            f"python3 arm/calibrate_arm_hand.py --mode fit --drop-points {worst}"
        )
    print(f"方位角跨度 = {stats['azimuth_spread_deg']:.1f}°")
    if stats["azimuth_spread_deg"] < AZIMUTH_SPREAD_WARN_DEG:
        print(
            f"Warning: 方位角跨度 < {AZIMUTH_SPREAD_WARN_DEG}°，臂长 L 可能不可辨识"
            "（与单应平移项接近共线），建议补采铺开方向的点后重新拟合。"
        )
    print(f"标定结果已保存: {mapping_path}")
    return mapping_path


def calibrate_flat(
    image_points_path: str,
    flange_xy_path: str,
    mapping_path: str,
    drop_indices: tuple[int, ...] = (),
):
    """由已采集的点重新拟合并保存（不再采集）。

    Args:
        drop_indices: 要剔除的点（1 起算，对应采集顺序）；剔除后的点集会写回
            原始 npy 再拟合。剔除后不足 MIN_POINTS_AFTER_DROP 个点则拒绝。
    """
    image_points = np.load(image_points_path)
    flange_xy = np.load(flange_xy_path)
    if drop_indices:
        drop_set = set(int(i) for i in drop_indices)
        keep = [i for i in range(1, len(image_points) + 1) if i not in drop_set]
        if len(keep) < MIN_POINTS_AFTER_DROP:
            raise ValueError(
                f"剔除 {sorted(drop_set)} 后仅剩 {len(keep)} 个点，"
                f"少于 {MIN_POINTS_AFTER_DROP} 个，拒绝拟合"
            )
        np.save(image_points_path, image_points[[k - 1 for k in keep]])
        np.save(flange_xy_path, flange_xy[[k - 1 for k in keep]])
        print(
            f"已剔除点（1 起算）{sorted(drop_set)}，剩余 {len(keep)} 点，"
            f"已写回 {image_points_path} / {flange_xy_path}"
        )
    return fit_and_save(image_points_path, flange_xy_path, mapping_path)


def test_flat_mapping(mapping_path: str):
    """点击画面验证平抓映射：预测掌心/法兰并让机械臂走平抓姿态前往（会动真机）。"""
    mapping = load_mapping(mapping_path)
    print(
        f"已加载平抓标定: L={mapping.L_m * 1000:.1f}mm  RMS={mapping.rms_m * 1000:.2f}mm"
        f"  φ0={mapping.heading_offset_deg:.1f}°"
    )
    heading_offset_deg = float(mapping.heading_offset_deg)
    arm = PiperFlatHandSDK()
    report_pin_state(arm)
    move_lock = threading.Lock()
    desktop_height = float(get_config_value("default_desktop_height"))
    # 平抓工作高度 = 绝对值（唯一来源，config linker_hand_flat_grasp_height_m，
    # 默认 0.11 = 原 0.15+0.01-0.05；2026-09-11 实机在 test 模式验证合适，
    # 抓取脚本同步用同一值）
    target_z = arm.flat_grasp_height_m
    print(
        f"test 目标高度 z={target_z:.3f} m"
        f"（linker_hand_flat_grasp_height_m，绝对值；桌面默认 {desktop_height:.3f}）"
    )
    if target_z < desktop_height:
        print(
            f"Warning: 目标高度低于桌面默认高度（{target_z:.3f} < "
            f"{desktop_height:.3f}）：掌心会低于配置的桌面平面，实机可能撞台，"
            "调整 linker_hand_flat_grasp_height_m 时请注意"
        )
    last_click: dict[str, tuple[float, float] | tuple[float, float]] = {}

    def move_to_predicted(palm_xy, flange_xy, heading_rad):
        with move_lock:
            res = arm.move_to_flat(
                [float(flange_xy[0]), float(flange_xy[1]), target_z],
                rot_rad=float(heading_rad),
            )
            print(
                f"move_to_flat -> 法兰 ({flange_xy[0]:.4f}, {flange_xy[1]:.4f}, {target_z:.3f})"
                f" 朝向 {np.rad2deg(heading_rad):.1f}° 结果={res}"
            )

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        palm_xy, flange_xy = mapping.predict(x, y)
        heading_rad = np.arctan2(palm_xy[1], palm_xy[0]) + np.deg2rad(
            heading_offset_deg
        )
        last_click["palm"] = (float(palm_xy[0]), float(palm_xy[1]))
        last_click["flange"] = (float(flange_xy[0]), float(flange_xy[1]))
        print(
            f"点击 ({x}, {y}) -> 掌心 ({palm_xy[0]:.4f}, {palm_xy[1]:.4f}) "
            f"法兰 ({flange_xy[0]:.4f}, {flange_xy[1]:.4f}) 朝向 {np.rad2deg(heading_rad):.1f}°"
        )
        threading.Thread(
            target=move_to_predicted,
            args=(palm_xy, flange_xy, heading_rad),
            daemon=True,
        ).start()

    window_name = "Camera"
    set_mouse_callback(window_name, mouse_callback)

    cam = Camera(color=True, depth=False, undistort=True)
    try:
        while True:
            try:
                frames = cam.get_frames()
                color_image = frames.get("color")
                if color_image is None:
                    print("failed to get color image")
                    time.sleep(0.5)
                    continue

                image_to_show = color_image.copy()
                if "palm" in last_click:
                    px, py = last_click["palm"]
                    fx, fy = last_click["flange"]
                    cv2.putText(
                        image_to_show,
                        f"palm ({px:.4f}, {py:.4f}) flange ({fx:.4f}, {fy:.4f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                show_image(window_name, image_to_show)

                key = poll_key(1)
                if key == 27:
                    break
            except KeyboardInterrupt:
                break
    finally:
        destroy_all_windows()
        cam.close()
        arm.disconnect_arm()


def main():
    argparser = argparse.ArgumentParser(description="平抓姿态 2D 手眼标定")
    argparser.add_argument(
        "--mode",
        type=str,
        default="test",
        help="模式: calibrate 手动采集多点; calibrate_board 半自动采集多点; "
        "fit 只重新拟合并保存; test 测试映射（会动机械臂）",
    )
    argparser.add_argument(
        "--pattern-cols", type=int, default=12, help="棋盘格每行内角点数（格子数-1）"
    )
    argparser.add_argument(
        "--pattern-rows", type=int, default=12, help="棋盘格每列内角点数（格子数-1）"
    )
    argparser.add_argument(
        "--capture-count",
        type=int,
        default=1,
        help="calibrate_board 模式下重复示教的轮数，多轮数据拼接后共同拟合",
    )
    argparser.add_argument(
        "--drop-points",
        type=str,
        default="",
        help="fit 模式：剔除这些点后重拟合（1 起算、逗号分隔，如 2,5，"
        "对应用户采集顺序；剔除后的点集会写回原始 npy）",
    )
    args = argparser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "hand-eye-data")
    # 平抓专用文件，不复用/不覆盖垂直模式的 2d_*.npy / 2d_homography.npy
    image_points_path = os.path.join(data_dir, "2d_flat_image_points.npy")
    flange_xy_path = os.path.join(data_dir, "2d_flat_flange_xy.npy")
    mapping_path = os.path.join(data_dir, "2d_homography_flat.npz")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.mode == "calibrate":
        arm = PiperFlatHandSDK()
        report_pin_state(arm)
        # 安全失能（与 arm/disable_arm.py 的流程一致）：先摆到 J5=+25° 的
        # 安全姿态再失能，避免"启动固定手腕"后的折叠+平掌构型突然失能因
        # 重力下坠；失能后再由用户手拖到平抓姿态开始采集
        arm.safe_disable()
        collect_image_pose(arm, image_points_path, flange_xy_path)
        fit_and_save(image_points_path, flange_xy_path, mapping_path)
        # 收尾：此时手臂已被手拖过且处于失能状态，先在安全条件下重新使能
        # 摆回安全姿态，再失能并断开 CAN（不再调父类 disconnect_arm，避免
        # 它对已失能机械臂发送回零指令后空等到位超时）
        arm.safe_disable(re_enable=True, disconnect_port=True)
    elif args.mode == "calibrate_board":
        arm = PiperFlatHandSDK()
        report_pin_state(arm)
        # 同 calibrate：先摆到安全姿态再失能供手拖（与 arm/disable_arm.py 一致）
        arm.safe_disable()
        collect_board_correspondences(
            arm,
            image_points_path,
            flange_xy_path,
            (args.pattern_cols, args.pattern_rows),
            args.capture_count,
        )
        fit_and_save(image_points_path, flange_xy_path, mapping_path)
        # 收尾：重新使能摆回安全姿态再失能并断开（理由同 calibrate 分支）
        arm.safe_disable(re_enable=True, disconnect_port=True)
    elif args.mode == "fit":
        drop_indices = tuple(
            int(x)
            for x in args.drop_points.replace("，", ",").split(",")
            if x.strip()
        )
        calibrate_flat(
            image_points_path, flange_xy_path, mapping_path, drop_indices=drop_indices
        )
    elif args.mode == "test":
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(
                f"未找到平抓标定文件: {mapping_path}，请先运行 --mode calibrate"
            )
        test_flat_mapping(mapping_path)


if __name__ == "__main__":
    main()
