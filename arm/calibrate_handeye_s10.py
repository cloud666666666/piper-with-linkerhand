# Description: S10 眼在手上 3D 手眼标定（臂上相机 ↔ 臂末端）
# 标定前先把机械臂伸出去、末端垂直向下停在观察点上方——该姿态与
# grab skill 抓取前的悬停姿态一致（末端欧拉角 [0, 180, 0] 朝下），
# 臂上相机随之垂直向下俯视，标定板平放在观察点正下方的地面/台面上。
# 流程：回零 → 伸臂到观察点 → 预览确认标定板完整可见 → 在观察姿态
# 附近小幅扰动自动采集（图像 + 末端位姿）→ cv2.calibrateHandEye 求解
# 相机↔末端变换 → 保存 hand-eye-data/3d_handeye.npz。
# 依赖：calibration/intrinsics-data 下已有相机内参与畸变系数。
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from arm.arm_base import Arm
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, poll_key, destroy_all_windows

# 末端垂直朝下的姿态（度，zyx 顺序），与 grab skill 抓取前悬停姿态一致
DEFAULT_DOWN_EULER_DEG_ZYX = [0.0, 180.0, 0.0]


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
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float32)


def make_object_points(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    cols, rows = pattern_size
    points = []
    for row in range(rows):
        for col in range(cols):
            points.append([col * square_size, row * square_size, 0.0])
    return np.array(points, dtype=np.float32)


def end_pose_to_matrix(pos_m: list[float], euler_deg_zyx: list[float]) -> np.ndarray:
    """末端位姿（位置单位米，欧拉角 zyx 单位度）→ 4x4 变换矩阵。"""
    from scipy.spatial.transform import Rotation as R

    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("zyx", euler_deg_zyx, degrees=True).as_matrix()
    matrix[:3, 3] = np.asarray(pos_m, dtype=np.float64)
    return matrix


class UsbCamera:
    """Linux UVC 相机，RealSense D435i 的彩色流可直接按 UVC 读取。

    注意：D435i 的 UVC 节点是多路复用的。index 2 打开后只有前 3 帧是彩色，
    之后一直是 IR（黑白 + 激光点阵）；index 4 打开后前 3 帧彩色、约 9 帧暗帧，
    之后稳定彩色。本类打开后先热机丢弃 20 帧，再验证画面确为彩色流
    （通道差 >20 且亮度 >25），失败则重开，最多 3 次。
    """

    def __init__(
        self,
        index: int,
        width: int = 1280,
        height: int = 720,
        max_attempts: int = 3,
        wb_blue_u: float | None = None,
        wb_red_v: float | None = None,
    ):
        """wb_blue_u / wb_red_v：手动白平衡增益（D435i 的 UVC 模式 AWB 失效，
        画面发绿）。不给则保持自动白平衡。"""
        for attempt in range(max_attempts):
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开相机 index={index}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if wb_blue_u is not None or wb_red_v is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                if wb_blue_u is not None:
                    self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, wb_blue_u)
                if wb_red_v is not None:
                    self.cap.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, wb_red_v)
            for _ in range(20):
                self.cap.read()
            ok, frame = self.cap.read()
            if ok and frame is not None and is_color_frame(frame):
                return
            print(
                f"[UsbCamera] index={index} 第 {attempt + 1} 次打开未获得彩色流，重开..."
            )
            self.cap.release()
        raise RuntimeError(
            f"index={index} 反复打开仍是 IR/暗帧，无法获得稳定彩色流"
            f"（D435i UVC 多路复用）。建议换 index=4 或检查光照。"
        )

    def get_frames(self) -> dict:
        ok, frame = self.cap.read()
        return {"color": frame if ok else None}

    def close(self) -> None:
        self.cap.release()


def is_color_frame(frame: np.ndarray) -> bool:
    """IR 帧通道差≈0 且带激光点阵；暗帧亮度<25；彩色帧两者都不满足。"""
    b, g, r = cv2.split(frame)
    diff = float(
        np.mean(np.abs(r.astype(int) - g.astype(int)))
        + np.mean(np.abs(g.astype(int) - b.astype(int)))
    )
    return diff > 20.0 and float(frame.mean()) > 25.0


def build_pose_variations(
    observe_pos: list[float], xy_range: float, z_step: float, tilt_deg: float
) -> list[tuple[list[float], list[float]]]:
    """在观察姿态附近生成扰动位姿集合，保证旋转多样性。"""
    x, y, z = observe_pos
    poses: list[tuple[list[float], list[float]]] = []
    # 倾角扰动：RY/RX 相对 180 小幅偏移，让相机轴偏离竖直
    for euler in (
        [0.0, 180.0, 0.0],
        [0.0, 180.0 + tilt_deg, 0.0],
        [0.0, 180.0 - tilt_deg, 0.0],
        [0.0, 180.0, tilt_deg],
        [0.0, 180.0, -tilt_deg],
    ):
        poses.append(([x, y, z], euler))
    # 水平平移扰动
    for dx, dy in (
        (xy_range, 0.0),
        (-xy_range, 0.0),
        (0.0, xy_range),
        (0.0, -xy_range),
    ):
        poses.append(([x + dx, y + dy, z], [0.0, 180.0, 0.0]))
    # 高度扰动
    poses.append(([x, y, z + z_step], [0.0, 180.0, 0.0]))
    # 混合扰动补足多样性
    poses.append(
        ([x + xy_range, y, z + z_step], [0.0, 180.0 + tilt_deg, 0.0])
    )
    poses.append(
        ([x, y - xy_range, z + z_step], [0.0, 180.0 - tilt_deg, 0.0])
    )
    return poses


def preview_board(cam: UsbCamera, pattern_size: tuple[int, int]) -> bool:
    """实时预览并标注角点，空格确认开始（需检测到棋盘），ESC 中止。"""
    while True:
        frames = cam.get_frames()
        color_image = frames.get("color")
        if color_image is None:
            print("failed to get color image")
            continue
        shown = color_image.copy()
        corners = detect_chessboard_points(color_image, pattern_size)
        if corners is not None:
            cv2.drawChessboardCorners(
                shown, pattern_size, corners.reshape(-1, 1, 2), True
            )
        cv2.putText(
            shown,
            "Space: start   ESC: abort",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        show_image("S10 Hand-Eye Calibration", shown)
        key = poll_key(1)
        if key == 27:
            return False
        if key == ord(" "):
            if corners is None:
                print("当前画面未检测到完整棋盘格，请调整标定板位置。")
                continue
            return True


def collect_pairs(
    arm: Arm,
    cam: UsbCamera,
    poses: list[tuple[list[float], list[float]]],
    pattern_size: tuple[int, int],
    data_dir: str,
    settle_s: float,
) -> list[tuple[str, list[float], list[float]]]:
    """逐个位姿移动、拍照、检测角点、记录末端位姿。"""
    saved: list[tuple[str, list[float], list[float]]] = []
    for i, (pos, euler) in enumerate(poses):
        print(f"[{i + 1}/{len(poses)}] 移动到位 {pos}, 欧拉角 {euler}")
        if not arm.move_to(pos, euler_angles_deg_zyx=euler):
            print("  移动失败（IK 无解或超时），跳过")
            continue
        time.sleep(settle_s)
        frames = cam.get_frames()
        color_image = frames.get("color")
        if color_image is None:
            print("  读取画面失败，跳过")
            continue
        corners = detect_chessboard_points(color_image, pattern_size)
        if corners is None:
            print("  未检测到完整棋盘格，跳过")
            continue
        end_pos, end_euler = arm.get_arm_pose()
        if end_pos is None or end_euler is None:
            print("  读取末端位姿失败，跳过")
            continue
        image_path = os.path.join(data_dir, f"handeye_{len(saved):03d}.png")
        cv2.imwrite(image_path, color_image)
        saved.append((image_path, end_pos, end_euler))
        print(f"  已采集 {len(saved)} 组（角点 {len(corners)} 个）")
    return saved


def solve_handeye(
    saved: list[tuple[str, list[float], list[float]]],
    pattern_size: tuple[int, int],
    square_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """求解相机↔末端变换，并返回每视图下标定板在基座坐标系中的位姿。"""
    objp = make_object_points(pattern_size, square_size)
    R_gripper2base: list[np.ndarray] = []
    T_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    T_target2cam: list[np.ndarray] = []
    rt_g2b_list: list[np.ndarray] = []
    rt_t2c_list: list[np.ndarray] = []
    for image_path, pos, euler in saved:
        image = cv2.imread(image_path)
        if image is None:
            continue
        corners = detect_chessboard_points(image, pattern_size)
        if corners is None:
            continue
        found, rvec, tvec = cv2.solvePnP(
            objp, corners, camera_matrix, dist_coeffs
        )
        if not found:
            continue
        rt_g2b = end_pose_to_matrix(pos, euler)
        rt_t2c = np.eye(4)
        rt_t2c[:3, :3], _ = cv2.Rodrigues(rvec)
        rt_t2c[:3, 3] = tvec.ravel()
        R_gripper2base.append(rt_g2b[:3, :3])
        T_gripper2base.append(rt_g2b[:3, 3].reshape(3, 1))
        R_target2cam.append(rt_t2c[:3, :3])
        T_target2cam.append(rt_t2c[:3, 3].reshape(3, 1))
        rt_g2b_list.append(rt_g2b)
        rt_t2c_list.append(rt_t2c)
    R_cam2gripper, T_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        T_gripper2base,
        R_target2cam,
        T_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    rt_c2g = np.eye(4)
    rt_c2g[:3, :3] = R_cam2gripper
    rt_c2g[:3, 3] = T_cam2gripper.ravel()
    # 每个视图推算标定板在基座坐标系中的位姿，理论上应完全一致
    target_in_base = [
        rt_g2b @ rt_c2g @ rt_t2c
        for rt_g2b, rt_t2c in zip(rt_g2b_list, rt_t2c_list, strict=True)
    ]
    return rt_c2g, target_in_base


def main():
    argparser = argparse.ArgumentParser(description="S10 眼在手上 3D 手眼标定")
    argparser.add_argument(
        "--pattern-cols", type=int, default=8, help="棋盘格每行内角点数（格子数-1）"
    )
    argparser.add_argument(
        "--pattern-rows", type=int, default=8, help="棋盘格每列内角点数（格子数-1）"
    )
    argparser.add_argument(
        "--square-size", type=float, default=0.04, help="棋盘格单格边长，单位米"
    )
    argparser.add_argument(
        "--observe-x", type=float, default=0.25, help="观察点 x 坐标，单位米"
    )
    argparser.add_argument(
        "--observe-y", type=float, default=0.0, help="观察点 y 坐标，单位米"
    )
    argparser.add_argument(
        "--observe-height",
        type=float,
        default=None,
        help="观察点高度（末端到标定板平面的距离），默认桌面高度+抓取抬起高度",
    )
    argparser.add_argument(
        "--xy-range", type=float, default=0.05, help="水平扰动幅度，单位米"
    )
    argparser.add_argument(
        "--z-step", type=float, default=0.05, help="高度扰动步长，单位米"
    )
    argparser.add_argument(
        "--tilt-deg", type=float, default=8.0, help="倾角扰动幅度，单位度"
    )
    argparser.add_argument(
        "--settle-s", type=float, default=1.0, help="移动到位后稳定等待时间"
    )
    argparser.add_argument(
        "--camera-index", type=int, default=0, help="UVC 相机编号（/dev/video*）"
    )
    argparser.add_argument(
        "--min-pairs", type=int, default=5, help="求解所需的最少有效位姿组数"
    )
    argparser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="采集数据保存目录，默认 arm/hand-eye-data/s10",
    )
    args = argparser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(__file__))
    intrinsics_dir = os.path.join(repo_root, "calibration", "intrinsics-data")
    camera_matrix_path = os.path.join(intrinsics_dir, "camera_matrix.npy")
    dist_coeffs_path = os.path.join(intrinsics_dir, "dist_coeffs.npy")
    if not os.path.exists(camera_matrix_path) or not os.path.exists(dist_coeffs_path):
        raise RuntimeError(
            "未找到相机内参，请先运行 calibration/calibrate_intrinsics.py 标定内参"
        )
    camera_matrix = np.load(camera_matrix_path)
    dist_coeffs = np.load(dist_coeffs_path)

    if args.observe_height is None:
        args.observe_height = (
            get_config_value("default_desktop_height")
            + get_config_value("catch_raise_height", 0.1)
        )

    data_dir = args.data_dir or os.path.join(
        repo_root, "arm", "hand-eye-data", "s10"
    )
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("正在初始化机械臂（驱动会先使能并自动回零）...")
    arm = Arm()
    cam = UsbCamera(args.camera_index)
    pattern_size = (args.pattern_cols, args.pattern_rows)

    try:
        # 第一步：伸臂到观察点，末端垂直向下（grab skill 抓取前的悬停姿态）
        observe_pos = [args.observe_x, args.observe_y, args.observe_height]
        print(
            f"机械臂伸向观察点 {observe_pos}，末端垂直向下，臂上相机俯视地面..."
        )
        if not arm.move_to(
            observe_pos, euler_angles_deg_zyx=list(DEFAULT_DOWN_EULER_DEG_ZYX)
        ):
            raise RuntimeError("移动机械臂到观察点失败")

        # 第二步：预览确认标定板可见
        if not preview_board(cam, pattern_size):
            print("已取消标定")
            return

        # 第三步：观察姿态附近扰动采集
        poses = build_pose_variations(
            observe_pos, args.xy_range, args.z_step, args.tilt_deg
        )
        saved = collect_pairs(
            arm, cam, poses, pattern_size, data_dir, args.settle_s
        )
        if len(saved) < args.min_pairs:
            print(f"有效位姿组不足（{len(saved)}/{args.min_pairs}），已退出")
            return

        # 第四步：求解并保存
        rt_cam2gripper, target_in_base = solve_handeye(
            saved, pattern_size, args.square_size, camera_matrix, dist_coeffs
        )
        result_path = os.path.join(
            repo_root, "arm", "hand-eye-data", "3d_handeye.npz"
        )
        np.savez(
            result_path,
            R_cam2gripper=rt_cam2gripper[:3, :3],
            T_cam2gripper=rt_cam2gripper[:3, 3],
            target_in_base=np.array(target_in_base),
        )

        print(f"使用位姿组数: {len(saved)}")
        print("相机在末端坐标系下的位姿 R_cam2gripper:")
        print(rt_cam2gripper)
        positions = np.array([m[:3, 3] for m in target_in_base])
        mean_pos = positions.mean(axis=0)
        std_pos = positions.std(axis=0)
        print(f"标定板在基座坐标系中的位置（各视图均值）: {mean_pos}")
        print(f"各视图间位置标准差: {std_pos}（越小越好）")
        print(f"已保存: {result_path}")
    finally:
        destroy_all_windows()
        cam.close()
        print("机械臂回零并断开...")
        arm.disconnect_arm()


if __name__ == "__main__":
    main()
