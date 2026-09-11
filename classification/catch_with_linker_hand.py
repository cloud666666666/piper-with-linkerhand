# Description: 调用目标识别和机械臂控制，使用灵心巧手（LinkerHand）灵巧手实现抓取。
# 抓取流程与 catch_with_arm.py（夹爪版）一致：
# 检测 -> 手眼标定转坐标 -> 手掌张开移到目标上方 -> 下降 -> 按物体宽度包络抓取
# -> 抬起 -> 触觉判定是否抓到 -> 移到放置位置 -> 松开 -> 回位。
import argparse
import concurrent.futures
import copy
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np

from arm.arm_base import Arm
from arm.linker_hand import LinkerHand
from camera.camera_api import Camera
from object_detect.detect import (
    detect_objects_in_frame,
    load_model,
    draw_box,
)
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, poll_key, destroy_all_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用目标识别和机械臂控制，使用灵巧手实现抓取功能。"
    )
    parser.add_argument(
        "--target",
        default='potato',
        help=(
            "指定要抓取的物体类别名（class_name）。"
            "指定后只抓该类别中置信度最高的物体；"
            "不指定则抓全部检测中置信度最高的物体。"
        ),
    )
    return parser.parse_args()


def main(target_class: str = None):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    model_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
        for path in get_config_value("classification_YOLO_model_path", [])
    ]
    default_gripper_aside_pos = get_config_value(
        "default_gripper_aside_pos", raise_if_missing=False
    )
    default_conf_thres = get_config_value("default_conf_thres")
    place_pos = get_config_value("place_pos", default={}, raise_if_missing=False)
    place_distance_threshold = get_config_value(
        "place_distance_threshold", default=0, raise_if_missing=False
    )
    offset = get_config_value("catch_offset")
    # 抓取时手掌中心高于桌面的偏移量（灵巧手从上方包络物体，手掌需高于物体）
    grasp_height_offset = get_config_value(
        "linker_hand_grasp_height_offset", 0.04, raise_if_missing=False
    )

    arm = Arm()
    hand = LinkerHand()
    arm.move_to_home()
    hand.open_palm()
    cam = Camera(color=True, depth=False, undistort=True)
    models = [load_model(model_path) for model_path in model_paths]
    detections = []
    future = None

    def catch_and_place(
        target_x: float,
        target_y: float,
        catch_rotate_rad: float,
        place_pos_: list[float | int],
        object_width_m: float,
    ) -> bool:
        """使用灵巧手执行一次完整抓取-放置，流程与夹爪版 arm.catch_and_place 一致。

        Args:
            target_x: 抓取点 x 坐标，单位为米。
            target_y: 抓取点 y 坐标，单位为米。
            catch_rotate_rad: 抓取时末端绕 z 轴的旋转角，单位为弧度。
            place_pos_: 放置位置，格式为 `[x, y]` 或 `[x, y, z]`，单位为米。
            object_width_m: 物体近似宽度，单位为米，用于包络抓取开度。

        Returns:
            整个流程是否成功。
        """
        if len(place_pos_) == 2:
            place_x, place_y = place_pos_
            place_z = arm.desktop_height + grasp_height_offset
        elif len(place_pos_) == 3:
            place_x, place_y, place_z = place_pos_
        else:
            print("放置位置格式错误，应该是[x, y]或[x, y, z]")
            return False

        target_z = arm.desktop_height + grasp_height_offset

        # 手掌张开，移到目标上方
        hand.open_palm()
        res = arm.move_to(
            [target_x, target_y, target_z + arm.catch_raise_height],
            rot_rad=catch_rotate_rad,
        )
        if not res:
            print("移动到目标位置上方失败，取消抓取")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s * 2)

        # 下降到抓取高度
        res = arm.move_to(
            [target_x, target_y, target_z],
            rot_rad=catch_rotate_rad,
        )
        if not res:
            print("移动到目标位置失败，取消抓取")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        # 按物体宽度包络抓取
        hand.grasp_by_size(object_width_m)
        time.sleep(arm.catch_time_interval_s)

        # 抬起
        res = arm.move_to(
            [target_x, target_y, target_z + arm.catch_raise_height],
            rot_rad=catch_rotate_rad,
        )
        if not res:
            print("抬起失败，取消抓取")
            hand.open_palm()
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        # 触觉判定是否抓到
        if not hand.check_grasped():
            print("灵巧手抓取失败")
            hand.open_palm()
            arm.move_to_home()
            return False

        # 移动到放置位置上方
        res = arm.move_to(
            [place_x, place_y, place_z + arm.place_raise_height],
        )
        if not res:
            print("移动到放置位置上方失败，取消放置")
            hand.open_palm()
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s * 2)

        # 灵巧手需要先下降到放置高度再松开，物体才能平稳落下
        res = arm.move_to(
            [place_x, place_y, place_z],
        )
        if not res:
            print("移动到放置位置失败，取消放置")
            hand.open_palm()
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        # 松开
        hand.open_palm()
        time.sleep(arm.catch_time_interval_s)

        res = arm.move_to(
            [place_x, place_y, place_z + arm.place_raise_height],
        )
        if not res:
            print("放置后抬起失败")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        arm.move_to_home()
        return True

    while True:
        start_time = time.time()
        try:
            frames = cam.get_frames()
            if frames is None:
                continue
            frame = frames.get("color")
            if frame is None:
                continue

            frame_w, frame_h = frame.shape[:2]
            if future is None or future.done():
                detections = []
                for model in models:
                    detections.extend(
                        detect_objects_in_frame(
                            model, frame, conf_thres=default_conf_thres
                        )
                    )

            # 先画出所有检测框，便于观察
            for (u, v, w, h, r), score, class_id, class_name in detections:
                draw_box(
                    frame, u, v, w, h, np.rad2deg(r), f"{class_name}: {score:.2f}"
                )

            # 选出本轮要抓取的目标：
            # - 指定了 target_class 时，只在该类别里挑；
            # - 未指定时，从所有检测里挑；
            # 两种情况都取置信度（score）最高的那个。
            if future is None or future.done():
                candidates = detections
                if target_class is not None:
                    candidates = [
                        d for d in detections if d[3] == target_class
                    ]
                if candidates:
                    (u, v, w, h, r), score, class_id, class_name = max(
                        candidates, key=lambda d: d[1]
                    )
                    angle_deg = np.rad2deg(r)
                    # 将图像坐标转换为机械臂坐标系
                    target_x, target_y = arm.pixel2pos(u, v)
                    gripper_angle_rad = arm.gripper_angle_by_longer(
                        u, v, w, h, angle_deg
                    )
                    # 物体近似宽度（米）：检测框短边两端经手眼标定换算
                    short_edge_px = min(w, h)
                    side_x, side_y = arm.pixel2pos(
                        min(u + short_edge_px, frame_w - 1), v
                    )
                    object_width_m = float(
                        np.linalg.norm(
                            np.array([side_x, side_y]) - np.array([target_x, target_y])
                        )
                    )
                    class_place_pos = copy.deepcopy(place_pos.get(class_name, None))
                    if class_place_pos is None or "pos" not in class_place_pos:
                        print("No placement location specified, place in origin.")
                        class_place_pos = {"pos": [target_x, target_y]}
                    else:
                        match class_place_pos["pos"][0]:
                            case "x":
                                class_place_pos["pos"][0] = target_x
                            case "-x":
                                class_place_pos["pos"][0] = -target_x
                            case "y":
                                class_place_pos["pos"][0] = target_y
                            case "-y":
                                class_place_pos["pos"][0] = -target_y
                        match class_place_pos["pos"][1]:
                            case "x":
                                class_place_pos["pos"][1] = target_x
                            case "-x":
                                class_place_pos["pos"][1] = -target_x
                            case "y":
                                class_place_pos["pos"][1] = target_y
                            case "-y":
                                class_place_pos["pos"][1] = -target_y
                    if (
                        np.linalg.norm(
                            np.array(class_place_pos.get("pos"))
                            - np.array([target_x, target_y])
                        )
                        < place_distance_threshold
                    ):
                        print(
                            f"Object {class_name} is too close to place position, skipping catch."
                        )
                    else:
                        future = executor.submit(
                            catch_and_place,
                            # 手掌向外偏移一些，避免刚好顶到物体
                            target_x + offset * np.cos(gripper_angle_rad),
                            target_y + offset * np.sin(-gripper_angle_rad),
                            gripper_angle_rad,
                            class_place_pos["pos"],
                            object_width_m,
                        )

            if default_gripper_aside_pos and (future is None or future.done()):
                # 移到旁边以免挡住视野
                future = executor.submit(
                    arm.move_to,
                    default_gripper_aside_pos,
                )
            end_time = time.time()
            if end_time - start_time == 0:
                fps = 0.0
            else:
                fps = 1 / (end_time - start_time)
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            show_image("Detections", frame)
            if poll_key(1) & 0xFF == 27:  # 按Esc键退出
                break
        except KeyboardInterrupt:
            print("Exiting...")
            break

    hand.open_palm()
    arm.move_to_home()
    time.sleep(1)
    hand.disconnect()
    arm.disconnect_arm()
    cam.close()
    destroy_all_windows()


if __name__ == "__main__":
    args = parse_args()
    main(target_class=args.target)
