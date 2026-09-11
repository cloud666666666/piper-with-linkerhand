# Description: 调用目标识别和机械臂控制，使用灵心巧手（LinkerHand）灵巧手
# 实现"水平包络抓取"：末端换成灵巧手后不能再用垂直朝下姿态，改为第 5/6
# 腕关节按示教钉死、掌面放平（约平行桌面），手水平地从侧面滑过去包络抓取。
# 与 catch_with_linker_hand.py（垂直下抓版）的区别：
# - 不走 Arm() 工厂，直接实例化 PiperFlatHandSDK（平抓姿态适配类，
#   避免修改 arm_base.py / piper_ctrl_by_sdk.py）；
# - 抓取序列（全程平抓姿态：PiperFlatHandSDK 启动即原地固定 J5/J6，
#   不再"先到位置再翻腕"）：原地确保手腕平抓（pin_wrist_in_place）
#   -> 手掌张开 -> 平抓姿态到目标正上方（direct_down 默认；slide_in 则先到
#   接近点上方）-> 下降到抓取高度 -> [slide_in 才水平滑入目标 xy]
#   -> 握拳闭合+触觉判定 -> 原姿态垂直抬起 -> 平移到放置点上方 -> 下降
#   -> 张手放置 -> 抬起 -> 回 home（下一周期开始时会原地重新固定手腕）。
#   接近方式由 config linker_hand_flat_approach_mode 决定（默认 direct_down：
#   正上方对准后垂直下抓，避免水平滑入时大拇指侧向扫过把物体推走）。
# 抓取模式由 config linker_hand_flat_catch_once 决定：默认 true＝完成一次抓取
# 尝试（无论成功失败）后打印结果并退出（调试/验证用，未命中目标时仍继续等待
# 检测）；false＝连续抓取。
# 双钉（J5/J6 钉死）走廊模式下，平抓朝向与抓取点方位角绑定：
# 接近/滑入方向由物体方位角决定（+示教走廊偏移 linker_hand_flat_heading_offset_deg），
# 不跟物体长边；gripper_angle_by_longer 的结果仅作参考，未来放开 J6 后可恢复参与运动。
# 坐标映射改用平抓联合标定（arm/flat_handeye.py 的 H+L 模型，文件由
# arm/calibrate_arm_hand.py 生成，config 键 linker_hand_flat_handeye_file）：
# 像素 -> 掌心（抓取接触点）/法兰，法兰目标 = 掌心 + L·u(方位角+φ0)。
import argparse
import concurrent.futures
import copy
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np

from arm.piper_ctrl_by_sdk_flat_hand import PiperFlatHandSDK
from arm import flat_handeye
from arm.linker_hand import LinkerHand
from camera.camera_api import Camera
from object_detect.detect import (
    detect_objects_in_frame,
    load_model,
    draw_box,
)
from utils.config_getter import get_config_value
from utils.cv2_display import show_image, poll_key, destroy_all_windows

# ---- 平抓标定质量自检阈值（只警告、不拦截；用户已知情并决定不重采）----
# L 命中搜索边界 [0, 0.25]m ±0.5mm 视为退化（臂长不可辨识）
HAND_EYE_L_MIN_M = 0.0
HAND_EYE_L_MAX_M = 0.25
HAND_EYE_L_BOUNDARY_TOL_M = 0.0005
# 联合模型 RMS 超过该值认为精度存疑（米）
HAND_EYE_RMS_WARN_M = 0.003


def report_mapping_quality(mapping, handeye_file: str) -> None:
    """打印平抓标定质量自检（L/RMS/φ0/点数/最差点），差则追加醒目 Warning。

    只提示、不拦截运行——用户已决定接受现有精度、不重采该标定点集。
    """
    residuals = getattr(mapping, "residuals_m", None)
    n_points = (
        mapping.n_points
        if mapping.n_points is not None
        else (len(residuals) if residuals is not None else 0)
    )
    print(
        f"已加载平抓标定: {handeye_file}  L={mapping.L_m * 1000:.1f}mm "
        f"RMS={mapping.rms_m * 1000:.2f}mm φ0={mapping.heading_offset_deg:.1f}° "
        f"点数={n_points}"
    )
    if residuals is not None and len(residuals) > 0:
        worst_index = int(np.argmax(residuals)) + 1  # 1 起算，按采集顺序
        print(
            f"  最差点: #{worst_index} {float(np.max(residuals)) * 1000:.1f}mm"
            f"（逐点残差 mm: {np.round(np.asarray(residuals) * 1000, 2).tolist()}）"
        )

    problems = []
    if (
        mapping.L_m <= HAND_EYE_L_MIN_M + HAND_EYE_L_BOUNDARY_TOL_M
        or mapping.L_m >= HAND_EYE_L_MAX_M - HAND_EYE_L_BOUNDARY_TOL_M
    ):
        problems.append("臂长 L 不可辨识（退化，命中搜索边界）")
    if mapping.rms_m > HAND_EYE_RMS_WARN_M:
        problems.append(
            f"RMS {mapping.rms_m * 1000:.2f}mm > {HAND_EYE_RMS_WARN_M * 1000:.0f}mm"
        )
    if problems:
        print("!" * 62)
        print("! Warning: 标定质量存疑：" + "；".join(problems))
        print(
            "! 跨区域可能存在厘米级位置误差——建议在 test 模式用铺开的点位复验"
            "或重采该标定点集"
        )
        print("! （用户已知情并决定不重采、接受现有精度；本警告不阻断运行）")
        print("!" * 62)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="调用目标识别和机械臂控制，使用灵巧手水平包络方式抓取。"
    )
    parser.add_argument(
        "--target",
        default='tomato',
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
    # 双钉走廊模式的朝向偏移（度）：平抓朝向 = 抓取点方位角 + 该偏移。
    # 来源 2026-09-10 示教走廊规律（示教基准姿态对应方位角 -4.2°），
    # 真机朝向不对就微调此值
    heading_offset_deg = float(
        get_config_value(
            "linker_hand_flat_heading_offset_deg", 4.2, raise_if_missing=False
        )
    )
    # 抓取接近方式（linker_hand_flat_approach_mode）：
    # - direct_down（默认）：正上方对准目标后垂直下抓（路径短、少中间状态，
    #   避免水平滑入时大拇指侧向扫过把物体推走，真机 2026-09-11 反馈）；
    # - slide_in：旧行为，后退 flat_approach_backoff_m 后水平滑入，一行切回。
    approach_mode = str(
        get_config_value(
            "linker_hand_flat_approach_mode", "direct_down", raise_if_missing=False
        )
    ).strip().lower()
    if approach_mode not in ("direct_down", "slide_in"):
        print(
            f"Warning: linker_hand_flat_approach_mode 配置无效({approach_mode})，"
            "按 direct_down 处理"
        )
        approach_mode = "direct_down"
    # 抓取模式（linker_hand_flat_catch_once）：
    # true（默认）＝完成**一次**抓取尝试（无论成功失败）后退出——调试/验证用；
    # false＝恢复连续抓取。未命中目标时本模式仍继续等待检测（"只抓一次"指
    # 抓到一次/尝试一次后退出，不是"看一眼就退"）。
    catch_once = bool(
        get_config_value("linker_hand_flat_catch_once", True, raise_if_missing=False)
    )
    # 平抓联合标定（像素 -> 掌心/法兰），由 arm/calibrate_arm_hand.py 生成；
    # 缺文件时明确提示先跑平抓标定
    handeye_file = get_config_value(
        "linker_hand_flat_handeye_file",
        os.path.join("arm", "hand-eye-data", "2d_homography_flat.npz"),
        raise_if_missing=False,
    )
    if not os.path.isabs(handeye_file):
        handeye_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), handeye_file
        )
    if not os.path.exists(handeye_file):
        print(f"未找到平抓标定文件: {handeye_file}")
        print(
            "请先运行平抓标定: python3 arm/calibrate_arm_hand.py"
            "（掌心对准画面点，J5/J6 平抓姿态）"
        )
        return
    mapping = flat_handeye.load_mapping(handeye_file)
    report_mapping_quality(mapping, handeye_file)

    # 直接实例化平抓适配类，构造参数与 Arm() 工厂用法一致；
    # 平抓欧拉角/J6 限位/接近退距/平抓高度偏移由 PiperFlatHandSDK
    # 从 config 的 linker_hand_flat_* / linker_hand_joint6_limit_deg 读取。
    arm = PiperFlatHandSDK()
    print(
        f"平抓工作高度 z={arm.flat_grasp_height_m:.3f} m"
        f"（linker_hand_flat_grasp_height_m，绝对值；桌面默认 {arm.desktop_height:.3f}，"
        f"抬高 {arm.catch_raise_height:.3f}）"
    )
    print(
        f"抓取接近模式: {approach_mode}（linker_hand_flat_approach_mode）——"
        + (
            "正上方对准→垂直下抓；接近退距 linker_hand_flat_approach_backoff_m 不生效"
            if approach_mode == "direct_down"
            else f"后退 {arm.flat_approach_backoff_m:.2f}m 后水平滑入（拇指可能扫到物体）"
        )
    )
    print(
        "抓取模式: "
        + (
            "单次（完成一次抓取尝试后退出，linker_hand_flat_catch_once=true）"
            if catch_once
            else "连续（linker_hand_flat_catch_once=false）"
        )
    )
    hand = LinkerHand()
    # 不再 arm.move_to_home()：构造函数已完成回 home + 原地固定手腕
    # （linker_hand_pin_on_start），再 home 一次会把刚固定的手腕转回原位
    hand.open_palm()
    cam = Camera(color=True, depth=False, undistort=True)
    models = [load_model(model_path) for model_path in model_paths]
    detections = []
    future = None
    once_submitted = False  # 单次模式：是否已提交过抓取任务（提交后只等它跑完）

    def catch_and_place(
        target_x: float,
        target_y: float,
        catch_rotate_rad: float,
        place_pos_: list[float | int],
        object_width_m: float,
    ) -> bool:
        """使用灵巧手水平包络方式执行一次完整抓取-放置。

        Args:
            target_x: 法兰目标 x 坐标，单位为米（由平抓标定映射预测并已含
                catch_offset 外偏）。
            target_y: 法兰目标 y 坐标，单位为米（同上）。
            catch_rotate_rad: 平抓朝向，单位为弧度（= 掌心方位角 + φ0，由
                linker_hand_flat_heading_offset_deg 决定），所有平抓运动按
                它把示教基准姿态绕世界 z 旋转；不再用物体长边角。
            place_pos_: 放置位置，格式为 `[x, y]` 或 `[x, y, z]`，单位为米。
            object_width_m: 物体近似宽度，单位为米（改用 SDK 握拳后仅作日志
                诊断；保留给"按宽度渐闭"方案使用）。

        Returns:
            整个流程是否成功。
        """
        if len(place_pos_) == 2:
            place_x, place_y = place_pos_
            # 放置高度用与抓取相同的平抓工作高度（绝对值，唯一来源）
            place_z = arm.flat_grasp_height_m
        elif len(place_pos_) == 3:
            place_x, place_y, place_z = place_pos_
        else:
            print("放置位置格式错误，应该是[x, y]或[x, y, z]")
            return False

        # 平抓工作高度：绝对值（config linker_hand_flat_grasp_height_m，默认
        # 0.11 = 原 0.15+0.01-0.05；2026-09-11 实机在 test 模式验证合适，
        # 抓取与之统一）；抬高/放置仍按 catch_raise_height / place_raise_height
        target_z = arm.flat_grasp_height_m
        safe_z = target_z + arm.catch_raise_height
        # 双钉走廊：平抓朝向 = 掌心方位角 + φ0（调用方由平抓标定映射算好传入）
        flat_heading_rad = float(catch_rotate_rad)
        # 接近方式（linker_hand_flat_approach_mode）：
        # - direct_down（默认）：正上方对准目标后垂直下降——路径短、少中间状态，
        #   避免水平滑入时大拇指侧向扫过把物体推走；
        # - slide_in：沿接近朝向后退 linker_hand_flat_approach_backoff_m 再水平滑入
        #   （旧行为，一行切回）。
        # 两种方式**最终位姿完全相同**（掌心落在目标、朝向 = 方位角 + φ0），
        # 区别只在路径（垂直 vs 侧向滑入）。
        if approach_mode == "slide_in":
            approach_x = target_x + arm.flat_approach_backoff_m * np.cos(
                flat_heading_rad
            )
            approach_y = target_y + arm.flat_approach_backoff_m * np.sin(
                flat_heading_rad
            )
            slide_in = True
        else:
            approach_x, approach_y = target_x, target_y
            slide_in = False
        # 放置段同理：放置点平抓朝向 = 放置点方位角 + 同一偏移
        place_heading_rad = np.arctan2(place_y, place_x) + np.deg2rad(
            heading_offset_deg
        )

        # 本周期开始前原地确保手腕在平抓构型（上一周期结束时已回 home，
        # 这里原地翻腕而不是到位置后再翻，避免带位姿翻腕的碰撞风险）
        arm.pin_wrist_in_place()

        # 手掌张开
        hand.open_palm()

        # 平抓姿态移到接近起点上方安全高度（direct_down 时即目标正上方）
        res = arm.move_to_flat(
            [approach_x, approach_y, safe_z],
            rot_rad=flat_heading_rad,
        )
        if not res:
            print("移动到接近起点上方失败，取消抓取")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s * 2)

        # 下降到抓取高度（direct_down 即垂直下抓）
        res = arm.move_to_flat(
            [approach_x, approach_y, target_z],
            rot_rad=flat_heading_rad,
        )
        if not res:
            print("下降到抓取高度失败，取消抓取")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        if slide_in:
            # 沿走廊朝向水平滑入到目标 xy（仅 slide_in 模式；direct_down 的
            # 接近点已在目标正上方，无需横向移动）
            res = arm.move_to_flat(
                [target_x, target_y, target_z],
                rot_rad=flat_heading_rad,
            )
            if not res:
                print("水平滑入到目标失败，取消抓取")
                arm.move_to_home()
                return False
        time.sleep(arm.catch_time_interval_s)

        # 使用 LinkerHand SDK 握拳闭合（O6 官方左手握拳姿态，finger_move 单帧
        # 下发；原按物体宽度渐闭的 hand.grasp_by_size(object_width_m) 仍保留在
        # LinkerHand 里，如需按宽度闭合，把下面一行改回即可）
        fist_pose = hand.close_fist()
        print(f"握拳闭合: {fist_pose}（物体宽度≈{object_width_m * 1000:.0f}mm，仅记录）")
        time.sleep(arm.catch_time_interval_s)

        # 原平抓姿态垂直抬起
        res = arm.move_to_flat(
            [target_x, target_y, safe_z],
            rot_rad=flat_heading_rad,
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

        # 保持平抓姿态平移到放置位置上方，避免翻转丢件（朝向 = 放置点方位角+偏移）
        res = arm.move_to_flat(
            [place_x, place_y, place_z + arm.place_raise_height],
            rot_rad=place_heading_rad,
        )
        if not res:
            print("移动到放置位置上方失败，取消放置")
            hand.open_palm()
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s * 2)

        # 灵巧手需要先下降到放置高度再松开，物体才能平稳落下
        res = arm.move_to_flat(
            [place_x, place_y, place_z],
            rot_rad=place_heading_rad,
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

        res = arm.move_to_flat(
            [place_x, place_y, place_z + arm.place_raise_height],
            rot_rad=place_heading_rad,
        )
        if not res:
            print("放置后抬起失败")
            arm.move_to_home()
            return False
        time.sleep(arm.catch_time_interval_s)

        # 全程平抓：不再切回默认垂直姿态，直接回 home；下一周期开始时会
        # 原地重新固定手腕（pin_wrist_in_place）
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
                    # 平抓联合标定映射：像素 -> 掌心（抓取接触点）/法兰
                    palm_xy, flange_xy = mapping.predict(u, v)
                    # 双钉走廊：朝向 = 掌心方位角 + φ0（物体长边角仅作参考/
                    # 记录，未来放开 J6 后恢复参与运动）
                    gripper_angle_rad = arm.gripper_angle_by_longer(
                        u, v, w, h, angle_deg
                    )
                    flat_heading_rad = np.arctan2(
                        palm_xy[1], palm_xy[0]
                    ) + np.deg2rad(heading_offset_deg)
                    # 物体近似宽度（米）：检测框短边两端在掌心平面上换算
                    short_edge_px = min(w, h)
                    side_palm_xy, _ = mapping.predict(
                        min(u + short_edge_px, frame_w - 1), v
                    )
                    object_width_m = float(
                        np.linalg.norm(side_palm_xy - palm_xy)
                    )
                    class_place_pos = copy.deepcopy(place_pos.get(class_name, None))
                    if class_place_pos is None or "pos" not in class_place_pos:
                        print("No placement location specified, place in origin.")
                        class_place_pos = {"pos": [palm_xy[0], palm_xy[1]]}
                    else:
                        match class_place_pos["pos"][0]:
                            case "x":
                                class_place_pos["pos"][0] = palm_xy[0]
                            case "-x":
                                class_place_pos["pos"][0] = -palm_xy[0]
                            case "y":
                                class_place_pos["pos"][0] = palm_xy[1]
                            case "-y":
                                class_place_pos["pos"][0] = -palm_xy[1]
                        match class_place_pos["pos"][1]:
                            case "x":
                                class_place_pos["pos"][1] = palm_xy[0]
                            case "-x":
                                class_place_pos["pos"][1] = -palm_xy[0]
                            case "y":
                                class_place_pos["pos"][1] = palm_xy[1]
                            case "-y":
                                class_place_pos["pos"][1] = -palm_xy[1]
                    if (
                        np.linalg.norm(
                            np.array(class_place_pos.get("pos"))
                            - np.array([palm_xy[0], palm_xy[1]])
                        )
                        < place_distance_threshold
                    ):
                        print(
                            f"Object {class_name} is too close to place position, skipping catch."
                        )
                    else:
                        future = executor.submit(
                            catch_and_place,
                            # 法兰目标 = 平抓标定预测的法兰点，沿走廊朝向再外偏
                            # 一份 catch_offset，避免刚好顶到物体（水平滑入的
                            # 接近起点退距在 catch_and_place 内按
                            # flat_approach_backoff_m 计算）
                            float(flange_xy[0] + offset * np.cos(flat_heading_rad)),
                            float(flange_xy[1] + offset * np.sin(flat_heading_rad)),
                            float(flat_heading_rad),
                            class_place_pos["pos"],
                            object_width_m,
                        )
                        if catch_once:
                            # 单次模式：只提交这一个任务，之后不再提交
                            once_submitted = True

            if default_gripper_aside_pos and (future is None or future.done()):
                # 移到旁边以免挡住视野（保持平抓姿态，朝向按旁边点方位角+φ0）
                aside_heading_rad = np.arctan2(
                    default_gripper_aside_pos[1], default_gripper_aside_pos[0]
                ) + np.deg2rad(heading_offset_deg)
                future = executor.submit(
                    arm.move_to_flat,
                    default_gripper_aside_pos,
                    aside_heading_rad,
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
            # 单次模式：提交过抓取任务后，等它跑完即退出循环（结果在循环外
            # 统一打印，随后走与 ESC/Ctrl-C 相同的收尾流程）
            if once_submitted and future is not None and future.done():
                break
        except KeyboardInterrupt:
            print("Exiting...")
            break

    # 单次模式的结果汇总（任务日志里已有各步骤成败原因，这里只给总判定；
    # 若用户中途手动退出且任务尚未结束，则不阻塞等待）
    if once_submitted and future is not None:
        if future.done():
            try:
                catch_result = future.result()
            except Exception as e:  # 任务内异常也归为失败
                catch_result = None
                print(f"抓取任务异常: {e}")
            print("=" * 62)
            print(
                "单次抓取结果: "
                + ("成功" if catch_result else "失败")
                + "（详见上方任务日志；随后回 home 并正常收尾）"
            )
            print("=" * 62)
        else:
            print("单次抓取任务尚未完成（已手动退出），跳过结果打印")

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
