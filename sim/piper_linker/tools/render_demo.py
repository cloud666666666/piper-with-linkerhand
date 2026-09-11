#!/usr/bin/env python3
# Description: MuJoCo 离屏渲染演示 —— 生成 README 用的静图与 GIF 动画（可重复生成）。
#
# 产物（默认输出到仓库 docs/media/）：
#   sim_iso.png          整机等轴测
#   sim_hand.png         手部特写
#   sim_flat_grasp.png   平抓姿态特写（臂 J1~J4=[-3.87, 90.29, -7.66, 0.76]°、J5=-69.0°、J6=6.75°，手张开）
#   hand_open_fist.gif   手：张开 → 握拳 → 张开（插值，20fps）
#   grasp_sequence.gif   抓取序列：home → 平抓姿态 → 垂直下降 → 握拳 → 抬起（末尾停 2 帧）
#
# 用法（必须离屏，Jetson/无显示器机器用 EGL）：
#   cd sim/piper_linker
#   MUJOCO_GL=egl python tools/render_demo.py                  # 用仓库默认路径
#   MUJOCO_GL=egl python tools/render_demo.py --width 960 --height 540 --fps 20 \
#       --outdir ../../docs/media
#
# 说明：
# - 手部驱动值与 server.py 的 GESTURES 保持一致（执行器顺序 = 拇指横摆/拇指弯曲/食指/中指/
#   无名指/小指的 MCP；单位为弧度；张开=全 0，握拳=[1.3, 0.58, 1.6, 1.6, 1.6, 1.6]）。
#   手指远端关节（thumb_ip / *_dip）由 scene.xml 的 equality 约束联动，无需单独驱动。
# - 模型的 6 个手部执行器 ctrllimited=False（ctrlrange 显示为 0,0 但不生效），可直接写 ctrl。
# - 垂直下降/抬升用 6 个臂关节的阻尼最小二乘 IK（mujoco.mj_jacBody）平移手掌，避免手写关节增量。
# - **取景**：动画先"预扫"一遍（只推进物理不渲染），把每一帧所有 body 的世界坐标取并集
#   包围盒，再据此定 lookat（并集中心）与 distance（= margin·包围球半径 / sin(fovy/2)），
#   保证每一帧整机都在框内并留边距；手部特写则只对 lh_* 手部 body 取并集。
#   渲完还会用相机位姿把所有 body 投影到 NDC，报告最贴边的 |ndc|（目标 ≤ 0.9，即 ≥10% 边距）。
# - EGL 在解释器退出时清理 GL 上下文可能报一次 EGLError（不影响已写出的文件），
#   脚本末尾用 os._exit(0) 跳过该析构报错。
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")  # 必须在 import mujoco 之前设置

import numpy as np

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(SIM_DIR))
SCENE = os.path.join(SIM_DIR, "scene.xml")

# 姿态常量（弧度）：与 README/config 中的平抓姿态一致
FLAT_ARM_DEG = [-3.87, 90.29, -7.66, 0.76, -69.0, 6.75]
HAND_OPEN = [0.0] * 6
HAND_FIST = [1.3, 0.58, 1.6, 1.6, 1.6, 1.6]  # server.py GESTURES['fist']
ARM_HOME_CTRL = [0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]  # scene/piper.xml 的 keyframe "home"


def smoothstep(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


class Demo:
    def __init__(self, width, height, fps, settle_steps=1500, substeps=None):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.width, self.height, self.fps = width, height, fps
        # 每帧推进的物理步数（timestep=1ms，20fps → 50 步 ≈ 实时）
        self.substeps = substeps or max(1, int(round(1000 / fps)))
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.hand_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")
        if self.hand_body < 0:  # 兜底：找第一个名字以 lh_ 开头的 body
            for i in range(self.model.nbody):
                n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                if n.startswith("lh_"):
                    self.hand_body = i
                    break
        self.flat_arm = [np.deg2rad(v) for v in FLAT_ARM_DEG]

    # ---------- 基础操作 ----------
    def set_ctrl(self, arm=None, hand=None):
        if arm is not None:
            self.data.ctrl[0:6] = arm
        if hand is not None:
            self.data.ctrl[6:12] = hand

    def settle(self, steps=None):
        for _ in range(steps or 1500):
            mujoco.mj_step(self.model, self.data)

    def advance(self):
        """推进一"帧"对应的物理步数。"""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def reset_at_home(self):
        """回到 keyframe home 并稳定（预扫后重新渲染用）。"""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.set_ctrl(arm=ARM_HOME_CTRL, hand=HAND_OPEN)
        mujoco.mj_forward(self.model, self.data)
        self.settle()

    def extent_points(self, body_ids=None):
        """可见几何的保守包围点集：geom 中心 ± geom_rbound。

        比 body 原点更接近真实外形（指尖/手掌网格离 body 原点的距离不可忽略）。
        """
        c = np.asarray(self.data.geom_xpos, dtype=float)
        r = np.asarray(self.model.geom_rbound, dtype=float)
        if body_ids is not None:
            mask = np.isin(self.model.geom_bodyid, np.asarray(body_ids))
            c, r = c[mask], r[mask]
        return np.concatenate([c - r[:, None], c + r[:, None]], axis=0)

    def ndc_margin(self, cam, points):
        """把世界点用当前相机投影到 NDC，返回最大 |ndc|（越小越居中；≥1 表示出框）。"""
        scn = self.renderer.scene
        c = scn.camera[0]
        pos = np.array(c.pos, dtype=float)
        fwd = np.array(c.forward, dtype=float)
        up = np.array(c.up, dtype=float)
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        view = np.stack([right, up, fwd])          # 相机坐标系（z 向前）
        rel = (points - pos) @ view.T
        fovy = np.deg2rad(self.model.vis.global_.fovy)
        aspect = self.width / self.height
        tan_y = np.tan(fovy / 2)
        ndc_x = (rel[:, 0] / np.maximum(rel[:, 2], 1e-6)) / (tan_y * aspect)
        ndc_y = (rel[:, 1] / np.maximum(rel[:, 2], 1e-6)) / tan_y
        return float(max(np.abs(ndc_x).max(), np.abs(ndc_y).max()))

    def hand_pos(self):
        return self.data.xpos[self.hand_body].copy()

    # ---------- 相机 ----------
    def make_camera(self, lookat=None, distance=0.9, azimuth=135, elevation=-18):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = self.model.stat.center if lookat is None else lookat
        cam.distance = float(distance)
        cam.azimuth = float(azimuth)
        cam.elevation = float(elevation)
        return cam

    def render(self, cam):
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    # ---------- 臂 IK：平移手掌到目标位置（阻尼最小二乘，只动 6 个臂关节） ----------
    def ik_hand_to(self, target, iters=300, damping=1e-3, step=1.0, tol=1e-4):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            err = target - self.data.xpos[self.hand_body]
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.hand_body)
            j = jacp[:, 0:6]  # 仅用 6 个臂关节
            # dq = J^T (J J^T + λI)^-1 err
            dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), err)
            for k in range(6):
                lo, hi = self.model.jnt_range[k]
                self.data.qpos[k] = float(np.clip(self.data.qpos[k] + step * dq[k], lo, hi))
            mujoco.mj_forward(self.model, self.data)


def union_bbox(frame_positions):
    """多帧 body 世界坐标的并集包围盒 -> (lo, hi)。"""
    allp = np.concatenate(frame_positions, axis=0)
    return allp.min(axis=0), allp.max(axis=0)


def frame_from_bbox(lo, hi, margin=1.10, fov_deg=None):
    """由包围盒算 (lookat_center, distance)：以 AABB 对角线为包围球，留 margin 边距。"""
    fov_deg = 45.0 if fov_deg is None else fov_deg
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo) / 2.0)
    distance = margin * radius / np.sin(np.deg2rad(fov_deg) / 2.0)
    return center, distance, radius


def max_ndc(demo, cam, points):
    """渲一帧并返回点集的最大 |ndc|（1.0 = 画面边缘）。"""
    demo.render(cam)
    scn = demo.renderer.scene
    c = scn.camera[0]
    pos = np.array(c.pos, dtype=float)
    fwd = np.array(c.forward, dtype=float); fwd /= np.linalg.norm(fwd)
    up = np.array(c.up, dtype=float); up /= np.linalg.norm(up)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    q = np.asarray(points, dtype=float) - pos
    z = np.maximum(q @ fwd, 1e-6)
    fovy = np.deg2rad(demo.model.vis.global_.fovy)
    tan_y = np.tan(fovy / 2)
    tan_x = tan_y * (demo.width / demo.height)
    return float(max(np.abs(q @ right / z / tan_x).max(), np.abs(q @ up / z / tan_y).max()))


def refine_distance(demo, cam, points, margin=1.10, iters=14):
    """把机位收紧到"恰好留 margin 边距"：以保守包围球距离为上界，对 distance 二分。

    判据用渲染后的真实 NDC 投影（max_ndc），目标 |ndc| = 1/margin（默认 0.909 ≈ 10% 边距）。
    """
    target = 1.0 / margin
    hi = float(cam.distance)          # 保守上界（包围球，必然满足）
    lo = 0.05
    if max_ndc(demo, cam, points) > target:
        return hi                     # 粗定机位已不满足，保守回退
    for _ in range(iters):
        mid = (lo + hi) / 2
        cam.distance = mid
        if max_ndc(demo, cam, points) <= target:
            hi = mid
        else:
            lo = mid
    cam.distance = hi
    return hi


def save_png(path, img, quality=95):
    from PIL import Image
    Image.fromarray(img).save(path, optimize=True)


def save_gif(path, frames, fps, colors=32):
    """写 GIF：PIL 自适应量化到 colors 色 + PIL 原生多帧写入（duration 单位 ms，精确）。"""
    from PIL import Image
    pal_frames = [
        Image.fromarray(f).convert("P", palette=Image.ADAPTIVE, colors=colors)
        for f in frames
    ]
    pal_frames[0].save(
        path,
        save_all=True,
        append_images=pal_frames[1:],
        duration=int(round(1000.0 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )


def stats_of(img):
    a = np.asarray(img, dtype=np.float32)
    return float(a.mean()), float(a.std())


def hand_bodies(model):
    """lh_* 手部 body 的 id 列表（用于手部特写取景）。"""
    ids = []
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("lh_"):
            ids.append(b)
    return ids


def run_hand_sequence(demo, cam=None, hand_ids=None):
    """手：张开→握拳→张开（插值）。

    返回 (帧图像列表, 每帧全部几何点集, 每帧手部几何点集)；cam=None 时只推进物理。
    """
    imgs, poss, hand_poss = [], [], []
    segments = [(HAND_OPEN, HAND_FIST, 12), (HAND_FIST, HAND_OPEN, 12)]
    demo.reset_at_home()  # 预扫与渲染两遍必须从同一状态出发

    def emit():
        if cam is not None:
            imgs.append(demo.render(cam))
        poss.append(demo.extent_points())
        hand_poss.append(demo.extent_points(hand_ids) if hand_ids else demo.extent_points())

    demo.set_ctrl(arm=ARM_HOME_CTRL, hand=HAND_OPEN)
    emit()
    for a, b, n in segments:
        for i in range(1, n + 1):
            demo.set_ctrl(hand=lerp(a, b, smoothstep(i / n)))
            demo.advance()
            emit()
    emit()  # 末尾停一帧（总时长里体现为停顿）
    return imgs, poss, hand_poss


def run_grasp_sequence(demo, cam=None):
    """抓取序列：home → 平抓姿态 → 垂直下降 4cm → 握拳 → 抬起。

    返回 (帧图像列表, 每帧全部 body 位置)。cam=None 时只推进物理（取景预扫用）。
    """
    imgs, poss = [], []
    demo.reset_at_home()  # 预扫与渲染两遍必须从同一状态出发

    def emit():
        if cam is not None:
            imgs.append(demo.render(cam))
        poss.append(demo.extent_points())

    demo.set_ctrl(arm=ARM_HOME_CTRL, hand=HAND_OPEN)
    emit()
    # (a) home → 平抓姿态
    for i in range(1, 17):
        demo.set_ctrl(arm=lerp(ARM_HOME_CTRL, demo.flat_arm, smoothstep(i / 16)), hand=HAND_OPEN)
        demo.advance()
        emit()
    # (b) 垂直下降到假想抓取高度（IK 平移手掌 -4cm）
    start = demo.hand_pos()
    goal = start - np.array([0.0, 0.0, 0.04])
    for i in range(1, 15):
        demo.ik_hand_to(start + (goal - start) * smoothstep(i / 14))
        demo.advance()
        emit()
    # (c) 握拳
    for i in range(1, 11):
        demo.set_ctrl(hand=lerp(HAND_OPEN, HAND_FIST, smoothstep(i / 10)))
        demo.advance()
        emit()
    # (d) 抬起
    for i in range(1, 15):
        demo.ik_hand_to(goal + (start - goal) * smoothstep(i / 14))
        demo.advance()
        emit()
    emit()  # 末尾停两帧
    emit()
    return imgs, poss


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 离屏渲染演示（PNG + GIF）")
    ap.add_argument("--width", type=int, default=800, help="静图宽度（默认 800）")
    ap.add_argument("--height", type=int, default=600, help="静图高度（默认 600）")
    ap.add_argument("--gif-width", type=int, default=800, help="GIF 宽度（默认同静图）")
    ap.add_argument("--gif-height", type=int, default=600, help="GIF 高度（默认同静图）")
    ap.add_argument("--fps", type=int, default=20, help="GIF 帧率（默认 20）")
    ap.add_argument("--gif-palette", type=int, default=32, help="GIF 量化色数（默认 32）")
    ap.add_argument("--margin", type=float, default=1.10, help="取景边距系数（默认 1.10 ≈ 留 10% 边距）")
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "docs", "media"), help="输出目录")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    results, framing = [], []

    # ============ 1) 静图（平抓姿态；按 body 并集取景） ============
    demo = Demo(args.width, args.height, args.fps)
    demo.set_ctrl(arm=demo.flat_arm, hand=HAND_OPEN)
    demo.settle()
    hi_ids = hand_bodies(demo.model)
    bodies = demo.extent_points()
    flat_pos = demo.hand_pos()

    # 整机等轴测：并集 = 全部 body；手部两张：并集 = lh_* 手部 body（+ 腕 link6）
    iso_c, iso_d, iso_r = frame_from_bbox(bodies.min(axis=0), bodies.max(axis=0), args.margin)
    hand_pts = demo.extent_points(hi_ids)
    hand_c, hand_d, hand_r = frame_from_bbox(hand_pts.min(axis=0), hand_pts.max(axis=0), args.margin)

    cam_iso = demo.make_camera(lookat=iso_c, distance=iso_d, azimuth=135, elevation=-20)
    refine_distance(demo, cam_iso, bodies, args.margin)
    img = demo.render(cam_iso)
    p = os.path.join(args.outdir, "sim_iso.png"); save_png(p, img)
    results.append(("sim_iso.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))
    framing.append(("sim_iso.png", iso_c, iso_r, cam_iso.distance, demo.ndc_margin(cam_iso, bodies), bodies))

    cam_hand = demo.make_camera(lookat=hand_c, distance=hand_d, azimuth=118, elevation=-28)
    refine_distance(demo, cam_hand, hand_pts, args.margin)
    img = demo.render(cam_hand)
    p = os.path.join(args.outdir, "sim_hand.png"); save_png(p, img)
    results.append(("sim_hand.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))
    framing.append(("sim_hand.png", hand_c, hand_r, cam_hand.distance, demo.ndc_margin(cam_hand, hand_pts), hand_pts))

    cam_flat = demo.make_camera(lookat=hand_c, distance=hand_d * 1.35, azimuth=152, elevation=-14)
    refine_distance(demo, cam_flat, hand_pts, args.margin)
    img = demo.render(cam_flat)
    p = os.path.join(args.outdir, "sim_flat_grasp.png"); save_png(p, img)
    results.append(("sim_flat_grasp.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))
    framing.append(("sim_flat_grasp.png", hand_c, hand_r * 1.35, cam_flat.distance,
                    demo.ndc_margin(cam_flat, hand_pts), hand_pts))

    # ============ 2) GIF：手 张开→握拳→张开 ============
    # 预扫（只推进物理）→ 并集包围盒 → 定机位 → 重新播一遍并渲染
    pre = Demo(args.gif_width, args.gif_height, args.fps)
    _, _, hand_only_list = run_hand_sequence(pre, hand_ids=hi_ids)
    hand_only = np.concatenate(hand_only_list, axis=0)
    c, d, r = frame_from_bbox(hand_only.min(axis=0), hand_only.max(axis=0), args.margin)

    demo2 = Demo(args.gif_width, args.gif_height, args.fps)
    cam2 = demo2.make_camera(lookat=c, distance=d, azimuth=118, elevation=-28)
    refine_distance(demo2, cam2, hand_only, args.margin)
    frames, poss2, hand2 = run_hand_sequence(demo2, cam=cam2, hand_ids=hi_ids)
    p = os.path.join(args.outdir, "hand_open_fist.gif")
    save_gif(p, frames, args.fps, args.gif_palette)
    ndc2 = max(demo2.ndc_margin(cam2, hp) for hp in hand2)  # 手部特写只要求手部在框内
    results.append(("hand_open_fist.gif", p, f"{args.gif_width}x{args.gif_height}",
                    f"{len(frames)}帧/{len(frames)/args.fps:.1f}s", *stats_of(frames[len(frames)//2])))
    framing.append(("hand_open_fist.gif", c, r, cam2.distance, ndc2, hand_only))

    # ============ 3) GIF：抓取序列 ============
    pre3 = Demo(args.gif_width, args.gif_height, args.fps)
    _, poss3 = run_grasp_sequence(pre3)
    lo, hi = union_bbox(poss3)
    c3, d3, r3 = frame_from_bbox(lo, hi, args.margin)

    demo3 = Demo(args.gif_width, args.gif_height, args.fps)
    cam3 = demo3.make_camera(lookat=c3, distance=d3, azimuth=140, elevation=-16)
    refine_distance(demo3, cam3, np.concatenate(poss3, axis=0), args.margin)
    frames, poss3r = run_grasp_sequence(demo3, cam=cam3)
    p = os.path.join(args.outdir, "grasp_sequence.gif")
    save_gif(p, frames, args.fps, args.gif_palette)
    ndc3 = max(demo3.ndc_margin(cam3, p_) for p_ in poss3r)
    results.append(("grasp_sequence.gif", p, f"{args.gif_width}x{args.gif_height}",
                    f"{len(frames)}帧/{len(frames)/args.fps:.1f}s", *stats_of(frames[len(frames)//2])))
    framing.append(("grasp_sequence.gif", c3, r3, cam3.distance, ndc3, np.concatenate(poss3, axis=0)))

    # ============ 验证报告 ============
    print("\n=== 取景（并集包围盒 → 机位）===")
    print(f"{'产物':<22}{'lookat 中心 (x,y,z)':<30}{'包围球半径':<12}{'最终distance':<13}{'最贴边 |ndc|':<12}留边")
    for name, c_, r_, d_, ndc_, pts_ in framing:
        flag = "  <-- 贴边/出框！" if ndc_ > 0.95 else ""
        margin_pct = (1.0 / ndc_ - 1.0) * 100 if ndc_ > 0 else float("inf")
        print(f"{name:<22}{str(np.round(c_, 3).tolist()):<30}{r_:<12.3f}{d_:>8.3f}   {ndc_:<10.3f}留边 {margin_pct:>5.1f}%{flag}")
    print("说明：|ndc| 为投影后最贴边点的归一化坐标（1.0=画面边缘）；留边 % = (1/|ndc|-1)。")

    print("\n=== 渲染产物（尺寸 / 帧数 / 体积 / 非空统计）===")
    print(f"{'文件':<22}{'分辨率':<12}{'帧数':<12}{'体积':<11}{'平均亮度':<10}{'标准差'}")
    ok = True
    for name, path, res, nframes, mean, std in results:
        size_kb = os.path.getsize(path) / 1024
        flag = "" if (std > 5 and mean > 5) else "  <-- 疑似全黑！"
        if flag:
            ok = False
        print(f"{name:<22}{res:<12}{nframes:<12}{size_kb:>8.0f}KB  {mean:<10.1f}{std:<8.1f}{flag}")
    print("结论:", "全部非空（std>5 且 mean>5）" if ok else "存在疑似空图，请检查相机参数")
    sys.stdout.flush()
    os._exit(0)  # 跳过 EGL 析构报错（文件已全部写出）


if __name__ == "__main__":
    main()
