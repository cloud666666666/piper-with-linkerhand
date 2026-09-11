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


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 离屏渲染演示（PNG + GIF）")
    ap.add_argument("--width", type=int, default=800, help="静图宽度（默认 800）")
    ap.add_argument("--height", type=int, default=600, help="静图高度（默认 600）")
    ap.add_argument("--gif-width", type=int, default=800, help="GIF 宽度（默认同静图）")
    ap.add_argument("--gif-height", type=int, default=600, help="GIF 高度（默认同静图）")
    ap.add_argument("--fps", type=int, default=20, help="GIF 帧率（默认 20）")
    ap.add_argument("--gif-palette", type=int, default=32, help="GIF 量化色数（默认 32，越小体积越小）")
    ap.add_argument("--outdir", default=os.path.join(REPO_ROOT, "docs", "media"), help="输出目录")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    results = []

    # ============ 1) 静图（用静图分辨率） ============
    demo = Demo(args.width, args.height, args.fps)
    demo.set_ctrl(arm=demo.flat_arm, hand=HAND_OPEN)
    demo.settle()
    hand_flat = demo.hand_pos()

    cam_iso = demo.make_camera(distance=1.05, azimuth=135, elevation=-20)
    img = demo.render(cam_iso)
    p = os.path.join(args.outdir, "sim_iso.png"); save_png(p, img)
    results.append(("sim_iso.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))

    cam_hand = demo.make_camera(lookat=hand_flat, distance=0.38, azimuth=118, elevation=-28)
    img = demo.render(cam_hand)
    p = os.path.join(args.outdir, "sim_hand.png"); save_png(p, img)
    results.append(("sim_hand.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))

    cam_flat = demo.make_camera(lookat=hand_flat, distance=0.50, azimuth=152, elevation=-14)
    img = demo.render(cam_flat)
    p = os.path.join(args.outdir, "sim_flat_grasp.png"); save_png(p, img)
    results.append(("sim_flat_grasp.png", p, f"{args.width}x{args.height}", "-", *stats_of(img)))

    # ============ 2) GIF：手 张开→握拳→张开 ============
    demo2 = Demo(args.gif_width, args.gif_height, args.fps)
    demo2.set_ctrl(arm=ARM_HOME_CTRL, hand=HAND_OPEN)
    demo2.settle()
    cam = demo2.make_camera(lookat=demo2.hand_pos(), distance=0.38, azimuth=118, elevation=-28)
    frames = []
    segments = [(HAND_OPEN, HAND_FIST, 12), (HAND_FIST, HAND_OPEN, 12)]
    cur = list(HAND_OPEN)
    demo2.set_ctrl(hand=cur)
    frames.append(demo2.render(cam))
    for a, b, n in segments:
        for i in range(1, n + 1):
            t = smoothstep(i / n)
            demo2.set_ctrl(hand=lerp(a, b, t))
            for _ in range(demo2.substeps):
                mujoco.mj_step(demo2.model, demo2.data)
            frames.append(demo2.render(cam))
    # 回到张开后再停 2 帧
    frames.extend([frames[-1]] * 2)
    p = os.path.join(args.outdir, "hand_open_fist.gif")
    save_gif(p, frames, args.fps, args.gif_palette)
    results.append(("hand_open_fist.gif", p, f"{args.gif_width}x{args.gif_height}",
                    f"{len(frames)}帧/{len(frames)/args.fps:.1f}s", *stats_of(frames[len(frames)//2])))

    # ============ 3) GIF：抓取序列 home → 平抓 → 下降 → 握拳 → 抬起 ============
    demo3 = Demo(args.gif_width, args.gif_height, args.fps)
    demo3.set_ctrl(arm=ARM_HOME_CTRL, hand=HAND_OPEN)
    demo3.settle()
    cam3 = demo3.make_camera(distance=1.0, azimuth=140, elevation=-16)
    frames = []
    frames.append(demo3.render(cam3))

    # (a) home → 平抓姿态（插值臂关节）
    for i in range(1, 17):
        t = smoothstep(i / 16)
        demo3.set_ctrl(arm=lerp(ARM_HOME_CTRL, demo3.flat_arm, t), hand=HAND_OPEN)
        for _ in range(demo3.substeps):
            mujoco.mj_step(demo3.model, demo3.data)
        frames.append(demo3.render(cam3))
    # (b) 垂直下降到假想抓取高度（IK 平移手掌 -4cm）
    start = demo3.hand_pos()
    goal = start - np.array([0.0, 0.0, 0.04])
    for i in range(1, 15):
        t = smoothstep(i / 14)
        demo3.ik_hand_to(start + (goal - start) * t)
        for _ in range(demo3.substeps):
            mujoco.mj_step(demo3.model, demo3.data)
        frames.append(demo3.render(cam3))
    # (c) 握拳（手插值）
    for i in range(1, 11):
        t = smoothstep(i / 10)
        demo3.set_ctrl(hand=lerp(HAND_OPEN, HAND_FIST, t))
        for _ in range(demo3.substeps):
            mujoco.mj_step(demo3.model, demo3.data)
        frames.append(demo3.render(cam3))
    # (d) 抬起回下降前高度
    for i in range(1, 15):
        t = smoothstep(i / 14)
        demo3.ik_hand_to(goal + (start - goal) * t)
        for _ in range(demo3.substeps):
            mujoco.mj_step(demo3.model, demo3.data)
        frames.append(demo3.render(cam3))
    frames.extend([frames[-1]] * 2)  # 末尾停两帧
    p = os.path.join(args.outdir, "grasp_sequence.gif")
    save_gif(p, frames, args.fps, args.gif_palette)
    results.append(("grasp_sequence.gif", p, f"{args.gif_width}x{args.gif_height}",
                    f"{len(frames)}帧/{len(frames)/args.fps:.1f}s", *stats_of(frames[len(frames)//2])))

    # ============ 验证报告 ============
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
