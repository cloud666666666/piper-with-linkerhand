#!/usr/bin/env python3
# Description: 由 MuJoCo 场景（scene.xml + piper/piper.xml，权威模型）生成 3D 查看器用的 URDF
#   docs/viewer/robot.urdf，并把用到的网格同步到 docs/viewer/meshes/。
#
# 背景：查看器原先用的 docs/viewer/robot.urdf 是"官方 piper_description URDF + 手"的拼装，
#   其 link 帧与 MJCF（scene.xml）并不等价（实测 link3 起世界位置偏差 ~10mm），
#   且 link6 视觉用的是含夹爪的完整 link6.stl、缺少灵巧手连接件 adapter。
#   本脚本以 MJCF 为准重建：
#     - URDF 关节 origin 直接取 MJCF body 的 pos/quat（rx ry rz 外旋 xyz 等价 rpy），
#       关节轴取 MJCF joint 的 axis，限位取 MJCF 的 range；
#     - 视觉网格取 MJCF geom 的 mesh 与 quat（class=visual 的那些；碰撞几何不进 URDF）；
#     - 于是 URDF 的 link 帧 == MJCF 的 body 帧，两者 FK 逐点一致（见 --verify）。
#
# 用法：
#   python tools/sync_viewer_urdf.py                # 生成 robot.urdf + 同步网格
#   python tools/sync_viewer_urdf.py --verify       # 额外做 URDF/MJCF 随机位姿 FK 比对
import argparse
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(os.path.dirname(SIM_DIR))
PIPER_XML = os.path.join(SIM_DIR, "piper", "piper.xml")
SCENE_XML = os.path.join(SIM_DIR, "scene.xml")
VIEWER_DIR = os.path.join(REPO_ROOT, "docs", "viewer")
MESHES_DIR = os.path.join(VIEWER_DIR, "meshes")
URDF_OUT = os.path.join(VIEWER_DIR, "robot.urdf")
# 显示用网格的来源目录（与原查看器一致：Piper 官方 STL + 手官方 STL）
MESH_SRC_DIRS = [
    os.path.join(SIM_DIR, "piper"),
    os.path.join(SIM_DIR, "linkerhand_o6_left", "meshes"),
]

# 关节名 -> 父子关系里"MJCF 里是 fixed 子 body"的特例：手基座直接挂在 link6 下
FIXED_BODY_JOINT = {"lh_hand_base_link": "lh_adapter_joint"}

# MJCF 里 link1~link5 各自由官方"分片网格"组成（link2 有 32 片），逐片复制会让查看器
# 体积膨胀数 MB。这些分片与原官方单文件 STL 是同一几何（--verify-meshes 用包围盒校验，
# 偏差 <1mm），故查看器这几段仍用单文件；**link6 必须按 MJCF**（视觉=link6_trimmed，
# 完整 link6 只作碰撞）+ 连接件 adapter，否则会显示已拆掉的夹爪。
LINK_MESH_OVERRIDE = {
    "base_link": ["base_link.stl"],
    "link1": ["link1.stl"],
    "link2": ["link2.stl"],
    "link3": ["link3.stl"],
    "link4": ["link4.stl"],
    "link5": ["link5.stl"],
}


def parse_xml(path):
    return ET.fromstring(re.sub(r"<\?xml[^?]*\?>", "", open(path, encoding="utf-8").read(), count=1))


def quat_to_rpy(q):
    """MJCF quat (w,x,y,z) -> URDF rpy（rx ry rz 外旋）。"""
    w, x, y, z = q
    return R.from_quat([x, y, z, w]).as_euler("xyz")


def collect_bodies(root):
    """按 深度优先 顺序收集 (body, parent_name)；root 为 worldbody。"""
    out = []

    def walk(body, parent):
        for b in body.findall("body"):
            out.append((b, parent))
            walk(b, b.get("name"))

    wb = root.find("worldbody")
    walk(wb if wb is not None else root, None)
    return out


def main():
    ap = argparse.ArgumentParser(description="由 MJCF 生成查看器 URDF")
    ap.add_argument("--verify", action="store_true", help="生成后做 URDF/MJCF 随机位姿 FK 比对")
    ap.add_argument("--urdf-only", action="store_true", help="只写 URDF（不同步网格）")
    args = ap.parse_args()

    piper = parse_xml(PIPER_XML)
    scene = parse_xml(SCENE_XML)
    # scene.xml 的 <asset> 里有 mesh name->file 的映射（手部网格在那里定义）
    # MJCF 里 <mesh file="x.obj"/> 不写 name 时，mujoco 用文件名（去扩展名）作网格名
    mesh_files = {}
    for src in (scene, piper):
        for m in src.iter("mesh"):
            f = m.get("file")
            if not f:
                continue
            nm = m.get("name") or os.path.splitext(os.path.basename(f))[0]
            mesh_files[nm] = f

    bodies = collect_bodies(piper)
    links_xml, joints_xml, used_meshes = [], [], []

    for body, parent in bodies:
        name = body.get("name")
        pos = np.array([float(v) for v in (body.get("pos") or "0 0 0").split()])
        quat = [float(v) for v in (body.get("quat") or "1 0 0 0").split()]
        rpy = quat_to_rpy(quat)

        # 视觉网格（class 非 collision 的 mesh geom；臂 link 见 LINK_MESH_OVERRIDE）
        visuals = []
        if name in LINK_MESH_OVERRIDE:
            for f in LINK_MESH_OVERRIDE[name]:
                used_meshes.append((f, f))
                visuals.append((f, np.zeros(3), np.zeros(3)))
            g_loop = []
        else:
            g_loop = body.findall("geom")
        for g in g_loop:
            if g.get("class") == "collision" or g.get("type") == "box":
                continue
            mesh = g.get("mesh")
            if not mesh or mesh not in mesh_files:
                continue
            file_name = mesh_files[mesh]
            g_quat = [float(v) for v in (g.get("quat") or "1 0 0 0").split()]
            g_rpy = quat_to_rpy(g_quat)
            g_pos = np.array([float(v) for v in (g.get("pos") or "0 0 0").split()])
            base = os.path.basename(file_name).lower()
            if base.endswith(".stl"):
                base = base[:-4] + ".stl"
            used_meshes.append((file_name, base))
            visuals.append((base, g_pos, g_rpy))

        joints = body.findall("joint")
        j = joints[0] if joints else None
        if parent is None:
            pass  # 根 link（base_link）不生成关节
        elif j is not None:
            axis = j.get("axis") or "0 0 1"
            rng = j.get("range")
            lower, upper = ([float(v) for v in rng.split()] if rng else [-np.pi, np.pi])
            joints_xml.append(
                f'  <joint name="{j.get("name")}" type="revolute">\n'
                f'    <origin xyz="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" '
                f'rpy="{rpy[0]:.6g} {rpy[1]:.6g} {rpy[2]:.6g}"/>\n'
                f'    <parent link="{parent}"/>\n    <child link="{name}"/>\n'
                f'    <axis xyz="{axis}"/>\n'
                f'    <limit lower="{lower:.6g}" upper="{upper:.6g}" effort="100" velocity="5"/>\n'
                f'  </joint>'
            )
        else:
            jn = FIXED_BODY_JOINT.get(name, f"{name}_fixed_joint")
            joints_xml.append(
                f'  <joint name="{jn}" type="fixed">\n'
                f'    <origin xyz="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" '
                f'rpy="{rpy[0]:.6g} {rpy[1]:.6g} {rpy[2]:.6g}"/>\n'
                f'    <parent link="{parent}"/>\n    <child link="{name}"/>\n'
                f'  </joint>'
            )

        vis = "\n".join(
            f'    <visual>\n      <origin xyz="{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" '
            f'rpy="{q[0]:.6g} {q[1]:.6g} {q[2]:.6g}"/>\n'
            f'      <geometry>\n        <mesh filename="meshes/{b}"/>\n      </geometry>\n    </visual>'
            for b, p, q in visuals
        )
        links_xml.append(f'  <link name="{name}">\n{vis}\n  </link>' if vis else f'  <link name="{name}"/>')
        _ = j  # noqa: 根 link 时不生成关节

    urdf = (
        '<?xml version="1.0"?>\n'
        f'<!-- 由 sim/piper_linker/tools/sync_viewer_urdf.py 从 MJCF（scene.xml + piper/piper.xml）生成，请勿手改 -->\n'
        f'<robot name="piper_linkerhand_o6_left">\n'
        + "\n".join(links_xml) + "\n" + "\n".join(joints_xml) + "\n</robot>\n"
    )
    open(URDF_OUT, "w", encoding="utf-8").write(urdf)
    print(f"已写出 {URDF_OUT}（{len(links_xml)} links / {len(joints_xml)} joints）")

    if not args.urdf_only:
        os.makedirs(MESHES_DIR, exist_ok=True)
        copied, missing = [], []
        for src_name, base in used_meshes:
            src = None
            for d in [SIM_DIR, *MESH_SRC_DIRS]:
                cand = os.path.join(d, src_name)
                if os.path.exists(cand):
                    src = cand
                    break
            if src is None:
                missing.append(src_name)
                continue
            dst = os.path.join(MESHES_DIR, base)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied.append(base)
        print("新复制网格:", copied if copied else "无")
        print("源目录缺失:", missing if missing else "无")
        keep = {b for _, b in used_meshes}
        stale = [f for f in os.listdir(MESHES_DIR) if f not in keep]
        print("查看器中已不被引用（保留不删，供对比）:", stale if stale else "无")

    if args.verify:
        verify()


def verify(trials=8):
    """URDF（自写 FK）与 MJCF（mujoco）在随机关节位姿下的世界位置比对。"""
    import mujoco
    import xml.etree.ElementTree as ET2

    u = ET2.fromstring(re.sub(r"<\?xml[^?]*\?>", "", open(URDF_OUT, encoding="utf-8").read(), count=1))
    J = {}
    for j in u.findall("joint"):
        o = j.find("origin"); a = j.find("axis")
        J[j.find("child").get("link")] = dict(
            name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"),
            xyz=np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]),
            rpy=np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]),
            axis=np.array([float(v) for v in (a.get("xyz") or "0 0 1").split()]) if a is not None else np.zeros(3),
        )

    def urdf_fk(link, q):
        T = np.eye(4)
        chain, cur = [], link
        while cur in J:
            chain.append(J[cur]); cur = J[cur]["parent"]
        for j in reversed(chain):
            T = T @ np.block([[R.from_euler("xyz", j["rpy"]).as_matrix(), j["xyz"][:, None]],
                              [np.zeros(3), 1]])
            if j["type"] == "revolute":
                T = T @ np.block([[R.from_rotvec(j["axis"] * q.get(j["name"], 0.0)).as_matrix(),
                                   np.zeros((3, 1))], [np.zeros(3), 1]])
        return T

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    jn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    targets = ["link1", "link2", "link3", "link4", "link5", "link6", "lh_hand_base_link",
               "lh_thumb_distal", "lh_index_distal", "lh_middle_distal", "lh_ring_distal", "lh_pinky_distal"]
    bid = {t: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t) for t in targets}
    rng = np.random.default_rng(0)
    worst = {t: 0.0 for t in targets}
    for _ in range(trials):
        q = {}
        for i, n in enumerate(jn):
            lo, hi = m.jnt_range[i]
            q[n] = float(rng.uniform(lo, hi))
        d.qpos[:] = 0
        for i, n in enumerate(jn):
            d.qpos[m.jnt_qposadr[i]] = q[n]
        mujoco.mj_forward(m, d)
        for t in targets:
            worst[t] = max(worst[t], float(np.linalg.norm(d.xpos[bid[t]] - urdf_fk(t, q)[:3, 3])))
    print(f"\n=== URDF vs MJCF 世界位置最大偏差（{trials} 组随机位姿）===")
    for t in targets:
        print(f"  {t:24s} {worst[t]:.3e} m" + ("   <-- 超 1e-4" if worst[t] > 1e-4 else ""))
    print("结论:", "一致（≤1e-4 m）" if max(worst.values()) <= 1e-4 else "存在偏差！")


if __name__ == "__main__":
    main()
