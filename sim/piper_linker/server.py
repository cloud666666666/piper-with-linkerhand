"""Web-based MuJoCo viewer for the Piper + Linker Hand O6 scene.

Aims to replace the desktop `mujoco.viewer` for this model:

  - MJPEG live stream (native <img>, no JS decode)
  - mouse orbit / pan / zoom, camera presets, follow-hand camera
  - play / pause / single-step / reset / speed control
  - per-actuator sliders (6 arm + 6 hand) with live joint readouts
  - gesture and arm-pose presets
  - visual toggles: contact points, contact forces, wireframe, transparent,
    collision geometry group
  - live state panel: time, contacts, total contact force, end-effector
    position, stream FPS + force sparkline

Start:
    MUJOCO_GL=egl uv run server.py
Then open the printed URL from any device on the LAN.
"""

import asyncio
import io
import json
import os
import socket
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 640, 480
STREAM_FPS = 30

app = FastAPI(title="Piper Linker Hand O6 MuJoCo")
model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
data = mujoco.MjData(model)

# ---------------------------------------------------------------------------
# shared sim/render state (everything runs on the event-loop thread, but the
# lock also keeps threadpool endpoints such as /state consistent)
# ---------------------------------------------------------------------------
lock = threading.Lock()
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0.2, 0.0, 0.35]
camera.distance = 1.3
camera.azimuth = 135
camera.elevation = -25

vis = {  # visualization toggles
    "contact_points": False,
    "contact_forces": False,
    "joint_axes": False,
    "inertia": False,
    "transparent": False,
    "collisions": False,   # geom group 3
}
follow_hand = False
paused = False
speed = 1.0
step_once = 0  # ms of physics to advance when paused (single-step button)
frame_count = 0
stream_fps = 0.0

renderer = mujoco.Renderer(model, WIDTH, HEIGHT)
vis_opt = mujoco.MjvOption()  # wireframe / geom-group switches live here
from numpy import zeros as np_zeros  # local helper for contact-force buffer


def reset_sim() -> None:
    key = model.key("home")
    with lock:
        mujoco.mj_resetData(model, data)
        data.qpos[: len(key.qpos)] = key.qpos
        data.ctrl[:] = 0.0
        data.ctrl[: len(key.ctrl)] = key.ctrl
        mujoco.mj_forward(model, data)


reset_sim()

# ---------------------------------------------------------------------------
# background physics loop: advances simulation by wall-clock time * speed
# ---------------------------------------------------------------------------
_step_acc = 0.0


async def sim_loop() -> None:
    global _step_acc, step_once
    last = time.monotonic()
    while True:
        await asyncio.sleep(0.002)
        now = time.monotonic()
        wall = now - last
        last = now
        if paused and step_once <= 0:
            _step_acc = 0.0
            continue
        sim_dt = wall * speed if not paused else 0.0
        sim_dt += step_once / 1000.0
        step_once = 0
        _step_acc += sim_dt
        steps = int(_step_acc / model.opt.timestep)
        _step_acc -= steps * model.opt.timestep
        if steps <= 0:
            continue
        steps = min(steps, 1000)
        with lock:
            for _ in range(steps):
                mujoco.mj_step(model, data)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(sim_loop())
    # Robonix ROS bridge (可选): 设置 BRIDGE_WS_URL 后连接 bridge_node.py
    bridge_url = os.environ.get("BRIDGE_WS_URL", "")
    if bridge_url:
        import logging
        logging.getLogger("bridge_client").setLevel(logging.INFO)
        import bridge_client
        bridge_client.start(model=model, data=data, lock=lock,
                            reset_fn=reset_sim, url=bridge_url)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def apply_vis_flags() -> None:
    scene = renderer.scene
    scene.flags[:] = 0
    if vis["contact_points"]:
        scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    if vis["contact_forces"]:
        scene.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    if vis["transparent"]:
        scene.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
    if vis["joint_axes"]:
        scene.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
    if vis["inertia"]:
        scene.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 1
    vis_opt.geomgroup[3] = 1 if vis["collisions"] else 0
    vis_opt.geomgroup[2] = 1


def render_jpeg() -> bytes:
    global frame_count
    with lock:
        if follow_hand:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")
            camera.lookat = camera.lookat * 0.85 + data.xpos[bid] * 0.15
        apply_vis_flags()
        renderer.update_scene(data, camera=camera, scene_option=vis_opt)
        rgb = renderer.render()
        frame_count += 1
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
@app.get("/model.json")
def model_info():
    actuators = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.jnt_range[jid]
        actuators.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "joint": jname,
            "range": [float(lo), float(hi)],
            "qadr": int(model.jnt_qposadr[jid]),
        })
    return {"nu": model.nu, "nq": model.nq, "timestep": model.opt.timestep,
            "version": mujoco.__version__, "actuators": actuators}


def snapshot() -> dict:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")
    wrench = np_zeros(6)
    with lock:
        force_total = 0.0
        for i in range(data.ncon):
            mujoco.mj_contactForce(model, data, i, wrench)
            force_total += float(wrench[0] ** 2 + wrench[1] ** 2 + wrench[2] ** 2) ** 0.5
        return {
            "time": float(data.time),
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(),
            "ncon": int(data.ncon),
            "force": force_total,
            "eef": data.xpos[bid].tolist(),
            "paused": paused,
            "speed": speed,
            "follow": follow_hand,
            "vis": dict(vis),
            "fps": stream_fps,
        }


@app.get("/state")
def get_state():
    return JSONResponse(snapshot())


@app.post("/control")
async def control(payload: dict):
    values = payload.get("ctrl")
    if isinstance(values, list):
        with lock:
            for i, v in enumerate(values[: model.nu]):
                data.ctrl[i] = float(v)
    elif isinstance(payload.get("i"), int) and "v" in payload:
        i = payload["i"]
        if 0 <= i < model.nu:
            with lock:
                data.ctrl[i] = float(payload["v"])
    return JSONResponse({"ok": True})


@app.post("/sim")
async def sim_control(payload: dict):
    global paused, speed, step_once
    action = payload.get("action")
    if action == "pause":
        paused = not paused
    elif action == "play":
        paused = False
    elif action == "step":
        step_once = int(payload.get("ms", 100))
    elif action == "reset":
        reset_sim()
    elif action == "speed":
        speed = max(0.05, min(float(payload.get("value", 1.0)), 4.0))
    return JSONResponse({"paused": paused, "speed": speed})


@app.post("/camera")
async def camera_control(payload: dict):
    global follow_hand
    op = payload.get("op")
    with lock:
        if op == "orbit":
            camera.azimuth -= payload.get("dx", 0.0) * 0.4
            camera.elevation = max(-89.0, min(89.0, camera.elevation + payload.get("dy", 0.0) * 0.4))
        elif op == "pan":
            az, el, dist = camera.azimuth, camera.elevation, camera.distance
            azr, elr = np_deg(az), np_deg(el)
            right = [-sin(azr), cos(azr), 0.0]
            fwd = [cos(elr) * cos(azr), cos(elr) * sin(azr), sin(elr)]
            up = cross(right, fwd)
            s = dist * 0.0016
            camera.lookat = camera.lookat + right * (-payload.get("dx", 0.0) * s) + up * (payload.get("dy", 0.0) * s)
        elif op == "zoom":
            camera.distance = max(0.25, min(camera.distance * (1.0 + payload.get("dz", 0.0) * 0.001), 8.0))
        elif op == "preset":
            name = payload.get("name")
            presets = {
                "default": (135, -25, [0.2, 0.0, 0.35], 1.3),
                "front": (180, -12, [0.25, 0.0, 0.3], 1.5),
                "side": (90, -10, [0.25, 0.0, 0.3], 1.5),
                "top": (135, -88, [0.2, 0.0, 0.2], 1.8),
                "closeup": (135, -20, [0.37, 0.0, 0.45], 0.45),
            }
            if name in presets:
                az, el, look, dist = presets[name]
                camera.azimuth, camera.elevation = az, el
                camera.lookat[:] = look
                camera.distance = dist
        elif op == "follow":
            follow_hand = bool(payload.get("on", not follow_hand))
    return JSONResponse({"ok": True})


@app.post("/vis")
async def vis_control(payload: dict):
    key = payload.get("key")
    if key in vis:
        vis[key] = bool(payload.get("on", not vis[key]))
    return JSONResponse({"vis": dict(vis)})


# math helpers (kept local & tiny; numpy avoided in hot endpoints)
def np_deg(deg):  # noqa: D401 - named for clarity, radians actually
    import math
    return deg * math.pi / 180.0


def sin(x):
    import math
    return math.sin(x)


def cos(x):
    import math
    return math.cos(x)


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def find_in_xml(name: str, kind: str):
    """在两个模型文件里定位 <body>/<joint>/<geom> 的行号, 返回 (文件, 行号1-based, 上下文)."""
    targets = {
        "body": f'<body name="{name}"',
        "joint": f'<joint name="{name}"',
        "geom": f'name="{name}"',
    }
    for fname in ["piper/piper.xml", "scene.xml"]:
        try:
            lines = (ROOT / fname).read_text().split("\n")
        except OSError:
            continue
        for i, ln in enumerate(lines):
            if targets[kind] in ln:
                lo = max(0, i - 2)
                hi = min(len(lines), i + 3)
                return fname, i + 1, "\n".join(f"{j+1:4d}| {lines[j]}" for j in range(lo, hi))
    return None, None, None


@app.post("/pick")
async def pick(payload: dict):
    """点击查询: 像素坐标 → 部件/关节 + 对应 XML 位置"""
    x = int(payload.get("x", -1)); y = int(payload.get("y", -1))
    if not (0 <= x < WIDTH and 0 <= y < HEIGHT):
        return JSONResponse({"error": "坐标越界"}, status_code=400)
    with lock:
        apply_vis_flags()
        renderer.update_scene(data, camera=camera)
        renderer.enable_segmentation_rendering()
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
    body_id = int(seg[y, x, 1]) - 1
    geom_id = int(seg[y, x, 0]) - 1
    if body_id < 0:
        return {"body": None, "hint": "点击处是背景"}
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) if geom_id >= 0 else None
    parent_id = int(model.body_parentid[body_id])
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id else "world"
    joints = []
    qpos_val = None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) == body_id:
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = model.jnt_range[j]
            f, ln, _ = find_in_xml(jn, "joint")
            joints.append({"name": jn, "range": [float(lo), float(hi)],
                           "file": f, "line": ln})
            qpos_val = float(data.qpos[model.jnt_qposadr[j]])
    f, ln, excerpt = find_in_xml(body, "body")
    return {"body": body, "geom": geom, "parent": parent, "joints": joints,
            "xml": {"file": f, "line": ln, "excerpt": excerpt},
            "qpos": qpos_val}


@app.post("/reload")
async def reload_model():
    """热重载: 重新解析 scene.xml, 保留相机/可视化/暂停状态"""
    global model, data, renderer, _last_frame_time
    try:
        with lock:
            new_model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    key = new_model.key("home")
    new_data = mujoco.MjData(new_model)
    new_data.qpos[: len(key.qpos)] = key.qpos
    new_data.ctrl[: len(key.ctrl)] = key.ctrl
    mujoco.mj_forward(new_model, new_data)
    with lock:
        model = new_model
        data = new_data
        renderer = mujoco.Renderer(model, WIDTH, HEIGHT)
        _last_frame_time = time.monotonic()
    return {"ok": True, "nq": model.nq, "nu": model.nu}


@app.get("/xml")
def view_xml(file: str, line: int):
    """浏览器内查看 XML 并高亮指定行 (点击识别跳转用)"""
    safe = file in ("piper/piper.xml", "scene.xml", "linkerhand_o6_left/linkerhand_o6_left.xml")
    if not safe:
        return JSONResponse({"error": "file not allowed"}, status_code=403)
    lines = (ROOT / file).read_text().split("\n")
    lo, hi = max(1, line - 8), min(len(lines), line + 12)
    rows = []
    for i in range(lo, hi + 1):
        mark = ' style="background:#2a3550"' if i == line else ""
        rows.append(f'<tr{mark}><td style="color:#8b93a3;padding-right:10px;text-align:right;user-select:none">{i}</td>'
                    f'<td><pre style="margin:0">{lines[i-1].replace("&","&amp;").replace("<","&lt;")}</pre></td></tr>')
    page = ('<!doctype html><meta charset="utf-8"><title>' + file + ':' + str(line) + '</title>'
            '<body style="background:#16191f;color:#dde3ec;font:12.5px/1.5 ui-monospace,monospace;margin:16px">'
            '<div style="color:#8b93a3;margin-bottom:8px">' + file + ' · 第 ' + str(line) + ' 行高亮</div>'
            '<table style="border-collapse:collapse">' + "".join(rows) + '</table></body>')
    return HTMLResponse(page)


@app.get("/stream.mjpg")
async def stream():
    global stream_fps

    async def gen():
        global stream_fps
        boundary = b"--frame\r\n"
        last_yield = time.monotonic()
        try:
            while True:
                jpg = render_jpeg()
                yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: " +
                       str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                now = time.monotonic()
                stream_fps = 0.9 * stream_fps + 0.1 * (1.0 / max(now - last_yield, 1e-6))
                delay = 1.0 / STREAM_FPS - (time.monotonic() - now)
                last_yield = time.monotonic()
                await asyncio.sleep(max(delay, 0.0))
        except asyncio.CancelledError:
            return

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="zh">
<meta charset="utf-8"><title>Piper + Linker Hand O6 · MuJoCo</title>
<style>
:root{--bg:#16191f;--panel:#1e222a;--line:#2c313c;--txt:#dde3ec;--dim:#8b93a3;--acc:#4f8cff;--ok:#3fb96f}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font:13px/1.45 system-ui,sans-serif;margin:0;display:flex;height:100vh;overflow:hidden}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#stage{flex:1;position:relative;background:#0c0e12;min-height:0;display:flex;align-items:center;justify-content:center}
#stage img{max-width:100%;max-height:100%;user-select:none;-webkit-user-drag:none;cursor:grab}
#stage img:active{cursor:grabbing}
#hud{position:absolute;top:10px;left:12px;background:rgba(10,12,16,.72);padding:6px 10px;border-radius:8px;font-size:12px;color:var(--dim);pointer-events:none}
#hud b{color:var(--txt)}
#pausedBadge{position:absolute;top:10px;right:12px;background:#c0392b;color:#fff;padding:4px 10px;border-radius:6px;display:none;font-weight:600}
#toolbar{display:flex;gap:6px;align-items:center;padding:8px 10px;background:var(--panel);border-top:1px solid var(--line);flex-wrap:wrap}
#side{width:308px;background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:10px 12px 40px}
details{margin-bottom:10px;border:1px solid var(--line);border-radius:8px;background:#191d24}
details>summary{cursor:pointer;padding:8px 10px;font-weight:600;list-style:none}
details>summary::before{content:"▸ ";color:var(--dim)}
details[open]>summary::before{content:"▾ "}
details .body{padding:4px 10px 10px}
button{background:#252a34;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px}
button:hover{border-color:var(--acc);color:#fff}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.on{background:var(--ok);border-color:var(--ok);color:#fff}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.ctl{margin:7px 0}
.ctl .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--dim)}
.ctl .lab b{color:var(--txt);font-weight:500}
input[type=range]{width:100%;accent-color:var(--acc)}
.stats{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:12px;color:var(--dim);margin:6px 0}
.stats b{color:var(--txt);font-weight:500}
canvas{width:100%;background:#10131a;border-radius:6px}
.chip{display:inline-block;background:#10131a;border:1px solid var(--line);border-radius:5px;padding:2px 7px;margin:2px;cursor:pointer;font-size:11.5px}
.chip.on{border-color:var(--ok);color:var(--ok)}
kbd{background:#10131a;border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:11px}
.help{font-size:11.5px;color:var(--dim);line-height:1.7}
</style>
<div id="main">
  <div id="stage">
    <img id="video" src="/stream.mjpg" alt="simulation stream">
    <div id="hud">仿真时间 <b id="tSim">0.00</b>s · 流 <b id="tFps">0</b>fps · 接触 <b id="tCon">0</b></div>
    <div id="pausedBadge">已暂停</div>
    <div id="pickBadge" style="position:absolute;display:none;background:rgba(10,12,16,.9);border:1px solid var(--acc);border-radius:6px;padding:4px 9px;font-size:12px;pointer-events:none"></div>
  </div>
  <div id="toolbar">
    <button id="btnPause" class="primary">⏸ 暂停</button>
    <button id="btnStep">单步 100ms</button>
    <button id="btnReset">⟲ 复位</button>
    <span style="color:var(--dim)">速度</span>
    <input id="speed" type="range" min="0.05" max="4" step="0.05" value="1" style="width:110px">
    <b id="speedVal" style="width:34px">1.0×</b>
    <button id="btnReload" onclick="reloadModel()" title="修改 XML 后点击, 无需重启服务">🔄 重载模型</button>
    <span style="margin-left:auto;color:var(--dim);font-size:11.5px">
      单击=识别部件 · 拖拽=环绕 · 右键/Shift拖拽=平移 · 滚轮=缩放 · <kbd>Space</kbd>暂停 · <kbd>1-6</kbd>选关节 · <kbd>←→</kbd>微调 · <kbd>F</kbd>跟手 · <kbd>R</kbd>复位
    </span>
  </div>
</div>
<div id="side">
  <details open><summary>🖐 手势预设（Linker Hand O6）</summary><div class="body">
    <div class="row">
      <button onclick="gesture('open')">张开</button>
      <button onclick="gesture('fist')">握拳</button>
      <button onclick="gesture('point')">食指</button>
      <button onclick="gesture('peace')">比耶</button>
      <button onclick="gesture('ok')">OK</button>
      <button onclick="gesture('thumb')">点赞</button>
    </div>
  </div></details>
  <details open><summary>🦾 臂姿态预设（Piper）</summary><div class="body">
    <div class="row">
      <button onclick="armPreset('home')">初始</button>
      <button onclick="armPreset('forward')">前伸</button>
      <button onclick="armPreset('side')">侧伸</button>
      <button onclick="armPreset('low')">低位</button>
    </div>
  </div></details>
  <details open><summary>🎚 关节控制（12 执行器）</summary><div class="body" id="sliders"></div></details>
  <details><summary>🎥 相机</summary><div class="body">
    <div class="row">
      <button onclick="camPreset('default')">默认</button>
      <button onclick="camPreset('front')">正面</button>
      <button onclick="camPreset('side')">侧面</button>
      <button onclick="camPreset('top')">顶部</button>
      <button onclick="camPreset('closeup')">手部特写</button>
      <button id="btnFollow" onclick="toggleFollow()">跟随手部</button>
    </div>
  </div></details>
  <details><summary>👁 可视化选项</summary><div class="body">
    <span class="chip" data-vis="contact_points">接触点</span>
    <span class="chip" data-vis="contact_forces">接触力</span>
    <span class="chip" data-vis="joint_axes">关节轴</span>
    <span class="chip" data-vis="inertia">惯量</span>
    <span class="chip" data-vis="transparent">半透明</span>
    <span class="chip" data-vis="collisions">碰撞几何</span>
  </div></details>
  <details open><summary>🔍 点击识别（点画面中的部件）</summary><div class="body" id="pickPanel" style="color:var(--dim)">在画面中单击任意部件，显示部件名、关节和对应 XML 位置</div></details>
  <details open><summary>📊 状态与曲线</summary><div class="body">
    <div class="stats">
      <span>仿真时间 <b id="sTime">0.00 s</b></span><span>流帧率 <b id="sFps">0 fps</b></span>
      <span>接触数 <b id="sCon">0</b></span><span>接触合力 <b id="sForce">0.0 N</b></span>
      <span>末端 X <b id="sX">0</b></span><span>末端 Y <b id="sY">0</b></span>
      <span>末端 Z <b id="sZ">0</b></span><span>物理速率 <b id="sSpeed">1.0×</b></span>
    </div>
    <div style="color:var(--dim);font-size:11.5px;margin:4px 0">接触合力曲线（近 30s）</div>
    <canvas id="chartForce" width="280" height="70"></canvas>
  </div></details>
  <details><summary>ℹ️ 关于</summary><div class="body help">
    模型：Piper 标准版 + 左 Linker Hand O6-CAN<br>
    自由度：17（臂 6 + 手 11，其中 5 个 DIP/IP 联动）<br>
    执行器：12（臂 6 + 手 6）<br>
    MuJoCo <b id="sMj">-</b> · MJPEG 流 · 服务端 EGL 渲染<br>
    手指联动比例：DIP=0.89×MCP · 拇指 IP=2.29×CMC
  </div></details>
</div>
<script>
const $=id=>document.getElementById(id);
const video=$('video'), stage=$('stage');
let ACT=[];           // actuator meta from /model.json
let ctrlVals=[];      // last known ctrl
let dragging=null, draggingButton=0;
let selectedJoint=-1;
const forceHist=[];

// ---------- video pointer control ----------
video.addEventListener('contextmenu',e=>e.preventDefault());
video.addEventListener('pointerdown',e=>{video.setPointerCapture(e.pointerId);dragging={x:e.clientX,y:e.clientY};draggingButton=e.button;});
video.addEventListener('pointermove',e=>{
  if(!dragging)return;
  const dx=e.clientX-dragging.x, dy=e.clientY-dragging.y;
  dragging={x:e.clientX,y:e.clientY};
  const op=(draggingButton===2||e.shiftKey)?'pan':'orbit';
  fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op,dx,dy})});
});
let downPos=null,downTime=0;
video.addEventListener('pointerdown',e=>{downPos={x:e.clientX,y:e.clientY};downTime=Date.now();});
video.addEventListener('pointerup',async e=>{
  dragging=null;
  if(!downPos) return;
  const moved=Math.hypot(e.clientX-downPos.x,e.clientY-downPos.y);
  if(moved<5 && Date.now()-downTime<400){
    const rect=video.getBoundingClientRect();
    const x=Math.round((e.clientX-rect.left)/rect.width*640);
    const y=Math.round((e.clientY-rect.top)/rect.height*480);
    doPick(x,y,e.clientX-rect.left,e.clientY-rect.top);
  }
  downPos=null;
});
async function doPick(x,y,ux,uy){
  try{
    const r=await fetch('/pick',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y})});
    const j=await r.json();
    const badge=$('pickBadge'), panel=$('pickPanel');
    if(!j.body){badge.style.display='none';panel.textContent='点击处是背景，请点在机器人上';return;}
    badge.style.display='block';
    badge.style.left=Math.min(ux,video.clientWidth-140)+'px';
    badge.style.top=Math.max(uy-34,4)+'px';
    badge.innerHTML=`<b style="color:#4f8cff">${j.body}</b>${j.joints.length?` · ${j.joints.map(q=>q.name).join(', ')}`:''}`;
    let html=`<div style="color:#dde3ec;font-size:14px"><b>${j.body}</b></div>`;
    html+=`<div class="stats"><span>父级 <b>${j.parent}</b></span><span>当前角度 <b>${j.qpos!==null?j.qpos.toFixed(3)+' rad':'—'}</b></span></div>`;
    if(j.joints.length){
      html+=j.joints.map(q=>`<div style="margin:4px 0">关节 <b style="color:#3fb96f">${q.name}</b> 限位 [${q.range[0].toFixed(2)}, ${q.range[1].toFixed(2)}] rad<br><span style="color:var(--acc);cursor:pointer" onclick="openXml('${q.file}',${q.line})">📄 ${q.file}:${q.line}</span></div>`).join('');
    }
    if(j.xml.file) html+=`<div style="margin-top:6px;color:var(--acc);cursor:pointer" onclick="openXml('${j.xml.file}',${j.xml.line})">📄 定位 body 定义: ${j.xml.file}:${j.xml.line}</div><pre style="background:#10131a;border-radius:6px;padding:6px;margin-top:4px;white-space:pre;overflow-x:auto;font-size:11px">${j.xml.excerpt.replace(/</g,'&lt;')}</pre>`;
    panel.innerHTML=html;
  }catch(err){console.error(err);}
}
function openXml(file,line){
  if(window.vscode&&window.vscode.window){/* 若在 VS Code webview 中可扩展 */}
  const url=`/xml?file=${encodeURIComponent(file)}&line=${line}`;
  window.open(url,'_blank');
}
async function reloadModel(){
  const r=await fetch('/reload',{method:'POST'});
  const j=await r.json();
  if(j.ok){alert(`重载成功: nq=${j.nq} nu=${j.nu}`);init();}
  else alert('重载失败:\n'+j.error);
}
video.addEventListener('pointercancel',()=>dragging=null);
video.addEventListener('wheel',e=>{e.preventDefault();fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op:'zoom',dz:e.deltaY})});},{passive:false});

// ---------- toolbar ----------
$('btnPause').onclick=()=>sim('pause');
$('btnStep').onclick=()=>sim('step');
$('btnReset').onclick=()=>{sim('reset');};
$('speed').oninput=e=>{sim('speed',e.target.value);};
async function sim(action,value){const r=await fetch('/sim',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,value})});const j=await r.json();$('speedVal').textContent=j.speed.toFixed(1)+'×';$('pausedBadge').style.display=j.paused?'block':'none';$('btnPause').textContent=j.paused?'▶ 继续':'⏸ 暂停';}

// ---------- keyboard ----------
document.addEventListener('keydown',e=>{
  if(e.repeat&&['ArrowLeft','ArrowRight'].includes(e.key)){}
  if(e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();sim('pause');}
  else if(e.key==='r'||e.key==='R'){sim('reset');}
  else if(e.key==='f'||e.key==='F'){toggleFollow();}
  else if(e.key>='1'&&e.key<='6'){selectedJoint=+e.key-1;renderSliders();}
  else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    if(selectedJoint<0)return;
    e.preventDefault();
    const d=(e.key==='ArrowRight'?1:-1)*(e.shiftKey?0.01:0.05);
    setCtrl(selectedJoint,clamp(ctrlVals[selectedJoint]+d,ACT[selectedJoint].range[0],ACT[selectedJoint].range[1]));
  }
});

// ---------- sliders ----------
async function init(){
  const r=await fetch('/model.json');const j=await r.json();
  $('sMj').textContent=j.version;
  ACT=j.actuators;
  const st=await(await fetch('/state')).json();ctrlVals=st.ctrl;
  renderSliders();renderVisChips(st.vis);
  setInterval(poll,200);
  $('speedVal').textContent=st.speed.toFixed(1)+'×';
}
function renderSliders(){
  const host=$('sliders');host.innerHTML='';
  ACT.forEach((a,i)=>{
    const div=document.createElement('div');div.className='ctl';
    const isHand=a.name.startsWith('lh_');
    div.innerHTML=`<div class="lab"><b>${i+1}. ${a.joint}</b><span id="ro${i}">–</span></div>
      <input type="range" id="sl${i}" min="${a.range[0]}" max="${a.range[1]}" step="0.01" value="${ctrlVals[i]??0}"
        style="accent-color:${isHand?'#3fb96f':'#4f8cff'}">
      ${i===selectedJoint?'<div style="font-size:11px;color:var(--acc)">← 已选中（方向键微调）</div>':''}`;
    host.appendChild(div);
    $('sl'+i).addEventListener('input',e=>setCtrl(i,+e.target.value));
    $('sl'+i).addEventListener('pointerdown',()=>{selectedJoint=i;});
  });
}
function setCtrl(i,v){ctrlVals[i]=v;fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({i,v})});$('sl'+i).value=v;}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

// ---------- presets ----------
const GESTURES={open:[0,0,0,0,0,0],fist:[1.3,0.58,1.6,1.6,1.6,1.6],point:[0.6,0.4,0,1.6,1.6,1.6],peace:[0.5,0.2,0,0,1.6,1.6],ok:[0.3,0.55,0.9,0,0,0],thumb:[0.9,0.05,1.6,1.6,1.6,1.6]};
const ARM={home:[0,1.57,-1.3485,0,0,0],forward:[0,0.55,-1.15,0,0,0],side:[1.45,0.75,-1.3,0,0,0],low:[0,2.2,-2.2,0,0,0]};
function gesture(name){const h=GESTURES[name];for(let i=0;i<6;i++)setCtrl(6+i,h[i]);}
function armPreset(name){const a=ARM[name];for(let i=0;i<6;i++)setCtrl(i,a[i]);}

// ---------- camera ----------
function camPreset(name){fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op:'preset',name})});}
async function toggleFollow(){const r=await fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({op:'follow'})});$('btnFollow').classList.toggle('on',(await r.json()).ok);}

// ---------- vis chips ----------
function renderVisChips(v){document.querySelectorAll('.chip').forEach(c=>{c.classList.toggle('on',!!v[c.dataset.vis]);});}
document.querySelectorAll('.chip').forEach(c=>c.onclick=async()=>{
  const r=await fetch('/vis',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:c.dataset.vis})});
  renderVisChips((await r.json()).vis);
});

// ---------- polling / charts ----------
async function poll(){
  try{
    const st=await(await fetch('/state')).json();
    ctrlVals=st.ctrl;
    $('tSim').textContent=st.time.toFixed(2);$('tFps').textContent=st.fps.toFixed(0);$('tCon').textContent=st.ncon;
    $('sTime').textContent=st.time.toFixed(2)+' s';$('sFps').textContent=st.fps.toFixed(0)+' fps';
    $('sCon').textContent=st.ncon;$('sForce').textContent=st.force.toFixed(1)+' N';
    $('sX').textContent=st.eef[0].toFixed(3);$('sY').textContent=st.eef[1].toFixed(3);$('sZ').textContent=st.eef[2].toFixed(3);
    $('sSpeed').textContent=st.speed.toFixed(1)+'×';
    $('pausedBadge').style.display=st.paused?'block':'none';
    $('btnPause').textContent=st.paused?'▶ 继续':'⏸ 暂停';
    $('btnFollow').classList.toggle('on',st.follow);
    ACT.forEach((a,i)=>{const ro=$('ro'+i);if(ro)ro.textContent=st.qpos[a.qadr].toFixed(2);});
    // sync sliders that aren't being dragged
    ACT.forEach((a,i)=>{const sl=$('sl'+i);if(sl&&document.activeElement!==sl&&Math.abs(+sl.value-st.ctrl[i])>0.001&&!dragging)sl.value=st.ctrl[i];});
    forceHist.push(st.force);if(forceHist.length>150)forceHist.shift();
    drawChart();
  }catch(e){}
}
function drawChart(){
  const c=$('chartForce'),ctx=c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(forceHist.length<2)return;
  const max=Math.max(0.1,...forceHist);
  ctx.beginPath();
  forceHist.forEach((v,i)=>{const x=i/(forceHist.length-1)*c.width,y=c.height-4-(v/max)*(c.height-8);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.strokeStyle='#4f8cff';ctx.lineWidth=1.5;ctx.stroke();
  ctx.fillStyle='#8b93a3';ctx.font='10px sans-serif';ctx.fillText(max.toFixed(1)+' N',4,10);
}
init();
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host_ip = sock.getsockname()[0]
    except OSError:
        host_ip = socket.gethostbyname(socket.gethostname())
    print(f"\n仿真服务已启动，复制下面地址到其他设备浏览器：")
    print(f"http://{host_ip}:{port}")
    print(f"本机地址：http://127.0.0.1:{port}\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
