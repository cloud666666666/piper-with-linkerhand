"""WebSocket bridge client for the Piper + LinkerHand MuJoCo sim.

Connects to the Robonix ROS bridge (piper_linker_deploy/sim/bridge_node.py,
WebSocket server on :8765) and exchanges the shared sim protocol:

  上行  hello + 30Hz state 帧 (17 关节 qpos/qvel + 手基座位姿)
  下行  arm_joint_command (按关节名映射到 12 个驱动执行器),
        arm_pose_command (记录日志, 本体无内置 IK 时由上层原语分解),
        reset (回调仿真自身的确定性复位)

设计要点 (对应接入指南):
  - 与 server.py 共享同一份 model/data/lock —— viewer、HTTP API、bridge
    看到的是同一个仿真;
  - 关节名 → ctrl 下标的映射在启动时从 actuator_trnid 动态构建,
    不依赖 XML 中的声明顺序;
  - 断连时位置执行器自动保持最后目标 (watchdog 语义), 重连后继续。

由 server.py 在 BRIDGE_WS_URL 环境变量设置时拉起, 默认不启动。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading

import mujoco

log = logging.getLogger("bridge_client")

STATE_HZ = 30.0


class BridgeClient:
    def __init__(self, model, data, lock, reset_fn, url: str) -> None:
        self.model, self.data, self.lock, self.reset_fn = model, data, lock, reset_fn
        self.url = url
        self.thread: threading.Thread | None = None

        # qpos 顺序的全部关节名 (6 臂 + 11 手, 含联动)
        self.joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
        ]
        # 驱动关节名 → ctrl 下标 (12 个 position 执行器)
        self.ctrl_of: dict[str, int] = {}
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a][0])
            self.ctrl_of[self.joint_names[jid] if jid < len(self.joint_names)
                         else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)] = a
        self.eef_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_hand_base_link")

    # ── 状态帧 ────────────────────────────────────────────────────────────────
    def build_state(self) -> dict:
        with self.lock:
            return {
                "type": "state",
                "simulationTime": float(self.data.time),
                "robot": {
                    "base": {  # 桌面固定基座, 协议占位
                        "position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "linearVelocity": [0.0, 0.0, 0.0],
                        "angularVelocity": [0.0, 0.0, 0.0],
                    },
                    "arm": {
                        "names": list(self.joint_names),
                        "positions": [float(q) for q in self.data.qpos],
                        "velocities": [float(v) for v in self.data.qvel],
                        "endPose": {
                            "position": [float(x) for x in self.data.xpos[self.eef_bid]],
                            "quaternion": [float(q) for q in self.data.xquat[self.eef_bid]],
                        },
                    },
                    "objects": [],
                },
            }

    # ── 命令处理 ──────────────────────────────────────────────────────────────
    def apply_command(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "arm_joint_command":
            names, positions = payload.get("names") or [], payload.get("positions") or []
            applied, skipped = [], []
            with self.lock:
                for name, value in zip(names, positions):
                    idx = self.ctrl_of.get(name)
                    if idx is None:
                        skipped.append(name)      # 联动关节等: 由等式约束跟随
                    else:
                        self.data.ctrl[idx] = float(value)
                        applied.append(name)
            if skipped:
                log.debug("arm_joint_command skipped (not driven): %s", skipped)
            log.info("arm_joint_command applied %d joints", len(applied))
        elif kind == "arm_pose_command":
            log.info("arm_pose_command %s (no onboard IK; decompose upstream)",
                     payload.get("position"))
        elif kind == "reset":
            self.reset_fn()
            log.info("reset applied")
        elif kind == "emergency_stop":
            with self.lock:
                self.data.ctrl[:] = self.data.qpos[: self.model.nu] if self.model.nu else self.data.ctrl
            log.warning("emergency_stop: actuators set to hold current pose")
        else:
            log.debug("ignoring bridge message type=%r", kind)

    # ── 连接循环 ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        import websockets

        while True:
            try:
                async with websockets.connect(self.url, ping_interval=10, ping_timeout=10) as ws:
                    await ws.send(json.dumps({
                        "type": "hello", "backend": "native",
                        "environment": "piper_linker_desktop",
                        "robot": "piper_linkerhand_o6_left",
                        "visualMode": "mesh",
                    }))
                    log.info("connected to bridge %s", self.url)

                    async def sender():
                        interval = 1.0 / STATE_HZ
                        while True:
                            await ws.send(json.dumps(self.build_state()))
                            await asyncio.sleep(interval)

                    send_task = asyncio.create_task(sender())
                    try:
                        async for raw in ws:
                            try:
                                self.apply_command(json.loads(raw))
                            except (json.JSONDecodeError, TypeError, ValueError) as e:
                                log.warning("bad bridge message: %s", e)
                    finally:
                        send_task.cancel()
            except Exception as e:  # noqa: BLE001
                log.info("bridge not reachable (%s); retrying in 2s", e)
                await asyncio.sleep(2.0)

    def _thread_main(self) -> None:
        asyncio.run(self.run())

    def start(self) -> None:
        self.thread = threading.Thread(target=self._thread_main, daemon=True,
                                       name="robonix-bridge-client")
        self.thread.start()
        log.info("bridge client thread started (%s @ %.0fHz)", self.url, STATE_HZ)


def start(model, data, lock, reset_fn, url: str) -> BridgeClient:
    client = BridgeClient(model, data, lock, reset_fn, url)
    client.start()
    return client
