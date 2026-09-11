"""Piper + left Linker Hand O6-CAN MuJoCo demo.

Run from this directory with:
    python main.py

The arm is loaded from the bundled Piper MJCF.  The hand is generated from the
official left O6 URDF and attached to Piper link6 through a fixed site/body
mount.  Press Space to pause, O to open the hand, and C to close it.
"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key = model.key("home")
    data.qpos[: len(key.qpos)] = key.qpos
    # The hand uses six independent position actuators after Piper's seven.
    data.ctrl[: len(key.ctrl)] = key.ctrl
    data.ctrl[len(key.ctrl):] = 0.0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.25
        viewer.cam.lookat[:] = [0.22, 0.0, 0.3]
        while viewer.is_running():
            # Keep the hand's six independent targets grouped at the end of
            # the actuator vector; the remaining seven are Piper actuators.
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
