import mujoco
import numpy as np

# 加载一个简单的内置模型
model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><geom type='sphere' size='0.1'/></body></worldbody></mujoco>")
data = mujoco.MjData(model)

print("MuJoCo 版本:", mujoco.__version__)
print("模型加载成功！")