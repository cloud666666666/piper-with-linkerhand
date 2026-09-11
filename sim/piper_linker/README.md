# Piper + 左 Linker Hand O6-CAN MuJoCo 仿真

模型组成：Piper 标准版（6 自由度臂）+ 灵巧手连接件（Ø39mm 同轴固定板，厚 17.5mm，由 `灵巧手连接件.STEP` 转换）+ 左 Linker Hand O6（11 关节 / 6 独立驱动 / 5 联动）。手基座安装在 link6 法兰前方 17.5mm 处（同轴）。数据溯源与修改记录见 [PROVENANCE.md](PROVENANCE.md)。

## 本机窗口

```bash
cd /home/czn/mujoco/piper_linker_sim
source /home/czn/mujoco/.venv/bin/activate
python main.py
```

## 通过局域网浏览器访问（Web 版 viewer）

```bash
cd /home/czn/mujoco/piper_linker_sim
MUJOCO_GL=egl uv run server.py
```

服务监听 `0.0.0.0:8000`，启动时打印可复制的访问地址。页面功能：

- **实时画面**：MJPEG 流（约 24-30fps），鼠标拖拽环绕、右键/Shift 拖拽平移、滚轮缩放
- **仿真控制**：暂停/继续（Space）、单步 100ms、复位（R）、0.05×-4× 变速
- **12 个执行器滑杆**：臂 6 关节 + 手 6 关节，实时显示实际关节角度；`1-6` 选关节、`←→` 微调
- **手势预设**：张开 / 握拳 / 食指 / 比耶 / OK / 点赞
- **臂姿态预设**：初始 / 前伸 / 侧伸 / 低位
- **相机预设**：默认 / 正面 / 侧面 / 顶部 / 手部特写 / 跟随手部（F）
- **可视化开关**：接触点、接触力、关节轴、惯量、半透明、碰撞几何
- **状态面板**：仿真时间、接触数、接触合力、末端位置、流帧率、接触力曲线

其他设备浏览器打开启动时打印的地址，例如：

```text
http://服务器IP:8000
```

例如：

```text
http://192.168.1.100:8000
```

查看状态：

```bash
curl http://127.0.0.1:8000/state
```

设置全部 12 个执行器目标值：

```bash
curl -X POST http://127.0.0.1:8000/control \
  -H 'Content-Type: application/json' \
  -d '{"ctrl":[0,1.57,-1.3485,0,0,0,0,0,0,0,0,0]}'
```

其中前 6 个是 Piper 执行器，后 6 个是 Linker Hand 的独立手指执行器（拇指旋转、拇指弯曲、食指/中指/无名指/小指弯曲，0=张开，1.6=最大弯曲；DIP/IP 关节按 0.89/2.29 比例联动）。

> `0.0.0.0` 只用于服务端监听；客户端必须使用服务器实际 IP。公网部署前请增加认证、HTTPS 和控制权限校验。
