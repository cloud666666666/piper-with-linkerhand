# Piper + LinkerHand O6 抓取与仿真

Piper 机械臂 + LinkerHand O6 灵巧手的抓取工程：**平掌包络抓取**（核心）、2D 手眼标定、
YOLO 目标检测，以及配套的 **MuJoCo 仿真**与静态 3D 模型查看器。

- 抓取：`classification/catch_with_linker_hand_flat.py`（灵巧手平掌包络，默认单次抓取）
- 标定：`arm/calibrate_arm_hand.py`（平抓 2D，掌心对准）、`arm/calibrate_handeye_2d.py`（垂直 2D）、`arm/calibrate_handeye_s10.py`（眼在手上 3D）
- 仿真：`sim/piper_linker/`（MuJoCo 场景 `scene.xml`，臂 + 灵巧手 + 连接件）
- 查看器：`docs/viewer/`（three.js + URDF，纯静态，可挂 GitHub Pages）

> 本仓库已收窄为「Piper 臂 + LinkerHand O6 灵巧手」单主题；LLM/象棋/主从臂/VLA(lerobot) 相关目录已移除。

---

## 硬件依赖

| 设备 | 说明 |
|------|------|
| Piper 机械臂 | USB-CAN 适配器连接（`gs_usb`），CAN 波特率 1M |
| LinkerHand O6 灵巧手（右手） | **Modbus RTU / RS485**（USB-485 转接器，115200 8N1）；SDK 以 submodule 提供 |
| 相机 | Orbbec 深度相机或 USB(UVC) 相机，固定俯视安装（手眼标定要求相机不动） |
| USB-CAN 适配器 / USB-485 转接器 | 分别连接机械臂与灵巧手 |
| Linux 主机 / Jetson Orin | 运行控制程序与仿真 |

---

## 环境搭建

### 1. 系统依赖

```bash
sudo apt update
sudo apt install -y can-utils ethtool build-essential
```

**Jetson 上额外需要 gs_usb 内核模块**（USB-CAN 通信必需）：

先检查是否已加载：

```bash
lsmod | grep gs_usb
```

如果没有输出，说明内核未启用 `gs_usb`，需要编译安装。详见文末 [故障排查 / gs_usb 内核模块](#gs_usb-内核模块未加载)。

### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

验证：

```bash
uv --version   # 应 >= 0.5.0
```

### 3. 克隆仓库（含 SDK submodule）并同步环境

```bash
# 一次性：克隆仓库 + 初始化 submodule（灵巧手 SDK）
git clone --recurse-submodules <repo-url>
cd roboarm
# 已经克隆过：
git submodule update --init --recursive

# 同步 Python 环境（依赖见 pyproject.toml；uv.lock 已按收窄后的依赖集重新生成）
uv sync

# 验证关键依赖
uv run python -c "from piper_sdk import C_PiperInterface_V2; print('piper_sdk OK')"
uv run python -c "from ultralytics import YOLO; print('ultralytics OK')"
uv run python -c "import mujoco; print('mujoco OK', mujoco.__version__)"
uv run python -c "import pymodbus; print('pymodbus OK', pymodbus.__version__)"
```

> 说明：
> - `torch` 由 `ultralytics` 传递引入（PyPI 版本）。Jetson 若要用 NVIDIA GPU 版 torch，
>   请自行安装（本仓库不再附带 `wheels/`、`stubs/` 与 `lerobot/` 本地源）。
> - 灵巧手 SDK 是 **git submodule**（`third_party/linkerhand-python-sdk`），不是 pip 包，
>   配置方式见下面第 6 节。

### 4. CAN 总线配置（Piper 机械臂）

#### 4.1 找到真实的 CAN 接口名

```bash
source .venv/bin/activate

SDK_DIR=$(python - <<'PY'
import inspect, os, piper_sdk
print(os.path.dirname(inspect.getfile(piper_sdk)))
PY
)

bash "$SDK_DIR/find_all_can_port.sh"
```

示例输出：

```
Interface can_piper is connected to USB port 1-4.2:1.0
```

> 记下输出中的 **接口名**（如 `can_piper`、`can4`）和 **USB 地址**（如 `1-4.2:1.0`）。

#### 4.2 激活 CAN 接口

```bash
CAN_IF="can_piper"    .
    # 替换为上一步的实际接口名
USB_ADDR="1-4.2:1.0"      # 替换为上一步的实际 USB 地址

sudo ip link set "$CAN_IF" down 2>/dev/null || true
bash "$SDK_DIR/can_activate.sh" "$CAN_IF" 1000000 "$USB_ADDR"
```

> **注意**：波特率固定为 `1000000`，不要填成 `100000`。

#### 4.3 验证 CAN 状态

```bash
ip -details link show "$CAN_IF"
```

应看到接口处于 `UP` 状态，波特率为 `1000000`。

### 5. 连通性验证

复制配置文件并修改：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，将 `arm_type` 改为 `piper`，`arm_port` 改为实际的 CAN 接口名（如 `can_piper` 或 `can4`）。

```bash
# 只读连通性测试
uv run python arm/calibrate_offset.py
```

如果能打印当前关节角度/末端位姿，说明 CAN 通信正常。

```bash
# 运动测试
uv run python arm/piper_ctrl_by_sdk.py
```

机械臂应能进行使能并执行测试运动。

---

### 6. 灵巧手（LinkerHand O6）环境准备

灵巧手 SDK 以 **git submodule** 形式随仓库（`third_party/linkerhand-python-sdk`），
上游：<https://github.com/linker-bot/linkerhand-python-sdk>，锁定的提交为
`0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8`（2026-08-11，SDK 自带版本号 3.1.1，支持 O6/L6 的 RS485 模式）。

```bash
# 克隆时一并拉取（本仓库只有这一个 submodule）
git clone --recurse-submodules <repo-url>
# 已经克隆过：
git submodule update --init --recursive

# 本机无法直连 GitHub 时，让 git 走代理（一次性配置）
git config --global url."https://v4.gh-proxy.org/https://github.com/".insteadOf "https://github.com/"
```

**让 Python 找到 SDK（二选一）**：

1. **推荐**：`config.yaml` 里设置 `linker_hand_sdk_path: <仓库>/third_party/linkerhand-python-sdk`，
   `arm/linker_hand.py` 会把该目录插入 `sys.path` 后再 `from LinkerHand.linker_hand_api import LinkerHandApi`；
2. 或设环境变量：`export PYTHONPATH=$PWD/third_party/linkerhand-python-sdk:$PYTHONPATH`。

> ⚠️ SDK **没有** `setup.py` / `pyproject.toml`，所以 **不能** `pip install -e third_party/linkerhand-python-sdk`
> （会报“找不到构建配置”）——只能用上面的 `linker_hand_sdk_path` 或 `PYTHONPATH` 方式。
>
> 🔧 **重要：上游 SDK 与本仓库有 `utils` 包名冲突，必须打补丁**。上游 SDK 内部用裸包名导入
> （`from utils.mapping import *`、`from core.rs485...`），而本仓库根目录**也有自己的 `utils/` 包**；
> 从仓库根运行脚本时（如 `python3 linker_hand_fist.py`）`utils` 会先解析到本仓库的 `utils`，
> 实测报错：`ModuleNotFoundError: No module named 'utils.mapping'`（把 SDK 根目录放在 `sys.path` 最前也无效，
> 因为冲突的是顶级名 `utils` 本身）。解决办法是把 SDK 内部的 `utils.*` / `core.*` 改成
> `LinkerHand.utils.*` / `LinkerHand.core.*`（等价重构、不改行为）：仓库已附带补丁文件，
> **一行应用**（只改 submodule 的工作区，不产生新提交）：
>
> ```bash
> git -C third_party/linkerhand-python-sdk apply ../linkerhand-imports.patch
> ```
>
> 该补丁针对当前 submodule 提交 `0cc0585` 生成；上游更新后可能需重做。
> 更省事、可复现的做法：把补丁提交到你自己的 **fork**，再把 `.gitmodules` 的 URL 与 submodule
> 指纹指向该 fork（推荐长期方案）。
> （开发机上的 `/home/czn/linkerhand-python-sdk` 已打好同一补丁，`config.yaml` 的
> `linker_hand_sdk_path` 指向它即可直接工作。）

**依赖**：灵巧手 Modbus RTU 链路需要 **`pymodbus==3.5.1`**（SDK 的 `requirements.txt` 即固定此版本；
它会带上 `pyserial`）与 **PyYAML**（读 SDK 的 `LinkerHand/config/setting.yaml`）。已在
`pyproject.toml` 的 `dependencies` 中声明；CAN 模式额外需要 `python-can`（本项目已声明），Modbus 模式不需要。

> `uv.lock` 已按收窄后的依赖集**重新生成**（`uv lock`，96 个包；已去掉 lerobot/torch 本地源）。
> `uv sync` 会按 `pyproject.toml` + `uv.lock` 安装，包含 `pymodbus==3.5.1` 与 `mujoco`。

**Modbus（RS485）串口准备**：

```bash
ls /dev/serial/by-id/        # 推荐：by-id 名稳定，填这个
ls /dev/ttyUSB*              # 简易：多个转接器时序号可能漂移
dmesg | tail -20             # 插入 USB-485 转接器后看内核识别到哪个口
```

- `config.yaml` 的 `linker_hand_modbus` 填真实设备名（SDK 的 `config/setting.yaml` 里
  `RIGHT_HAND.MODBUS` 目前也是 `/dev/ttyUSB0`，可互相对照）；
- **波特率 115200 / 8N1 由 SDK 固定**（`linker_hand_api.py` 构造时硬编码，无配置项）；
- 从站地址由 `linker_hand_type` 决定：**右手 0x27(39)、左手 0x28(40)**；
- 权限：`sudo chmod 777 /dev/ttyUSB0` 或把用户加入 `dialout` 组（SDK 文档写 777）。

**验证（仅显式运行脚本时才发帧；请先应用上面的导入补丁，否则会报 `No module named 'utils.mapping'`）**：

```bash
# 先确认补丁已应用（无输出即已应用）
git -C third_party/linkerhand-python-sdk diff --quiet && echo "尚未打补丁" || echo "补丁已应用"

uv run python linker_hand_open.py --hold 2     # 张开，保持 2 秒
uv run python linker_hand_fist.py --hold 2     # 握拳，保持 2 秒
# 两者都会打印：下发姿态值 + 来源（config 覆盖还是内置默认）+ 通信链路（Modbus/CAN），
# 并在保持后回读关节位置，便于确认是否到位。
```

> 提醒：O6 **右手握拳值**官方 SDK 未定义（`O6_positions.yaml` 的 RIGHT_HAND 只有「张开」），
> 内置默认取官方通用示例 `[102,18,0,0,0,0]`，**待实机验证**；不满意时用 config
> `linker_hand_fist_pose` 覆盖（优先微调第 1、2 个分量），无需改代码。

---

## config.yaml 说明

复制模板后按机器修改：

```bash
cp config.yaml.example config.yaml
```

关键键（完整清单与中文注释见 `config.yaml.example`）：

| 类别 | 键 |
|---|---|
| 机械臂 | `arm_type: piper`、`arm_port`（如 `can4`）、`arm_reach_mse_threshold_deg2` |
| 相机 | `camera_ip`（留空用本地相机）、`camera_port`、`cv2_headless_port` |
| 桌面/时序 | `default_desktop_height`、`catch_time_interval_s`、`get_arm_angles_retry_times` |
| YOLO | `classification_YOLO_model_path`（列表）、`default_conf_thres` |
| 抓取动作 | `place_pos`（按类别的放置点）、`place_distance_threshold`、`catch_offset`、`catch_raise_height`、`place_raise_height`、`go_down_before_open_gripper_in_place` |
| 灵巧手 | `linker_hand_type: right`、`linker_hand_modbus`、`linker_hand_sdk_path`、`linker_hand_speed/torque`、张开/握拳手势键 |
| 平掌抓取 | `linker_hand_pin_joints`、`linker_hand_pin_on_start`、`linker_hand_flat_euler_deg_zyx`、`linker_hand_flat_grasp_height_m`、`linker_hand_flat_heading_offset_deg`、`linker_hand_flat_approach_mode`、`linker_hand_flat_catch_once`、`linker_hand_flat_handeye_file` |

> 「平掌抓取」相关键的取值都带实机标定/验证记录，改动前请先读 `config.yaml.example` 里的中文注释
> （尤其 J5 钉值、平抓高度、接近方式三项）。

### YOLO 模型准备

`classification_YOLO_model_path` 指向 YOLO 权重（本仓库不附带 `object_detect/runs/`）。
自备模型放到该路径即可，例如：

```bash
ls object_detect/runs/     # 放 best.pt（或任意自训练权重）
# 训练不在本仓库范围内，可用 ultralytics 官方流程在自己的数据上训练
```

---

## 手眼标定（抓取前必须）

固定相机（eye-to-hand）俯视场景，标定**像素 ↔ 机械臂平面**的映射；相机在标定与抓取期间都不能移动。

| 脚本 | 用途 | 产物 |
|---|---|---|
| `arm/calibrate_arm_hand.py` | **平掌包络抓取专用**：把「掌心（抓取接触点）」对准画面目标点，联合拟合单应 H（像素→掌心）+ 臂长 L（掌心→法兰），支持 `--mode test` 复验、`--mode fit --drop-points` 剔除坏点 | `arm/hand-eye-data/2d_homography_flat.npz` |
| `arm/calibrate_handeye_2d.py` | 垂直抓取（夹爪/垂直手）2D 标定：假设法兰 xy = 末端 xy 的单应 | `arm/hand-eye-data/2d_homography.npy` |
| `arm/calibrate_handeye_s10.py` | 眼在手上（臂上相机）3D 标定：自动采集多姿态图像+末端位姿，`cv2.calibrateHandEye` 求解 | `arm/hand-eye-data/3d_handeye.npz` |

平掌标定流程要点（详见脚本头部注释）：先回零 → 摆平掌姿态（J5 落在手拖止点 ≈-74.9
或电动钉值 -69.0 均可，J6≈6.75，掌面贴平）→ 掌心对准画面点按空格采集 ≥6 个点、方位角铺开 ≥±20°
→ ESC 结束自动拟合并打印 L/RMS；L 命中边界或 RMS>3mm 会醒目警告（提示重采或以 test 模式复验）。

---

## 运行抓取

```bash
# 1) 夹爪版（垂直下抓，Piper + 原夹爪）
uv run python classification/catch_with_arm.py --target potato

# 2) 灵巧手垂直下抓版（手从上方包络）
uv run python classification/catch_with_linker_hand.py --target potato

# 3) 灵巧手平掌包络版（推荐；默认单次抓取、直接下抓）
uv run python classification/catch_with_linker_hand_flat.py --target potato
```

- `--target <类别名>`：只抓该类别置信度最高的目标；省略则抓全画面最优目标。
- 平掌版的行为由 config 决定（启动会打印）：`linker_hand_flat_catch_once`（默认 true＝抓一次即收尾退出）、
  `linker_hand_flat_approach_mode`（默认 direct_down＝正上方对准后垂直下抓）、
  `linker_hand_flat_grasp_height_m`（平抓工作高度，绝对值 0.11m）。
- 灵巧手单发自检（只显式运行时发帧）：
  ```bash
  uv run python linker_hand_open.py --hold 2    # 张开
  uv run python linker_hand_fist.py --hold 2    # 握拳（含回读，便于核对右手握拳值）
  ```
- 机械臂侧自检：`uv run python test_joints.py`（逐关节）、`uv run python arm/piper_ctrl_by_sdk_flat_hand.py`（平抓姿态自测）。

---

## MuJoCo 仿真（sim/piper_linker）

Piper 臂 + 连接件 + LinkerHand O6 左手的 MuJoCo 场景（`scene.xml` 组装 `piper/piper.xml` 与手部网格/执行器/手指联动约束），
数据溯源与相对官方模型的改动见 `sim/piper_linker/PROVENANCE.md`。

```bash
# 本机打开 MuJoCo viewer
uv run python sim/piper_linker/main.py

# Web 端查看器（FastAPI + 浏览器）
uv run python sim/piper_linker/server.py

# 与仿真桥接的客户端示例
uv run python sim/piper_linker/bridge_client.py
```

离线自检（确认网格齐全、路径正确）：

```bash
# 注意：mujoco 对相对路径 + OBJ 的组合有已知行为，请用绝对路径或在 sim/piper_linker 目录内执行
uv run python -c "import mujoco; m=mujoco.MjModel.from_xml_path('$PWD/sim/piper_linker/scene.xml'); print(m.nq, m.nu)"
# 实测输出：17 12
```

---

## 3D 交互式模型查看器（three.js + URDF）

`docs/viewer/` 是一个**纯静态**的整机 3D 查看器（不连接机械臂/灵巧手），用于离线核对 URDF、
关节限位与抓取姿态：

- `docs/viewer/index.html`：查看器页面（three.js + URDFLoader，依赖已本地 vendor，无需联网）
- `docs/viewer/robot.urdf`：整机 URDF（Piper 7 段 + LinkerHand O6 左手 12 段；mesh 改为相对路径 `meshes/*.stl`）
- `docs/viewer/meshes/`：精简后的网格（原 22 MB → 约 5.7 MB；用 trimesh 二次误差简化面片，保留外形）

功能：轨道旋转/平移/缩放、6 个机械臂关节滑条（度）、6 个灵巧手驱动滑条（O6 值 0–255）、
预设按钮（手：张开 `[255,70,255,255,255,255]` / 握拳 `[102,18,0,0,0,0]`；臂：初始 /
平抓姿态 `J1~J4=[-3.87,90.29,-7.66,0.76]°、J5=-69.0°、J6=6.75°`）、相机预设、深色主题、中文界面。

### 本地查看

浏览器出于同源策略不能直接打开 `file://` 下的 URDF/mesh，需起静态服务（任选其一）：

```bash
# 在仓库根目录
python3 -m http.server 8000
# 然后浏览器访问
#   http://localhost:8000/docs/viewer/index.html     （查看器）
#   http://localhost:8000/docs/index.html            （文档落地页）
```

### GitHub Pages 开启方式与访问 URL

仓库已带 `.nojekyll`，Pages 直接托管 `docs/` 目录即可：

1. 打开仓库 **Settings → Pages**；
2. **Build and deployment → Source** 选 `Deploy from a branch`；
3. **Branch** 选 `main`（或默认分支）、目录选 **`/docs`**，保存；
4. 等 1–2 分钟，访问：

```
https://cloud666666666.github.io/piper-with-linkerhand/            # 文档落地页
https://cloud666666666.github.io/piper-with-linkerhand/viewer/index.html   # 3D 查看器
```

> 注：页面里的模型与 three.js 均为同目录相对路径，Pages 上无需任何额外配置或 CDN。

---

## 目录结构

```
roboarm/
  arm/                     # Piper 控制（SDK/平掌适配）、灵巧手封装、手眼标定
  camera/                  # Orbbec / USB 相机与远程相机客户端
  classification/          # 抓取脚本（夹爪版 / 灵巧手垂直版 / 灵巧手平掌包络版）
  object_detect/           # YOLO 推理（detect.py、model_server.py）与训练相关示例
  sim/piper_linker/        # MuJoCo 仿真（scene.xml + piper/ + linkerhand_o6_left/）
  docs/                    # 文档落地页 + viewer/（three.js 3D 查看器）
  third_party/             # 灵巧手 SDK（git submodule）
  urdf/piper/              # Piper URDF（IK 用）
  utils/                   # 配置读取、无头显示
  linker_hand_*.py         # 灵巧手张开/握拳单发脚本
  test_joints.py / test_linker_hand.py / test_mujoco.py
  config.yaml.example      # 配置模板（Piper + 灵巧手 + 平掌抓取）
  pyproject.toml / uv.lock
```

---

## 故障排查

### gs_usb 内核模块未加载

**现象**：`lsusb` 能看到 USB-CAN 设备，但 `ip link show | grep can` 看不到 CAN 接口；SDK 报 `SEND_MESSAGE_FAILED (100017)`。

**原因**：内核未编译 `gs_usb` 模块（Jetson 默认内核常见）。

**解决**（Jetson）：

```bash
# 检查
zcat /proc/config.gz | grep GS_USB
# 如果输出 "# CONFIG_CAN_GS_USB is not set"，则需要编译

# 1. 安装编译依赖
sudo apt install -y build-essential bc kmod flex bison libncurses-dev libssl-dev dwarves wget git

# 2. 获取与当前 uname -r 完全匹配的 Jetson 内核源码
# 3. 在源码目录执行：
zcat /proc/config.gz > .config
sed -i 's/CONFIG_LOCALVERSION=""/CONFIG_LOCALVERSION="-tegra"/' .config
make olddefconfig

export LOCALVERSION=-tegra
export IGNORE_PREEMPT_RT_PRESENCE=1
make -j$(nproc) modules

# 4. 安装
sudo mkdir -p /lib/modules/$(uname -r)/kernel/drivers/net/can/usb
sudo cp drivers/net/can/usb/gs_usb.ko /lib/modules/$(uname -r)/kernel/drivers/net/can/usb/
sudo depmod -a
sudo modprobe gs_usb

# 5. 设置开机自动加载
echo "gs_usb" | sudo tee /etc/modules-load.d/gs_usb.conf
```

### CAN 发送失败 / Message NOT sent

**恢复流程**：

```bash
CAN_IF="can_piper"   # 改成你的实际接口名

# 1. 关闭接口
sudo ip link set "$CAN_IF" down 2>/dev/null || true

# 2. 卸载驱动
sudo modprobe -r gs_usb

# 3. 物理拔掉 USB-CAN，等待 3 秒后重新插入
# 4. 机械臂断电再上电

# 5. 重新加载驱动
sudo modprobe gs_usb

# 6. 重新确认接口名并激活
ls /sys/class/net | grep can
sudo ip link set "$CAN_IF" type can bitrate 1000000
sudo ip link set "$CAN_IF" txqueuelen 1000
sudo ip link set "$CAN_IF" up
```

### 机械臂不响应运动指令

- 确认机械臂处于 **slave 模式**（非 master 模式），否则需重启臂体
- `judge_flag=False` 适用于第三方 USB-CAN 适配器
- 先跑 `arm/calibrate_offset.py` 验证连通性，再跑运动脚本

### 相机无画面

- 确认 Orbbec 相机 USB 已连接
- 确认 `camera_ip` 为空（使用本地相机）或填写了正确的远程相机 IP
- 检查 udev 规则是否已安装：`camera/scripts/` 中有 Orbbec 设备的 udev 配置文件

### 报错「没有手眼标定数据，无法转换图像坐标」

**原因**：`arm/hand-eye-data/2d_homography.npy` 不存在。该目录被 `.gitignore` 忽略，克隆后需自行标定生成。

**解决**：先执行 [手眼标定（抓取/采集前必须）](#手眼标定抓取采集前必须)，生成标定矩阵后再运行抓取/采集脚本。

---

### 灵巧手无响应（Modbus/RS485）

1. 确认设备名：`ls /dev/serial/by-id/`（推荐）或 `ls /dev/ttyUSB*`，与 `config.yaml` 的
   `linker_hand_modbus`、SDK 的 `LinkerHand/config/setting.yaml`（`RIGHT_HAND.MODBUS`）一致；
2. 权限：`sudo chmod 777 /dev/ttyUSB0` 或把用户加入 `dialout` 组；
3. 站号/手别：右手 0x27、左手 0x28（由 `linker_hand_type` 决定）；
4. 导入报 `No module named 'utils.mapping'`：SDK submodule 未打导入补丁，
   执行 `git -C third_party/linkerhand-python-sdk apply ../linkerhand-imports.patch`（详见第 6 节）；
5. 触摸/位置读数始终为 0：先 `uv run python linker_hand_open.py --hold 2` 看回读，再查 485 接线 A/B。

### 仿真加载失败（Error opening file ... .obj/.stl）

用**绝对路径**加载 `scene.xml`，或先 `cd sim/piper_linker` 再执行（mujoco 对相对路径 + OBJ 的行为如此，
源目录同样如此）；确认 `sim/piper_linker/piper/` 下 84 个网格与 `linkerhand_o6_left/meshes/` 12 个 STL 齐全。
