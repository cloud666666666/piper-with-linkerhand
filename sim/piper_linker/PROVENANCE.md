# Piper + Linker Hand O6 数据溯源与修改清单

生成日期：2026-09-10
审计方式：网格逐字节校验和对比、MJCF 与官方 URDF 数值级比对（运动学 11 连杆 × 10 随机位姿最大偏差 1.2e-6、惯量全匹配、关节轴全一致）。

源项目：

- **A** = `agx_arm_sim/`（Piper 官方仿真，来源 AgileX）
- **B** = `linkerhand-urdf/`（Linker Hand 官方 URDF，来源灵心巧手）
- **C** = `/home/czn/mujoco/灵巧手连接件.STEP`（用户提供，臂-手连接件 CAD）

---

## 一、文件级清单

| 本项目文件 | 来源 | 状态 |
|---|---|---|
| `piper/piper.xml` | A: `mujoco/agilex_arm/agilex_piper/piper.xml` | **修改**（见第二节） |
| `piper/assets/`（84 个网格） | A: `mujoco/agilex_arm/agilex_piper/assets/` | **未改**，字节级一致 |
| `piper/` 根目录的 84 个网格 | 同上 | 未改，但是**冗余副本**（路径调试产生） |
| `assets/piper/`（84 个网格） | 同上 | 未改，冗余副本，**未被引用可删** |
| `linkerhand_o6_left/meshes/`（12 个 STL） | B: `O6/left/meshes/` | **未改**，字节级一致 |
| `linkerhand_o6_left/linkerhand_o6_left.xml` | B: `O6/left/linkerhand_o6_left.urdf` 转换生成 | 数值全部来自 URDF（转换工具产出，已验证） |
| `piper/piper.xml` 内嵌的手指树 | 同上 | 数值全部来自 URDF（见第三节） |
| `scene.xml` | 自建 | 场景组合文件（地板/灯光/执行器/联动约束） |
| `tools/urdf_to_mjcf.py` | 自建 | URDF→MJCF 转换器 |
| `server.py` / `main.py` / `README.md` | 自建 | Web viewer 与本机启动脚本 |

---

## 二、`piper/piper.xml` 相对官方原版的修改

### 改了

| 项 | 原版 | 修改后 | 原因 |
|---|---|---|---|
| `<compiler meshdir>` | `assets` | `.` | 目录布局 |
| 聚光灯 `target` | `link8` | `lh_hand_base_link` | link8 已删除 |
| link7/link8 两指夹爪（含 joint7/8、碰撞盒） | 存在 | **删除** | 换装灵巧手 |
| joint8=−joint7 联动等式 | 存在 | 删除 | 随夹爪移除 |
| `gripper` 执行器 | 存在 | 删除 | 随夹爪移除 |
| keyframe `home` | qpos 8 值 / ctrl 7 值 | 6 值 / 6 值（保留的前 6 个数值不变） | 去掉两个手指关节 |
| 在 link6 下新增 | — | O6 手指树（12 个 body，见下节） | 本次任务本体 |

### 没改

- 臂部 6 个关节的**全部参数**：位置/四元数/惯量/关节轴/限位/执行器增益（kp=80/80/80/40/10/10, kv=5/5/5/5/1.5/1.5）
- 全部材质定义（gray_mat、red_mat 等 9 种颜色）
- 全部 84 个网格引用
- `default` 类、`option`（integrator/cone/impratio）、`contact` 排除项
- base_link→link1→…→link6 的整棵运动树

---

## 三、手指树数据来源（`piper/piper.xml` 内嵌部分）

### 完全来自 B: `O6/left/linkerhand_o6_left.urdf`

| 数据 | 验证 |
|---|---|
| 碰撞几何 | 手部 12 个碰撞体 = URDF 碰撞网格（type="mesh"）。注意：需显式声明 `type="mesh"`，否则会继承 Piper 默认 class 的 `type="capsule"` 退化为胶囊代理（2026-09-10 修复，此前拇指两节的胶囊体曾显示为两个"菱形"） |
|---|---|
| 12 个 body 的 pos/quat | 与 URDF 数值级一致（偏差 ≤1.2e-6，文本舍入级） |
| 12 个 body 的质量/质心/惯量张量 | 特征值分解后全匹配（拇指树曾遗漏 11 个，2026-09-10 已回填修复） |
| 11 个关节的轴、驱动关节限位 | 全一致（thumb_cmc_yaw 0–1.3、thumb_cmc_pitch 0–0.58、四指 MCP 0–1.6） |
| mimic 联动比 | 2.29（拇指 IP）、0.89（四指 DIP），与 URDF `<mimic multiplier>` 相同 |
| 材质颜色 | hand_silver=0.5 0.5 0.52、hand_black=0.15 0.15 0.15，来自 URDF `<material>` |
| 手基座安装位 | link6 + 工具轴×17.5mm（同轴，经连接件安装），依据 C 的 CAD 几何；link6 外观已裁剪至法兰面（`link6_trimmed.stl`，前伸凸台真机已被灵巧手替换） |
| `piper/adapter.stl` | C: `灵巧手连接件.STEP` 经 gmsh 网格化 + 坐标变换（mm→m、CAD 装配系→link6 法兰系） | 由 CAD 生成，非手工建模 |
| link6 视觉网格 | 官方 `link6.stl` 含法兰前 40mm 凸台（真机已被灵巧手替换）。已生成 `link6_trimmed.stl`（裁至法兰面 + 封盖圆盘）并用于外观；碰撞胶囊未动；原行注释保留可一键切回 |

### 有意的设计差异（与 URDF 字面不同，已声明）

| 项 | URDF 原值 | 本项目 | 原因 |
|---|---|---|---|
| mimic 子关节限位（dip/ip） | 1.43 / 1.08 | 1.424 / 1.3282（=联动比×驱动范围） | MuJoCo 等式约束需子关节能跟满驱动范围；URDF 的 1.08 与自身 mimic 比 2.29×0.58=1.33 矛盾 |
| mimic 实现方式 | `<mimic>` 标签 | `<equality><joint polycoef>` | MuJoCo 不支持 URDF mimic |

### 两个源项目都没有、为仿真稳定自定的参数

| 参数 | 值 | 说明 |
|---|---|---|
| 手部执行器增益 | kp=2, kv=0.2, forcerange ±2 | URDF 仅有 effort=100；实测 kp=30 会发散 |
| 等式约束 solref | 0.02 1 | 接近 MuJoCo 默认的软约束参数 |
| 独立手模型关节阻尼 | damping 0.05 / frictionloss 0.15 / armature 0.001 | 仅 `linkerhand_o6_left.xml`，场景嵌入树未使用 |

---

## 四、`scene.xml`（自建，无对应源文件）

来自 A 的：`<include file="piper/piper.xml">` 整体。
来自 B 的：12 个手网格路径、2 个材质定义（数值同 URDF）。
自建的：地板、灯光、`timestep=0.001`、link6↔手基座接触排除、6 个手部执行器、5 个 mimic 等式。

---

## 五、网页/脚本（`server.py`、`main.py`、`tools/`）

全部为本次开发自建，不含任何源项目数据。
其中手势预设（握拳/比耶/OK 等）与臂姿态预设的关节值为经验设定，非硬件标定数据。

---

## 六、已知遗留

- Piper 网格存在三份拷贝（`piper/assets/`、`piper/` 根、`assets/piper/`），当前仅前两份被引用，可清理 `assets/` 目录及 `piper/` 根副本其一。
- `.render/` 为调试期渲染临时目录，可删。
