# Description: 平抓姿态 2D 手眼标定（像素 <-> 掌心/法兰 的联合模型）。
# 垂直模式（arm/calibrate_handeye_2d.py + 2d_homography.npy）假设"法兰 xy ==
# 末端 xy"（夹爪垂直朝下）；平抓模式下手掌沿工具 z 前伸出臂长 L，法兰与
# 掌心不是同一点，且偏移方向 = 接近朝向 = 抓取点方位角 + φ0
# （config linker_hand_flat_heading_offset_deg，默认 4.2°），随点位旋转——
# 因此"像素 -> 法兰"不满足单应性，必须联合拟合：
#     H   ：像素 -> 掌心平面（单应）
#     L   ：掌心到法兰的臂长（米）
#     palm_i = flange_i − L·u(azimuth(palm_i) + φ0)
# 其中 u(θ) = (cosθ, sinθ)，azimuth 用 palm 自身迭代 2~3 次（初值取法兰方位角）。
# 拟合办法：网格搜 L（先 1mm 粗扫 [0, 0.25]m，再在最优附近 0.1mm 细化），
# 每个 L 下对 (像素 -> 掌心) 做单应拟合（L 扫描用全点最小二乘），取重投影
# RMS 最小的 L。最终 H：n < RANSAC_MIN_POINTS(15) 用全点最小二乘（小样本用
# RANSAC 会挑最小集精确拟合、把其余点当外点丢弃，RMS 假性好看且 L 会退化），
# n ≥ 15 才用 RANSAC。拟合后会做退化检测并打印醒目报警：点数 <8、臂长 L
# 命中搜索边界（0/0.25m，即不可辨识）、最差 2 个点的序号与残差（供剔除）。
# 同时给出"L=0（纯单应）"的对照残差，说明为什么需要联合模型。
#
# 典型用法（标定脚本 arm/calibrate_arm_hand.py 已接入）：
#     H, L, stats = fit_flat_mapping(image_points, flange_xy, heading_offset_deg)
#     save_mapping(path, H, L, heading_offset_deg, stats, image_points, flange_xy)
#     mapping = load_mapping(path)          # FlatHandEyeMapping
#     palm_xy, flange_xy = mapping.predict(u, v)
#
# 注意：本模型假设相机固定（eye-to-hand）且标定/使用期间不复位相机；
# 臂载相机下像素到桌面的映射随位姿变化，单应+臂长模型不成立。
import os
import time

import cv2
import numpy as np

# 掌心 -> 法兰 的模型方向：palm = flange + SIGN * L * u(azimuth(palm) + φ0)。
# 取 -1 表示掌心在法兰的"接近方向（朝向）"一侧（手从外侧伸向物体）。
PALM_MODEL_SIGN = -1.0
# L 网格搜索范围与步长（米）
L_MIN_M, L_MAX_M = 0.0, 0.25
L_COARSE_STEP_M = 0.001
L_FINE_STEP_M = 0.0001
# findHomography RANSAC 阈值（米）
RANSAC_THRESH_M = 0.01
# 方位角迭代次数
AZIMUTH_ITERATIONS = 3
# 小样本不用 RANSAC：n 小于该值时最终 H 用全点最小二乘——RANSAC 在点少时会
# 挑最小集精确拟合、把其余点当外点丢弃（实测 n=6 时 4 点零残差、RMS 假性
# 好看且 L 退化到边界），n ≥ 该值才启用 RANSAC
RANSAC_MIN_POINTS = 15
# 建议的最少点数（不足则拟合不可靠，报警告）
RECOMMENDED_MIN_POINTS = 8
# 硬性下限（少于该值直接拒绝拟合）
MIN_POINTS_TO_FIT = 4
# L 命中搜索边界（0 或上界）的判定容差（米）
L_BOUNDARY_TOL_M = L_COARSE_STEP_M / 2


def _unit(angle_rad: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=-1)


def palms_from_flanges(
    flange_xy: np.ndarray, L_m: float, heading_offset_deg: float
) -> np.ndarray:
    """按联合模型由法兰点推掌心点（azimuth 自迭代）。

    Args:
        flange_xy: `(N, 2)` 法兰平面坐标，单位为米。
        L_m: 掌心到法兰的臂长，单位为米。
        heading_offset_deg: 接近朝向相对掌心方位角的偏移 φ0，单位为度。

    Returns:
        `(N, 2)` 掌心平面坐标，单位为米。
    """
    flange = np.asarray(flange_xy, dtype=float).reshape(-1, 2)
    palm = flange.copy()
    offset = np.deg2rad(float(heading_offset_deg))
    for _ in range(AZIMUTH_ITERATIONS):
        azimuth = np.arctan2(palm[:, 1], palm[:, 0])
        palm = flange + PALM_MODEL_SIGN * L_m * _unit(azimuth + offset)
    return palm


def flange_from_palm(
    palm_xy: np.ndarray, L_m: float, heading_offset_deg: float
) -> np.ndarray:
    """按联合模型由掌心点推法兰点（azimuth 取掌心自身）。"""
    palm = np.asarray(palm_xy, dtype=float).reshape(-1, 2)
    azimuth = np.arctan2(palm[:, 1], palm[:, 0])
    return palm - PALM_MODEL_SIGN * L_m * _unit(
        azimuth + np.deg2rad(float(heading_offset_deg))
    )


def _fit_single_L(
    image_points: np.ndarray,
    flange_xy: np.ndarray,
    heading_offset_deg: float,
    L_m: float,
    use_ransac: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """固定 L 时拟合单应并计算重投影残差（米）。

    Args:
        use_ransac: `True` 用 RANSAC（最终 H）；`False` 用全点最小二乘
            （L 扫描阶段用，避免少量内点被 RANSAC 精确拟合从而误选 L）。

    Returns:
        `(H, residuals_m, inlier_mask)`；拟合失败返回 `(None, None, None)`。
    """
    palm = palms_from_flanges(flange_xy, L_m, heading_offset_deg)
    if use_ransac:
        H, mask = cv2.findHomography(
            np.asarray(image_points, dtype=np.float64),
            palm.astype(np.float64),
            cv2.RANSAC,
            RANSAC_THRESH_M,
        )
    else:
        H, mask = cv2.findHomography(
            np.asarray(image_points, dtype=np.float64),
            palm.astype(np.float64),
            0,
        )
    if H is None:
        return None, None, None
    projected = cv2.perspectiveTransform(
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    residuals = np.linalg.norm(projected - palm, axis=1)
    return H, residuals, (mask.reshape(-1).astype(bool) if mask is not None else None)


def fit_flat_mapping(
    image_points: np.ndarray,
    flange_xy: np.ndarray,
    heading_offset_deg: float,
    L_range_m: tuple[float, float] = (L_MIN_M, L_MAX_M),
) -> tuple[np.ndarray, float, dict]:
    """拟合"像素 -> 掌心"单应 H 与臂长 L 的联合模型。

    Args:
        image_points: `(N, 2)` 像素坐标（用户点选/棋盘角点）。
        flange_xy: `(N, 2)` 每条记录的法兰平面坐标（米，来自 get_arm_pose）。
        heading_offset_deg: 接近朝向偏移 φ0（度，config
            linker_hand_flat_heading_offset_deg）。
        L_range_m: 臂长搜索范围，默认 [0, 0.25] m。

    Returns:
        `(H, L_m, stats)`：
        - `H`: 3x3 单应（像素 -> 掌心，米）
        - `L_m`: 最优臂长（米）
        - `stats`: 含 `residuals_m`/`rms_m`/`inlier_mask`/`n`/`n_inliers`/
          `fit_method`（小样本用全点最小二乘、n≥15 才 RANSAC）、
          `L_at_boundary`（L 命中搜索边界=不可辨识的退化标志，同时会打印
          醒目 Warning 与建议）、`warnings`、`worst_points`（最差 2 个点的
          1 起算索引与残差，供剔除重拟合），以及 `L=0` 纯单应对照
          `(H_l0, residuals_m_l0, rms_m_l0)`。
    """
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    flange_xy = np.asarray(flange_xy, dtype=np.float64).reshape(-1, 2)
    if len(image_points) != len(flange_xy):
        raise ValueError("像素点与法兰点数量不匹配")
    if len(image_points) < MIN_POINTS_TO_FIT:
        raise ValueError(f"标定点不足，至少需要 {MIN_POINTS_TO_FIT} 个点")

    def evaluate(L_m: float):
        # L 扫描用全点最小二乘 RMS 作为评分：RANSAC 在点少时可能只留少数
        # 内点并精确拟合（RMS≈0），会让 L 搜索误选到退化值（实测遇到过）。
        H, residuals, mask = _fit_single_L(
            image_points, flange_xy, heading_offset_deg, L_m, use_ransac=False
        )
        if H is None:
            return None
        return H, residuals, mask, float(np.sqrt(np.mean(residuals**2)))

    # 1mm 粗扫
    L_candidates = np.arange(
        L_range_m[0], L_range_m[1] + L_COARSE_STEP_M / 2, L_COARSE_STEP_M
    )
    best = None
    for L_m in L_candidates:
        result = evaluate(float(L_m))
        if result is None:
            continue
        if best is None or result[3] < best[1][3]:
            best = (L_m, result)
    if best is None:
        raise RuntimeError("单应性拟合失败，请检查标定点数据")
    L_coarse = float(best[0])

    # 0.1mm 细化（粗扫最优 ±2mm）
    fine_lo = max(L_range_m[0], L_coarse - 2 * L_COARSE_STEP_M)
    fine_hi = min(L_range_m[1], L_coarse + 2 * L_COARSE_STEP_M)
    for L_m in np.arange(fine_lo, fine_hi + L_FINE_STEP_M / 2, L_FINE_STEP_M):
        result = evaluate(float(L_m))
        if result is None:
            continue
        if result[3] < best[1][3]:
            best = (L_m, result)

    L_m = float(best[0])
    H, residuals, mask, rms_m = best[1]
    # 最终 H：n ≥ RANSAC_MIN_POINTS 才用 RANSAC；小样本用全点最小二乘，
    # 让 RMS/逐点残差反映全部点（否则 RANSAC 会挑最小集精确拟合、其余点被
    # 当外点丢弃，RMS 假性好看且 L 会退化到边界——实测 n=6 的踩坑）
    if len(image_points) >= RANSAC_MIN_POINTS:
        H_final, residuals_final, mask_final = _fit_single_L(
            image_points, flange_xy, heading_offset_deg, L_m, use_ransac=True
        )
        fit_method = f"RANSAC(n≥{RANSAC_MIN_POINTS})"
        if H_final is None:  # RANSAC 失败兜底
            H_final, residuals_final, mask_final = _fit_single_L(
                image_points, flange_xy, heading_offset_deg, L_m, use_ransac=False
            )
            fit_method = f"least_squares(RANSAC 失败回退, n≥{RANSAC_MIN_POINTS})"
    else:
        H_final, residuals_final, mask_final = _fit_single_L(
            image_points, flange_xy, heading_offset_deg, L_m, use_ransac=False
        )
        fit_method = f"least_squares(n={len(image_points)}<{RANSAC_MIN_POINTS})"
    if H_final is not None:
        H, residuals = H_final, residuals_final
        mask = mask_final if mask_final is not None else np.ones(len(residuals), bool)
        rms_m = float(np.sqrt(np.mean(residuals**2)))

    # ---- 退化检测与报警 ----
    warnings_list: list[str] = []
    if len(image_points) < RECOMMENDED_MIN_POINTS:
        warnings_list.append(
            f"点数不足（n={len(image_points)} < {RECOMMENDED_MIN_POINTS}），拟合不可靠，"
            "建议补采后再拟合"
        )
    L_at_boundary = bool(
        L_m <= L_range_m[0] + L_BOUNDARY_TOL_M
        or L_m >= L_range_m[1] - L_BOUNDARY_TOL_M
    )
    if L_at_boundary:
        warnings_list.append(
            f"臂长 L 不可辨识（命中搜索边界 {L_m * 1000:.1f}mm，搜索范围 "
            f"[{L_range_m[0] * 1000:.0f}, {L_range_m[1] * 1000:.0f}]mm）。"
            f"建议：点数 ≥{RECOMMENDED_MIN_POINTS}、点位方位角铺开 ≥±20°、"
            "检查下方最差点是否坏点（可剔除后重拟合）、检查是否有重复/近似重复点"
        )
    worst_idx = np.argsort(residuals)[::-1][:2] if residuals is not None else []
    worst_points = [(int(i) + 1, float(residuals[i])) for i in worst_idx]
    for msg in warnings_list:
        print("!" * 62)
        print(f"! Warning: {msg}")
        print("!" * 62)
    if worst_points:
        print(
            "最差 2 个点（1 起算，按采集顺序）: "
            + "  ".join(f"#{i} {r * 1000:.1f}mm" for i, r in worst_points)
        )
    print(f"最终单应拟合方式: {fit_method}，内点 {int(np.sum(mask))}/{len(image_points)}")

    # L=0 纯单应对照（说明联合模型的必要性）
    H_l0, residuals_l0, mask_l0 = _fit_single_L(
        image_points, flange_xy, heading_offset_deg, 0.0
    )
    rms_m_l0 = (
        float(np.sqrt(np.mean(residuals_l0**2))) if residuals_l0 is not None else float("nan")
    )

    # 方位角跨度（度）：跨度太小（点位方向集中）时 L 与单应平移项近乎不可分，
    # 标定数据应铺开方位角范围（建议 ≥±20°），该值供调用方提示用户
    palm_azimuth_deg = np.rad2deg(
        np.arctan2(
            palms_from_flanges(flange_xy, L_m, heading_offset_deg)[:, 1],
            palms_from_flanges(flange_xy, L_m, heading_offset_deg)[:, 0],
        )
    )
    azimuth_spread_deg = float(palm_azimuth_deg.max() - palm_azimuth_deg.min())

    stats = {
        "L_m": L_m,
        "residuals_m": residuals,
        "rms_m": rms_m,
        "inlier_mask": mask,
        "n": int(len(image_points)),
        "n_inliers": int(mask.sum()) if mask is not None else int(len(image_points)),
        "fit_method": fit_method,
        "L_at_boundary": L_at_boundary,
        "warnings": warnings_list,
        "worst_points": worst_points,
        "heading_offset_deg": float(heading_offset_deg),
        "azimuth_spread_deg": azimuth_spread_deg,
        "H_l0": H_l0,
        "residuals_m_l0": residuals_l0,
        "rms_m_l0": rms_m_l0,
    }
    return H, L_m, stats


class FlatHandEyeMapping:
    """平抓 2D 标定映射（像素 -> 掌心/法兰）。"""

    def __init__(
        self,
        H: np.ndarray,
        L_m: float,
        heading_offset_deg: float,
        rms_m: float = float("nan"),
        rms_m_l0: float = float("nan"),
        n_points: int | None = None,
        timestamp: str | None = None,
        residuals_m: np.ndarray | None = None,
        image_points: np.ndarray | None = None,
        flange_xy: np.ndarray | None = None,
        palm_xy: np.ndarray | None = None,
    ):
        self.H = np.asarray(H, dtype=np.float64)
        self.L_m = float(L_m)
        self.heading_offset_deg = float(heading_offset_deg)
        self.rms_m = float(rms_m)
        self.rms_m_l0 = float(rms_m_l0)
        self.n_points = n_points
        self.timestamp = timestamp
        self.residuals_m = residuals_m
        self.image_points = image_points
        self.flange_xy = flange_xy
        self.palm_xy = palm_xy

    def predict(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """像素 -> 掌心/法兰平面坐标（米）。

        Returns:
            `(palm_xy, flange_xy)`，均为长度 2 的数组。
        """
        homogeneous = self.H @ np.array([u, v, 1.0], dtype=np.float64)
        if homogeneous[2] == 0:
            raise ValueError(f"单应映射出现除零（u={u}, v={v}）")
        palm_xy = homogeneous[:2] / homogeneous[2]
        flange_xy = flange_from_palm(
            palm_xy.reshape(1, 2), self.L_m, self.heading_offset_deg
        ).reshape(2)
        return palm_xy, flange_xy

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            H=self.H,
            L_m=self.L_m,
            heading_offset_deg=self.heading_offset_deg,
            rms_m=self.rms_m,
            rms_m_l0=self.rms_m_l0,
            n=self.n_points if self.n_points is not None else -1,
            timestamp=self.timestamp
            if self.timestamp is not None
            else time.strftime("%Y-%m-%dT%H:%M:%S"),
            residuals_m=self.residuals_m
            if self.residuals_m is not None
            else np.zeros(0),
            image_points=self.image_points
            if self.image_points is not None
            else np.zeros((0, 2)),
            flange_xy=self.flange_xy if self.flange_xy is not None else np.zeros((0, 2)),
            palm_xy=self.palm_xy if self.palm_xy is not None else np.zeros((0, 2)),
        )
        return path

    @classmethod
    def load(cls, path: str) -> "FlatHandEyeMapping":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"平抓标定文件不存在: {path}（请先运行平抓标定脚本臂/掌心标定）"
            )
        with np.load(path) as data:
            return cls(
                H=data["H"],
                L_m=float(data["L_m"]),
                heading_offset_deg=float(data["heading_offset_deg"]),
                rms_m=float(data["rms_m"]) if "rms_m" in data else float("nan"),
                rms_m_l0=float(data["rms_m_l0"]) if "rms_m_l0" in data else float("nan"),
                n_points=int(data["n"]) if "n" in data else None,
                timestamp=str(data["timestamp"]) if "timestamp" in data else None,
                residuals_m=data["residuals_m"] if "residuals_m" in data else None,
                image_points=data["image_points"] if "image_points" in data else None,
                flange_xy=data["flange_xy"] if "flange_xy" in data else None,
                palm_xy=data["palm_xy"] if "palm_xy" in data else None,
            )


def save_mapping(
    path: str,
    H: np.ndarray,
    L_m: float,
    heading_offset_deg: float,
    stats: dict,
    image_points: np.ndarray,
    flange_xy: np.ndarray,
) -> str:
    """保存联合标定结果到 npz（H、L_m、φ0、rms、n、时间戳、原始点）。"""
    mapping = FlatHandEyeMapping(
        H=H,
        L_m=L_m,
        heading_offset_deg=heading_offset_deg,
        rms_m=stats.get("rms_m", float("nan")),
        rms_m_l0=stats.get("rms_m_l0", float("nan")),
        n_points=stats.get("n", len(image_points)),
        residuals_m=stats.get("residuals_m"),
        image_points=np.asarray(image_points, dtype=float),
        flange_xy=np.asarray(flange_xy, dtype=float),
        palm_xy=palms_from_flanges(
            np.asarray(flange_xy, dtype=float), L_m, heading_offset_deg
        ),
    )
    return mapping.save(path)


def load_mapping(path: str) -> FlatHandEyeMapping:
    """加载平抓标定 npz；文件不存在时抛出带操作提示的 FileNotFoundError。"""
    return FlatHandEyeMapping.load(path)


if __name__ == "__main__":
    # 合成数据自检：已知 H_true/L_true 生成 (像素, 法兰) 样本，验证 L 复原。
    rng = np.random.default_rng(0)
    H_true = np.array(
        [[0.00035, -0.00005, 0.18], [0.00004, 0.00034, -0.12], [0.000002, 0.000001, 1.0]]
    )
    L_true = 0.09
    phi0 = 4.2
    palms = np.column_stack(
        [
            0.15 + 0.25 * rng.random(12),
            -0.15 + 0.30 * rng.random(12),
        ]
    )
    H_inv = np.linalg.inv(H_true)
    pixels = cv2.perspectiveTransform(
        palms.reshape(-1, 1, 2).astype(np.float64), H_inv
    ).reshape(-1, 2)
    pixels += rng.normal(0, 1.5, pixels.shape)  # 1.5 像素噪声
    flanges = flange_from_palm(palms, L_true, phi0)
    H, L_m, stats = fit_flat_mapping(pixels, flanges, phi0)
    print(f"L_true={L_true:.4f}m  L_fit={L_m:.4f}m  误差={abs(L_m-L_true)*1000:.2f}mm")
    print(f"联合模型 RMS={stats['rms_m']*1000:.2f}mm  L=0 纯单应 RMS={stats['rms_m_l0']*1000:.2f}mm")
