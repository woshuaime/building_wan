"""
OBJ 模型加载器 + 严格规范化（导师规范化要求）

规范化目标（在 world 系下）：
    1) 几何中心 -> 原点 (0, 0, 0)
    2) RANSAC 拟合地面 -> 法向对齐到 +Z（建筑竖直站立）
    3) X-Y 主方向 -> +X（yaw 锁死，避免多视角拍摄时绕 Z 摆歪）
    4) 缩放到 TARGET_EXTENT 单位以内（默认 12.0）

返回值结构保持与旧版兼容：
    load() -> (vertices, normals, faces)
    vertices 已包含上述 4 步规范化后的世界坐标
"""

import numpy as np


# Open3D 延迟导入：仅在真正执行规范化时才需要，未安装时给出清晰提示
try:
    import open3d as o3d  # noqa: F401
    _HAS_OPEN3D = True
except Exception:  # ImportError 或 PyInstaller 打包缺模块等
    o3d = None
    _HAS_OPEN3D = False


# 规范化目标包围盒最大边长（与原 _normalize 保持一致：12 个单位）
TARGET_EXTENT = 12.0

# RANSAC 平面拟合参数
RANSAC_DISTANCE_THRESHOLD = 0.05   # 与点距（按归一化前米/单位粗估 5cm 即可）
RANSAC_ITERATIONS = 1000
RANSAC_MIN_POINTS = 3
BOTTOM_BOUNDARY_MIN_AREA_RATIO = 0.03
BOTTOM_BOUNDARY_TRIANGLE_MIN_ALIGNMENT = 0.80
BOTTOM_BOUNDARY_NORMAL_MIN_ALIGNMENT = 0.95


def parse_obj_file(filepath):
    """Parse an OBJ file without normalization or coordinate changes.

    Dataset Mode depends on this raw parser so prealigned triplets can be loaded
    exactly as exported by Blender.
    """
    vertices = []
    normals = []
    faces = []

    def parse_stream(stream):
        vertices.clear()
        normals.clear()
        faces.clear()
        for line in stream:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'vn':
                normals.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                face = []
                n_v = len(vertices)
                for part in parts[1:]:
                    indices = part.split('/')
                    raw = int(indices[0])
                    if raw < 0:
                        vertex_idx = n_v + raw
                    else:
                        vertex_idx = raw - 1
                    face.append(vertex_idx)
                faces.append(face)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            parse_stream(f)
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            parse_stream(f)

    return vertices, normals, faces


class OBJLoader:
    """OBJ模型加载器（含严格规范化）"""

    def __init__(self):
        self.vertices = []
        self.normals = []
        self.faces = []

        # 规范化状态（对外暴露，便于调用方读取/调试）
        self.model_center = np.array([0.0, 0.0, 0.0])  # 居中前几何中心
        self.model_scale = 1.0                          # 最终缩放系数
        self.ground_normal = np.array([0.0, 0.0, 1.0])  # RANSAC 拟合得到的地面法向（已对齐 +Z）
        self.yaw_axis = np.array([1.0, 0.0])           # X-Y 平面主方向（已对齐到 target_dir_xy，默认 +X）
        self.yaw_angle_deg = 0.0                       # 实际应用的 yaw 旋转角度（度）
        self.bounds_after = None                        # 规范化后 AABB（min, max, size, center）
        self.normalization_report = {
            "status": "not_run",
            "fallback_used": True,
            "reason": "",
        }

    # ------------------------------------------------------------------
    # OBJ 解析（与旧版完全一致：v / vn / f，支持负索引，UTF-8 -> latin-1 回退）
    # ------------------------------------------------------------------
    def load(self, filepath, target_dir_xy=None):
        """加载OBJ文件，并执行规范化。返回值结构与旧版一致。

        参数:
            filepath: OBJ 文件路径。
            target_dir_xy: 可选，X-Y 平面上希望建筑"主面/头顶"指向的目标方向。
                可以是：
                  - None 或 (1, 0)：向后兼容，等价于"主方向对齐 +X"。
                  - 长度为 2 的可迭代对象 (dx, dy)，会被归一化。
                  - 长度为 3 的可迭代对象 (dx, dy, _)，自动忽略 Z 分量。
                若传入的是空间中的"目标点"（非方向），调用方应先做 (p - center) 后归一化。
        """
        self.vertices = []
        self.normals = []
        self.faces = []
        self.orient_dir_xy = None   # 从 OBJ 注释行解析的朝向元数据（XY平面单位向量）
        self.normalization_report = {
            "status": "not_run",
            "fallback_used": True,
            "reason": "",
        }

        # ── 解析朝向元数据（# orient: x y）──────────────────────────────
        # 支持格式：# orient: 1 0   /   #orient: 0.707 0.707
        # 若文件不含此行，orient_dir_xy = None，load() 末行按 None 处理（默认 +X）
        for line in open(filepath, 'r', encoding='utf-8'):
            stripped = line.strip()
            if stripped.startswith('#') and 'orient' in stripped.lower():
                parts = stripped.split()
                nums = []
                for p in parts[1:]:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        continue
                if len(nums) >= 2:
                    dx, dy = nums[0], nums[1]
                    n = float(np.hypot(dx, dy))
                    if n > 1e-9:
                        self.orient_dir_xy = np.array([dx/n, dy/n], dtype=np.float64)
                break

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        self.vertices.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'vn':
                        self.normals.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'f':
                        face = []
                        n_v = len(self.vertices)
                        for part in parts[1:]:
                            indices = part.split('/')
                            raw = int(indices[0])
                            if raw < 0:
                                vertex_idx = n_v + raw
                            else:
                                vertex_idx = raw - 1
                            face.append(vertex_idx)
                        self.faces.append(face)
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == 'v':
                        self.vertices.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'vn':
                        self.normals.append([float(x) for x in parts[1:4]])
                    elif parts[0] == 'f':
                        face = []
                        n_v = len(self.vertices)
                        for part in parts[1:]:
                            indices = part.split('/')
                            raw = int(indices[0])
                            if raw < 0:
                                vertex_idx = n_v + raw
                            else:
                                vertex_idx = raw - 1
                            face.append(vertex_idx)
                        self.faces.append(face)

        # 关键：规范化必须执行；这里会用到 Open3D
        # orient_dir_xy（元数据解析的朝向）优先于调用方传入的 target_dir_xy
        effective_dir = self.orient_dir_xy if self.orient_dir_xy is not None else target_dir_xy
        if self.orient_dir_xy is not None:
            print(f"[OBJLoader] 读取到朝向元数据 orient -> ({self.orient_dir_xy[0]:+.4f}, {self.orient_dir_xy[1]:+.4f})，用于 yaw 对齐")
        self._normalize(target_dir_xy=effective_dir)
        return self.vertices, self.normals, self.faces

    # ------------------------------------------------------------------
    # 严格规范化（4 步）
    # ------------------------------------------------------------------
    def _normalize(self, target_dir_xy=None):
        """4 步规范化：居中 -> 地面法向 -> Z -> yaw -> 目标方向 -> 缩放

        参数:
            target_dir_xy: X-Y 平面上希望建筑主方向指向的目标单位方向。
                None 时默认为 (1, 0)（向后兼容）。
                若传入接近零向量，会被规约为 (1, 0)。

        失败兜底：若 Open3D 不可用 / 顶点不足 / RANSAC 退化，回退到
        旧版"几何中心 + 缩放"的简化逻辑，保证 load() 不会直接抛异常。
        """
        if not self.vertices:
            return

        # ── 解析 target_dir_xy：第 3 步 yaw 对齐仍在使用，默认对齐到 +X ───
        # 历史背景：外部可能仍传 target_dir_xy；这里继续保持接口稳定，并用
        # t_xy / target_angle 驱动后续 AABB 主方向 yaw 对齐。
        # 解析出的 t_xy / target_angle 会继续供后续 yaw 对齐使用。
        if target_dir_xy is None:
            t_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            t = np.asarray(target_dir_xy, dtype=np.float64).reshape(-1)[:2]
            n = float(np.linalg.norm(t))
            if n < 1e-9:
                t_xy = np.array([1.0, 0.0], dtype=np.float64)
            else:
                t_xy = t / n
        # 把角度归一到 [-π, π]
        target_angle = float(np.arctan2(t_xy[1], t_xy[0]))

        if not _HAS_OPEN3D:
            self._normalize_fallback(
                reason="未检测到 open3d，请先执行: pip install open3d",
                target_dir_xy=target_dir_xy,
            )
            return

        try:
            verts = np.asarray(self.vertices, dtype=np.float64)

            # ── 0. 构建 Open3D 点云/三角网格 ─────────────────────────────
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)

            mesh = None
            if self.faces:
                # 仅保留 >=3 顶点的三角/多边形面，转成三角形（Open3D TriangleMesh 仅吃三角面）
                tri_faces = []
                for f in self.faces:
                    if len(f) < 3:
                        continue
                    # 简单扇形三角化：f[0], f[i], f[i+1]
                    for i in range(1, len(f) - 1):
                        tri_faces.append([int(f[0]), int(f[i]), int(f[i + 1])])
                if tri_faces:
                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(verts)
                    mesh.triangles = o3d.utility.Vector3iVector(tri_faces)

            # ── 1. 求几何中心并平移到原点 ─────────────────────────────────
            center_before = pcd.get_center()  # = mean(points)
            self.model_center = np.asarray(center_before, dtype=np.float64)
            pcd.translate(-self.model_center)
            if mesh is not None:
                mesh.translate(-self.model_center)

            # ── 2. 地面法向量对齐（ground -> +Z） ────────────────────────
            # 建筑 OBJ 里墙面、屋顶、底面都可能是大平面。单次 RANSAC 会倾向于
            # "面积/点数最大"的平面，可能误选竖墙；这里枚举多个候选，再选
            # 最接近源坐标 +Z 的水平顶/底候选。
            z_axis = np.array([0.0, 0.0, 1.0])
            plane_candidates = []
            pts_centered = np.asarray(pcd.points, dtype=np.float64)
            x_span = float(np.ptp(pts_centered[:, 0])) if len(pts_centered) else 0.0
            y_span = float(np.ptp(pts_centered[:, 1])) if len(pts_centered) else 0.0
            z_span = float(np.ptp(pts_centered[:, 2])) if len(pts_centered) else 0.0
            footprint_area2 = max(2.0 * x_span * y_span, 1e-9)
            bottom_z = float(np.min(pts_centered[:, 2])) if len(pts_centered) else 0.0
            bottom_tol = max(z_span * 0.03, 1e-8)
            bottom_area2 = 0.0
            bottom_normal_sum = np.zeros(3, dtype=np.float64)
            bottom_centroid_sum = np.zeros(3, dtype=np.float64)
            bottom_indices = []
            if self.faces:
                for f in self.faces:
                    if len(f) < 3:
                        continue
                    face_indices = [int(idx) for idx in f if 0 <= int(idx) < len(pts_centered)]
                    if len(face_indices) < 3:
                        continue
                    for i in range(1, len(face_indices) - 1):
                        tri = [face_indices[0], face_indices[i], face_indices[i + 1]]
                        p0, p1, p2 = pts_centered[tri[0]], pts_centered[tri[1]], pts_centered[tri[2]]
                        n_i = np.cross(p1 - p0, p2 - p0)
                        area2 = float(np.linalg.norm(n_i))
                        if area2 < 1e-12:
                            continue
                        n_i /= area2
                        if np.dot(n_i, z_axis) < 0.0:
                            n_i = -n_i
                        alignment = float(abs(np.dot(n_i, z_axis)))
                        centroid = (p0 + p1 + p2) / 3.0
                        if (
                            centroid[2] <= bottom_z + bottom_tol
                            and alignment >= BOTTOM_BOUNDARY_TRIANGLE_MIN_ALIGNMENT
                        ):
                            bottom_area2 += area2
                            bottom_normal_sum += n_i * area2
                            bottom_centroid_sum += centroid * area2
                            bottom_indices.extend(tri)
                        d_i = float(-np.dot(n_i, p0))
                        plane_candidates.append(
                            (
                                2,
                                alignment,
                                area2,
                                [float(n_i[0]), float(n_i[1]), float(n_i[2]), d_i],
                                n_i,
                                np.asarray(tri, dtype=np.int64),
                                "mesh_face",
                            )
                        )

            if bottom_area2 / footprint_area2 >= BOTTOM_BOUNDARY_MIN_AREA_RATIO:
                bottom_normal = bottom_normal_sum / max(float(np.linalg.norm(bottom_normal_sum)), 1e-12)
                bottom_alignment = float(abs(np.dot(bottom_normal, z_axis)))
                bottom_centroid = bottom_centroid_sum / max(bottom_area2, 1e-12)
                bottom_d = float(-np.dot(bottom_normal, bottom_centroid))
                if bottom_alignment >= BOTTOM_BOUNDARY_NORMAL_MIN_ALIGNMENT:
                    plane_candidates.append(
                        (
                            3,
                            bottom_alignment,
                            bottom_area2,
                            [
                                float(bottom_normal[0]),
                                float(bottom_normal[1]),
                                float(bottom_normal[2]),
                                bottom_d,
                            ],
                            bottom_normal,
                            np.asarray(sorted(set(bottom_indices)), dtype=np.int64),
                            "mesh_bottom_boundary",
                        )
                    )

            remaining = pcd
            remaining_indices = np.arange(len(verts), dtype=np.int64)
            max_plane_trials = 8
            for _ in range(max_plane_trials):
                if len(remaining.points) < RANSAC_MIN_POINTS:
                    break
                plane_model_i, inliers_i = remaining.segment_plane(
                    distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                    ransac_n=RANSAC_MIN_POINTS,
                    num_iterations=RANSAC_ITERATIONS,
                )
                if len(inliers_i) < RANSAC_MIN_POINTS:
                    break

                n_i = np.array(plane_model_i[:3], dtype=np.float64)
                n_norm = np.linalg.norm(n_i)
                if n_norm >= 1e-9:
                    n_i /= n_norm
                    if np.dot(n_i, z_axis) < 0.0:
                        n_i = -n_i
                    alignment = float(abs(np.dot(n_i, z_axis)))
                    global_inliers = remaining_indices[np.asarray(inliers_i, dtype=np.int64)]
                    plane_candidates.append(
                        (1, alignment, int(len(inliers_i)), plane_model_i, n_i, global_inliers, "ransac_point_cloud")
                    )

                remaining = remaining.select_by_index(inliers_i, invert=True)
                remaining_indices = np.delete(remaining_indices, np.asarray(inliers_i, dtype=np.int64))

            if not plane_candidates:
                raise RuntimeError("RANSAC 没有找到可用平面候选")

            plane_candidates.sort(key=lambda item: (item[0], item[1] >= 0.95, item[2], item[1]), reverse=True)
            _, ground_alignment_before, _, plane_model, ground_normal, inliers, normalizer_source = plane_candidates[0]

            # 计算将 ground_normal -> +Z 的最小旋转（旋转轴 = n × Z，角度 = arccos(n·Z)）
            R_ground = _rotation_between_vectors(ground_normal, z_axis)

            pcd.rotate(R_ground, center=(0.0, 0.0, 0.0))
            if mesh is not None:
                mesh.rotate(R_ground, center=(0.0, 0.0, 0.0))

            # 重新计算 inliers 平面方程的 d'（旋转后 d 应该接近 0，因为中心已在原点）
            # 这里仅用作日志，不影响后续
            d_after = float(plane_model[3])

            # ── 3. Yaw 对齐到 target_dir_xy ──────────────────────────────
            # 基于 X-Y 平面 AABB 长轴方向：找到模型在地面上的"主朝向"，
            # 然后绕 Z 轴旋转使其对齐到 target_dir_xy（默认 +X，即四轨交叉点方向）。
            pts_ground = np.asarray(pcd.points)[:, :2]   # 仅 XY 投影
            x_min, y_min = pts_ground.min(axis=0)
            x_max, y_max = pts_ground.max(axis=0)
            dx = x_max - x_min
            dy = y_max - y_min
            if abs(dx - dy) > 1e-6:
                # 长轴方向（X 轴更长则主轴沿 X，否则沿 Y）
                main_dir = np.array([1.0, 0.0]) if dx >= dy else np.array([0.0, 1.0])
            else:
                # 退化：使用 PCA 首主成分
                cov = np.cov((pts_ground - pts_ground.mean(axis=0)).T)
                _, eigvecs = np.linalg.eigh(cov)
                principal = eigvecs[:, -1]
                main_dir = np.array([float(principal[0]), float(principal[1])])
                n_m = float(np.linalg.norm(main_dir))
                if n_m > 1e-9:
                    main_dir /= n_m

            # 计算绕 Z 轴旋转：main_dir -> target_dir_xy
            # 旋转角 = atan2(t×m) / atan2(m×t)，用向量叉积符号判断方向
            if np.linalg.norm(t_xy) < 1e-9:
                t_xy[:] = [1.0, 0.0]  # 退化：默认 +X

            cross_val = float(t_xy[0] * main_dir[1] - t_xy[1] * main_dir[0])
            dot_val   = float(t_xy[0] * main_dir[0] + t_xy[1] * main_dir[1])
            yaw_angle = float(np.arctan2(cross_val, dot_val))
            c, s = np.cos(yaw_angle), np.sin(yaw_angle)
            R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

            pcd.rotate(R_yaw, center=(0.0, 0.0, 0.0))
            if mesh is not None:
                mesh.rotate(R_yaw, center=(0.0, 0.0, 0.0))

            # ── 4. 缩放到 TARGET_EXTENT ─────────────────────────────────
            pts_after = np.asarray(pcd.points)
            lo = pts_after.min(axis=0)
            hi = pts_after.max(axis=0)
            max_extent = float((hi - lo).max())
            self.model_scale = (TARGET_EXTENT / max_extent) if max_extent > 1e-9 else 1.0

            pcd.scale(self.model_scale, center=(0.0, 0.0, 0.0))
            if mesh is not None:
                mesh.scale(self.model_scale, center=(0.0, 0.0, 0.0))

            # 同步到 self.vertices（以世界坐标形式输出，调用方可直接用）
            final_pts = np.asarray(pcd.points, dtype=np.float64)
            self.vertices = final_pts.tolist()

            # 如果原 normals 存在，按相同旋转应用到 normals（仅旋转，不缩放）
            if self.normals:
                nms = np.asarray(self.normals, dtype=np.float64)
                # 注意：法向不应被平移、也不应被等比缩放改变方向
                R_full = R_yaw @ R_ground   # 第3步 yaw 已启用
                nms_rot = nms @ R_full.T
                # 重新归一化，防止浮点误差累积
                norms = np.linalg.norm(nms_rot, axis=1, keepdims=True)
                norms[norms < 1e-9] = 1.0
                nms_rot = nms_rot / norms
                self.normals = nms_rot.tolist()

            # 记录最终 AABB
            lo = final_pts.min(axis=0)
            hi = final_pts.max(axis=0)
            self.bounds_after = {
                'min': lo,
                'max': hi,
                'center': (lo + hi) / 2.0,
                'size': hi - lo,
            }
            self.ground_normal = z_axis.copy()                              # 规范化后必然是 +Z
            self.yaw_axis = t_xy.copy()                                    # yaw 目标方向（从元数据或默认 +X）
            self.yaw_angle_deg = float(np.degrees(yaw_angle))              # 实际 yaw 旋转角度（度）
            self.normalization_report = {
                "status": "ground_normalized",
                "fallback_used": False,
                "reason": "",
                "normalizer_source": normalizer_source,
                "ground_inliers": int(len(inliers)),
                "vertex_count": int(len(final_pts)),
                "inlier_ratio": float(len(inliers) / max(len(final_pts), 1)),
                "ground_axis_alignment_before": float(ground_alignment_before),
                "model_scale": float(self.model_scale),
                "yaw_angle_deg": float(self.yaw_angle_deg),
            }

            print(
                f"[规范化] 中心 {self.model_center} -> 原点  | "
                f"地面法向 -> +Z ({normalizer_source}, inliers={len(inliers)})  | "
                f"yaw -> {t_xy[0]:+.3f},{t_xy[1]:+.3f} ({self.yaw_angle_deg:+.2f}°)  | "
                f"缩放 {self.model_scale:.4f}  | "
                f"d_after={d_after:.4f}"
            )

        except Exception as e:
            # 任何 Open3D / RANSAC / 数值异常都回退到旧版简化逻辑
            self._normalize_fallback(reason=f"严格规范化失败：{e}", target_dir_xy=target_dir_xy)

    def _normalize_fallback(self, reason: str, target_dir_xy=None):
        """回退到旧版"几何中心 + 缩放"逻辑 + AABB yaw 旋转，保证 load() 不崩。

        该兜底不会满足导师的"RANSAC 地面法向"要求（因为没装 open3d），
        但仍会完成：
          - 几何中心 -> 原点
          - AABB 长轴 -> target_dir_xy（默认 +X），与 Open3D 路径选向策略一致
          - 缩放到 TARGET_EXTENT

        参数:
            reason: 触发回退的原因（用于日志）
            target_dir_xy: 与 _normalize 相同的 X-Y 目标方向
        """
        print(f"[规范化] 回退到简化逻辑：{reason}")
        verts = np.asarray(self.vertices, dtype=np.float64)
        if verts.size == 0:
            return

        # ── 解析 target_dir_xy ─────────────────────────────────────
        if target_dir_xy is None:
            t_xy = np.array([1.0, 0.0], dtype=np.float64)
        else:
            t = np.asarray(target_dir_xy, dtype=np.float64).reshape(-1)[:2]
            n = float(np.linalg.norm(t))
            t_xy = (t / n) if n > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
        target_angle = float(np.arctan2(t_xy[1], t_xy[0]))

        # ── 居中 ───────────────────────────────────────────────────
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        self.model_center = (lo + hi) / 2.0
        centered = verts - self.model_center

        # ── AABB yaw（与 Open3D 路径选向一致：AABB 长轴 -> target） ──────
        xy = centered[:, :2]
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        dx = x_max - x_min
        dy = y_max - y_min
        if abs(dx - dy) > 1e-6:
            main_angle = 0.0 if dx >= dy else (np.pi / 2.0)
        else:
            # 退化情形：PCA 主成分
            cov = np.cov((xy - xy.mean(axis=0)), rowvar=False)
            _, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, -1]
            main_angle = float(np.arctan2(principal[1], principal[0]))
        yaw_angle = target_angle - main_angle
        while yaw_angle > np.pi:
            yaw_angle -= 2.0 * np.pi
        while yaw_angle < -np.pi:
            yaw_angle += 2.0 * np.pi
        c, s = np.cos(yaw_angle), np.sin(yaw_angle)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        rotated = centered @ Rz.T

        # ── 缩放 ───────────────────────────────────────────────────
        lo = rotated.min(axis=0)
        hi = rotated.max(axis=0)
        max_extent = float((hi - lo).max())
        self.model_scale = (TARGET_EXTENT / max_extent) if max_extent > 1e-9 else 1.0
        transformed = rotated * self.model_scale
        self.vertices = transformed.tolist()
        self.bounds_after = {
            'min': transformed.min(axis=0),
            'max': transformed.max(axis=0),
            'center': (transformed.min(axis=0) + transformed.max(axis=0)) / 2.0,
            'size': transformed.max(axis=0) - transformed.min(axis=0),
        }
        self.ground_normal = np.array([0.0, 0.0, 1.0])
        self.yaw_axis = np.array([t_xy[0], t_xy[1]], dtype=np.float64)
        self.yaw_angle_deg = float(np.degrees(yaw_angle))
        self.normalization_report = {
            "status": "fallback_normalized",
            "fallback_used": True,
            "reason": str(reason),
            "normalizer_source": "fallback",
            "ground_inliers": 0,
            "vertex_count": int(len(transformed)),
            "inlier_ratio": 0.0,
            "ground_axis_alignment_before": 0.0,
            "model_scale": float(self.model_scale),
            "yaw_angle_deg": float(self.yaw_angle_deg),
        }
        print(f"[规范化 fallback] yaw -> ({t_xy[0]:+.3f}, {t_xy[1]:+.3f}) Δ={self.yaw_angle_deg:+.2f}°")

    # ------------------------------------------------------------------
    # 对外接口（保留旧版 API）
    # ------------------------------------------------------------------
    def get_transformed_vertices(self):
        """获取规范化后的顶点（与 self.vertices 等价；保留兼容旧调用方）。"""
        return list(self.vertices)

    def get_bounding_box(self):
        """获取模型规范化后的包围盒。"""
        if self.bounds_after is None:
            return None
        return self.bounds_after

    def get_normalization_report(self):
        """获取本次规范化报告，供 UI 做导入姿态验收与错误提示。"""
        return dict(self.normalization_report)


# ----------------------------------------------------------------------
# 几何工具函数
# ----------------------------------------------------------------------
def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """求把单位向量 v_from 旋转到 v_to 的最小旋转矩阵（Rodrigues 形式）。

    v_from / v_to 必须已是单位向量；若 v_from 与 -v_to 几乎重合，会返回
    一个绕任意垂直轴的 180° 旋转矩阵（处理退化情形）。
    """
    v_from = np.asarray(v_from, dtype=np.float64)
    v_to = np.asarray(v_to, dtype=np.float64)
    v_from /= max(np.linalg.norm(v_from), 1e-12)
    v_to /= max(np.linalg.norm(v_to), 1e-12)

    cross = np.cross(v_from, v_to)
    dot = float(np.dot(v_from, v_to))

    # 退化：v_from ≈ v_to，直接返回单位阵
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)

    # 退化：v_from ≈ -v_to，找一个与 v_from 垂直的轴，绕它旋转 180°
    if dot < -1.0 + 1e-9:
        # 任意选一个不与 v_from 平行的轴
        if abs(v_from[0]) < 0.9:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, v_from) * v_from
        axis /= max(np.linalg.norm(axis), 1e-12)
        return _rotation_axis_angle(axis, np.pi)

    # 通用情形：Rodrigues 旋转公式
    s = np.linalg.norm(cross)
    K = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - dot) / (s * s))
    return R


def _rotation_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """绕单位轴 axis 旋转 angle 弧度的旋转矩阵（Rodrigues）。"""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-12)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ], dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    """绕 +Z 轴旋转 angle 弧度的旋转矩阵。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
