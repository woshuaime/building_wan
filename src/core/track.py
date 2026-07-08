import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# 轨道系统模块（纯净状态）
# ═══════════════════════════════════════════════════════════════════════
# TRACK_YAW_OFFSET_DEG 当前设定为 0.0（纯净状态）：
#   橙红点（get_intersection_point）回到原生 (radius, 0, 0)，不做任何轨道旋转。
#
# 导入姿态方案：
#   OBJLoader 先输出 MeshLab 语义坐标（顶面为 +Z），UI 层再把 +Z 映射到软件 +X。
#   因此四轨拍摄出发点固定在 +X，灰色虚拟平面为 YZ 面，模型底面与之平行。
#   MESHLAB_FRONT_AXIS 只影响 UI 初始观察相机，不影响模型本体朝向。
#
# 坐标系约定：
#   软件拍摄坐标中，+X 是模型顶面/相机出发方向，YZ 是底面参考平面。
TRACK_YAW_OFFSET_DEG = 0.0


def _apply_yaw_offset(points, angle_deg=None):
    """对若干 3D 点施加历史轨道 yaw 补偿。

    几何含义：
        这是旧轨道偏转接口，当前 TRACK_YAW_OFFSET_DEG = 0，不参与导入姿态。
        软件当前模型顶面方向是 +X，底面参考平面是 YZ；不要把这里的 Y 轴
        旋转理解为模型高度轴。

    输入：
        points:   (N, 3) 或 (3,) 的 ndarray/float 序列
        angle_deg: 可选，要旋转的角度（度）。
                   - None（默认）→ 使用模块常量 TRACK_YAW_OFFSET_DEG
                   - 显式数值     → 用该值覆盖（例如 -TRACK_YAW_OFFSET_DEG 做反向补偿）

    输出：
        与输入同形状的 ndarray。非零偏转时按旧实现绕 Y 轴旋转。

    数学：
        设 θ = angle_deg（度）。
        旧实现的绕 Y 轴右手旋转矩阵：
            R_y(θ) = | cosθ   0   sinθ |
                     |  0     1    0   |
                     | -sinθ  0   cosθ |
        作用到列向量 (x, y, z) 上：
            x' =  cosθ * x + sinθ * z
            y' =  y
            z' = -sinθ * x + cosθ * z
        当 θ = 90° 时：cosθ=0, sinθ=1 → x' = z, z' = -x（顺时针）
        当 θ = -90° 时：cosθ=0, sinθ=-1 → x' = -z, z' = x（逆时针）
    """
    if angle_deg is None:
        angle_deg = TRACK_YAW_OFFSET_DEG
    theta = np.radians(float(angle_deg))
    if abs(theta) < 1e-9:
        # 0° 补偿时直接返回副本，避免无意义计算
        arr = np.asarray(points, dtype=np.float64)
        return arr.copy() if hasattr(arr, "copy") else np.array(arr, dtype=np.float64)

    c, s = np.cos(theta), np.sin(theta)
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        x, y, z = arr[0], arr[1], arr[2]
        return np.array([c * x + s * z, y, -s * x + c * z], dtype=np.float64)
    out = np.empty_like(arr)
    out[:, 0] = c * arr[:, 0] + s * arr[:, 2]
    out[:, 1] = arr[:, 1]
    out[:, 2] = -s * arr[:, 0] + c * arr[:, 2]
    return out


def _max_in_plane_radius(rel, plane):
    """顶点相对中心 rel (N,3) 在轨道平面上的最大投影半径。"""
    if plane == "horizontal":
        return float(np.max(np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)))
    if plane == "vertical":
        return float(np.max(np.sqrt(rel[:, 0]**2 + rel[:, 2]**2)))
    if plane == "diag_left":
        u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        v = np.array([0.0, 0.0, 1.0])
        proj = np.column_stack([rel @ u, rel @ v])
        return float(np.max(np.sqrt(proj[:, 0]**2 + proj[:, 1]**2)))
    if plane == "diag_right":
        u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        v = np.array([0.0, 0.0, 1.0])
        proj = np.column_stack([rel @ u, rel @ v])
        return float(np.max(np.sqrt(proj[:, 0]**2 + proj[:, 1]**2)))
    return 0.0


class CircularTrack:
    """单条圆形轨道（在过模型中心的平面上）"""

    def __init__(self, name, plane, color, radius=8.0):
        self.name = name
        self.plane = plane
        self.color = color
        self.radius = float(radius)
        self.center = np.zeros(3, dtype=np.float64)

    def set_center(self, center):
        self.center = np.array(center, dtype=np.float64).copy()

    def set_radius(self, radius):
        self.radius = max(0.5, float(radius))

    def sample_circle(self, num=128):
        """用于绘制的折线点 (N,3)

        采样流程：
            1) 在本轨道的"几何平面"内，用三角函数生成基础圆点 (mx, my, mz)。
            2) 拼成 (N, 3) 后，调用 _apply_yaw_offset 在 PyVista X-Z 水平面上
               绕 Y 轴统一旋转 TRACK_YAW_OFFSET_DEG，让 4 条圆环整体偏转。
               模型本身完全不动。
        """
        t = np.linspace(0, 2 * np.pi, num, endpoint=False)
        c = self.center
        r = self.radius
        if self.plane == "horizontal":
            mx = c[0] + r * np.cos(t)
            my = c[1] + r * np.sin(t)
            mz = np.full_like(t, c[2])
        elif self.plane == "vertical":
            mx = c[0] + r * np.cos(t)
            my = np.full_like(t, c[1])
            mz = c[2] + r * np.sin(t)
        elif self.plane == "diag_left":
            # 基圆：竖直 xz 平面（y=0），绕 X 轴旋转 +45°
            # 旋转轴 = LookAt 轴（模型中心→初始相机位置）= X 轴
            # 旋转后：所有轨道均过 (r,0,0)（初始相机点/橙色点）
            ct, st = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            base_circle = np.stack([r * np.cos(t), np.zeros_like(t), r * np.sin(t)], axis=1)
            rotated = (Rx @ base_circle.T).T
            mx = c[0] + rotated[:, 0]
            my = c[1] + rotated[:, 1]
            mz = c[2] + rotated[:, 2]
        elif self.plane == "diag_right":
            # 基圆：竖直 xz 平面，绕 X 轴旋转 -45°
            ct, st = np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            base_circle = np.stack([r * np.cos(t), np.zeros_like(t), r * np.sin(t)], axis=1)
            rotated = (Rx @ base_circle.T).T
            mx = c[0] + rotated[:, 0]
            my = c[1] + rotated[:, 1]
            mz = c[2] + rotated[:, 2]
        else:
            mx = my = mz = t * 0

        # 在所有几何计算完成之后，施加统一的水平 Y 轴旋转补偿，
        # 使 4 条圆环在 UI 中整体打转到模型正脸一侧。
        return _apply_yaw_offset(np.column_stack([mx, my, mz]))

    def _point_at_angle(self, ti):
        """返回角度 ti 对应的轨道采样点（应用了水平 Y 轴旋转补偿后）。"""
        c = self.center
        r = self.radius
        if self.plane == "horizontal":
            p = np.array([c[0] + r * np.cos(ti), c[1] + r * np.sin(ti), c[2]])
        elif self.plane == "vertical":
            p = np.array([c[0] + r * np.cos(ti), c[1], c[2] + r * np.sin(ti)])
        elif self.plane == "diag_left":
            # 绕 X 轴旋转 +45°（基圆 = 竖直 xz 平面，y=0）
            ct, st = np.cos(np.pi / 4.0), np.sin(np.pi / 4.0)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            base = np.array([r * np.cos(ti), 0.0, r * np.sin(ti)])
            p = c + Rx @ base
        elif self.plane == "diag_right":
            # 绕 X 轴旋转 -45°
            ct, st = np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            base = np.array([r * np.cos(ti), 0.0, r * np.sin(ti)])
            p = c + Rx @ base
        else:
            p = c.copy()

        # 在所有几何计算完成之后，施加统一的水平 Y 轴旋转补偿，
        # 使相机采样点与 sample_circle 圆环上的点保持严格一致。
        return _apply_yaw_offset(p)

    def _best_start_angle(self, align_dir):
        """使轨道上离观察者方向最近的点作为起点（与 align_dir 点积最大）"""
        if align_dir is None:
            return 0.0
        d = np.asarray(align_dir, dtype=np.float64).reshape(3)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return 0.0
        d = d / n
        fine = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        best_a, best_s = 0.0, -1e30
        for ti in fine:
            p = self._point_at_angle(ti)
            s = float(np.dot(p - self.center, d))
            if s > best_s:
                best_s = s
                best_a = float(ti)
        return best_a

    def get_camera_positions(self, num_positions=16, align_dir=None):
        """沿圆均匀分布；align_dir 为从场景中心指向观察者的单位方向时，起点靠近用户一侧"""
        t0 = self._best_start_angle(align_dir)
        t = t0 + np.linspace(0, 2 * np.pi, num_positions, endpoint=False)
        return [self._point_at_angle(float(ti)) for ti in t]

    def get_circle_points_for_rendering(self, num_points=64):
        """获取用于渲染的完整圆环点列表"""
        return self.sample_circle(num_points)

    def get_bounds(self):
        """获取轨道上所有采样点的包围盒 (xmin,xmax,ymin,ymax,zmin,zmax)"""
        pts = self.sample_circle(128)
        return (float(pts[:, 0].min()), float(pts[:, 0].max()),
                float(pts[:, 1].min()), float(pts[:, 1].max()),
                float(pts[:, 2].min()), float(pts[:, 2].max()))


class TrackSystem:
    """四条圆形轨道：水平、竖直、左斜、右斜，共面过模型中心"""

    TRACK_CONFIGS = [
        {"name": "水平圆轨道", "plane": "horizontal", "color": (1.0, 0.3, 0.3)},   # 红色
        {"name": "竖直圆轨道", "plane": "vertical", "color": (0.3, 1.0, 0.3)},     # 绿色
        {"name": "左斜圆轨道", "plane": "diag_left", "color": (0.3, 0.3, 1.0)},   # 蓝色
        {"name": "右斜圆轨道", "plane": "diag_right", "color": (1.0, 0.7, 0.3)},  # 橙色
    ]

    def __init__(self):
        self.center = np.zeros(3, dtype=np.float64)
        self.last_virtual_anchor = self.center.copy()
        self.default_radius = 8.0
        self.camera_distance = 10.0
        self.tracks = []
        for cfg in self.TRACK_CONFIGS:
            self.tracks.append(
                CircularTrack(cfg["name"], cfg["plane"], cfg["color"], self.default_radius)
            )

    def set_center(self, center):
        self.center = np.array(center, dtype=np.float64).copy()
        for t in self.tracks:
            t.set_center(self.center)

    def set_camera_distance(self, distance):
        """设置相机到中心的真实物理距离

        同时更新所有轨道的半径为这个距离。

        Args:
            distance: 相机到模型中心的距离
        """
        self.camera_distance = max(0.1, float(distance))
        for t in self.tracks:
            t.set_radius(self.camera_distance)

    def get_camera_distance(self):
        """获取相机到中心的真实物理距离"""
        return self.camera_distance

    def get_all_track_points(self):
        """获取四条轨道的数据，用于渲染

        Returns:
            list: 四条轨道的渲染数据，每条包含：
                - 'name': 轨道名称
                - 'color': 轨道颜色 (R, G, B)
                - 'circle_points': 用于渲染圆环的顶点列表 (N, 3)
                - 'plane': 轨道所在平面
        """
        return [
            {
                'name': t.name,
                'color': t.color,
                'circle_points': t.get_circle_points_for_rendering(64),
                'plane': t.plane
            }
            for t in self.tracks
        ]

    def get_track_colors(self):
        """获取四条轨道的颜色"""
        return [t.color for t in self.tracks]

    def set_default_radius_from_extent(self, max_extent, margin=1.25):
        """根据轴对齐包围盒最大边长设置半径（粗估）"""
        ext = float(max_extent)
        r = 0.5 * ext * margin
        r = max(r, ext * 0.26)
        self.default_radius = r
        for t in self.tracks:
            t.set_radius(r)

    def set_radii_from_vertices(self, vertices, center, margin=1.14):
        """用统一的外接球半径（所有轨道相同）"""
        rel = np.asarray(vertices, dtype=np.float64) - np.asarray(center, dtype=np.float64).reshape(1, 3)
        if rel.size == 0:
            return
        sphere_r = float(np.max(np.linalg.norm(rel, axis=1))) * float(margin)
        sphere_r = max(sphere_r, 1e-4)
        for t in self.tracks:
            t.set_radius(sphere_r)
        self.default_radius = sphere_r

    def set_track_radius(self, track_index, radius):
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].set_radius(radius)

    def get_track(self, index):
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None

    def compute_horizontal_vertical_anchor(self, view_dir):
        """计算水平圆与竖直圆的最近交点"""
        if len(self.tracks) < 2:
            return self.center.copy()
        h, v = self.tracks[0], self.tracks[1]
        ph = h.sample_circle(120)
        pv = v.sample_circle(120)
        best_d = 1e30
        best_pair = (ph[0], pv[0])
        for a in ph:
            for b in pv:
                d = float(np.linalg.norm(a - b))
                if d < best_d:
                    best_d = d
                    best_pair = (a, b)
        mid = (best_pair[0] + best_pair[1]) * 0.5
        alt = self.center * 2.0 - mid
        vd = np.asarray(view_dir, dtype=np.float64).reshape(3)
        vn = np.linalg.norm(vd)
        if vn < 1e-9:
            return mid
        vd = vd / vn
        s_mid = float(np.dot(mid - self.center, vd))
        s_alt = float(np.dot(alt - self.center, vd))
        return mid if s_mid >= s_alt else alt

    def get_view_direction_to_camera(self, elev_deg, azim_deg):
        """从场景中心指向观察者的方向"""
        el = np.radians(float(elev_deg))
        az = np.radians(float(azim_deg))
        xe = np.cos(az) * np.cos(el)
        ye = np.sin(az) * np.cos(el)
        ze = np.sin(el)
        return np.array([xe, ye, ze], dtype=np.float64)

    def get_all_camera_positions(self, elev_deg=22.0, azim_deg=135.0, num_positions=16, dome_only=False):
        """获取所有轨道上的相机位置

        Args:
            elev_deg: 仰角（度）
            azim_deg: 方位角（度）
            num_positions: 每条轨道的相机数量（每轨精确产生 num_positions 个位置）
            dome_only: 已废弃，保留参数仅避免调用方报错

        Returns:
            list: 每条轨道的相机位置数据，每轨精确包含 num_positions 个位置
        """
        view_dir = self.get_view_direction_to_camera(elev_deg, azim_deg)
        self.last_virtual_anchor = self.compute_horizontal_vertical_anchor(view_dir)

        for t in self.tracks:
            t.set_radius(self.camera_distance)

        all_positions = []
        for t in self.tracks:
            t0 = t._best_start_angle(view_dir)
            t_angles = t0 + np.linspace(0, 2 * np.pi, num_positions, endpoint=False)
            positions = [t._point_at_angle(float(ti)) for ti in t_angles]
            all_positions.append(
                {"track_name": t.name, "positions": positions}
            )
        return all_positions

    def get_virtual_camera_anchor(self, elev_deg, azim_deg):
        """获取轨道交界处的虚拟相机锚点"""
        view_dir = self.get_view_direction_to_camera(elev_deg, azim_deg)
        anchor = self.compute_horizontal_vertical_anchor(view_dir)
        self.last_virtual_anchor = np.asarray(anchor, dtype=np.float64).copy()
        return self.last_virtual_anchor

    def get_intersection_point(self):
        """获取四条轨道的公共交点，即实际拍摄序列第一帧的确切坐标。

        计算逻辑（与 get_all_camera_positions 完全一致）：
        - 用 view_dir 搜索所有轨道上离观察者最近的两条轨道（水平+竖直）
        - 取其最近一对点的中点作为交点候选
        - 返回视线方向上最靠近观察者的那个候选点

        这样确保视锥体、橙色圆点、实际拍摄第一帧三者 100% 坐标相同。

        ── 反向补偿（关键！避免交点跟着轨道一起转走） ─────────────────────
        由于 sample_circle / _point_at_angle 已经在最终输出上施加了
        TRACK_YAW_OFFSET_DEG 的水平旋转，compute_horizontal_vertical_anchor
        拿到的最近点对、进而返回的 anchor 坐标也是被旋转过的——也就是说，
        交点（橙红点）会跟着 4 条圆环一起"原地打转"，与模型之间没有任何
        相对位移，初始拍摄第一帧依然对着模型的"侧面/山墙"。

        为了让相机出发点死死锚定在**未旋转前的原生机位**（即 OBJ 加载后
        原始 (r, 0, 0) 位置，正对模型正脸），这里必须把 anchor 反向旋转
        回去：传入 -TRACK_YAW_OFFSET_DEG，让水平面旋转量刚好抵消。

        直观图解（TRACK_YAW_OFFSET_DEG = 90°）：
            - 4 条圆环整体顺时针转 90° → 圆环偏移
            - 橙红点本来也跟转 → 现在被反向补偿 -90° 拧回原位
            - 结果：圆环视觉上偏了，但初始拍摄机位永远在 (r, 0, 0)
        """
        view_dir = np.array([1.0, 0.0, 0.0])  # X+ 方向，与硬编码 (r,0,0) 对齐
        anchor = self.compute_horizontal_vertical_anchor(view_dir)

        # 反向补偿：沿旧轨道 yaw 轴旋转 -TRACK_YAW_OFFSET_DEG，
        # 让它回到"未旋转轨道系统"时的原生位置 (r, 0, 0)。
        anchor = _apply_yaw_offset(anchor, angle_deg=-TRACK_YAW_OFFSET_DEG)

        return np.asarray(anchor, dtype=np.float64).copy()

    def get_back_intersection_point(self):
        """获取四条轨道的背侧公共交点（与 get_intersection_point 相反方向）。"""
        view_dir = np.array([-1.0, 0.0, 0.0])  # X- 方向
        anchor = self.compute_horizontal_vertical_anchor(view_dir)
        return np.asarray(anchor, dtype=np.float64).copy()

    def snapshot_radii(self):
        return [t.radius for t in self.tracks]

    def restore_radii(self, radii):
        for i, r in enumerate(radii):
            if i < len(self.tracks):
                self.tracks[i].set_radius(r)

    def get_all_camera_positions_flat(self, elev_deg=22.0, azim_deg=135.0):
        """返回扁平的64个相机位置列表"""
        all_data = self.get_all_camera_positions(elev_deg, azim_deg)
        flat = []
        for item in all_data:
            c = item["track_name"]
            for i, track in enumerate(self.tracks):
                if track.name == c:
                    col = track.color
                    break
            else:
                col = (0.8, 0.8, 0.8)
            for pos in item["positions"]:
                flat.append((np.asarray(pos, dtype=np.float64), col))
        return flat

    def compute_coverage(self, elev_deg=22.0, azim_deg=135.0, num_samples=2000):
        """估算相机覆盖率"""
        positions = self.get_all_camera_positions_flat(elev_deg, azim_deg)
        if not positions:
            return 0.0

        phi = np.pi * (3.0 - np.sqrt(5.0))
        samples = np.zeros((num_samples, 3), dtype=np.float64)
        for i in range(num_samples):
            y = 1.0 - (i / float(num_samples - 1)) * 2.0
            r = np.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            samples[i] = np.array([np.cos(theta) * r, y, np.sin(theta) * r], dtype=np.float64)

        covered = 0
        half_fov_rad = np.radians(30.0)
        cos_half = np.cos(half_fov_rad)

        for s in samples:
            for pos, _ in positions:
                to_sample = s - pos
                dist = np.linalg.norm(to_sample)
                if dist < 1e-9:
                    continue
                view_dir = -pos / np.linalg.norm(pos)
                sample_dir = to_sample / dist
                if float(np.dot(view_dir, sample_dir)) >= cos_half:
                    covered += 1
                    break

        return covered / num_samples
