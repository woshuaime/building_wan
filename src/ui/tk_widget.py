"""
3D Scanner - 基于 PyVista + PySide6 的渲染引擎
===============================================
重构改动（相比 Tkinter 版）：
1. UI 框架：PySide6 替代 Tkinter，配合 pyvistaqt.QtInteractor 嵌入 3D 视口。
2. 布局：QHBoxLayout 将左侧 3D 视口与右侧控制面板水平分栏。
3. 透视投影：FOV=55°, Near=0.01, Far=2000，完全支持近大远小。
4. 3D 视口：pyvistaqt.QtInteractor 作为 Qt Widget 直接嵌入布局。
5. 渲染：Qt 事件循环驱动，无需手动管理定时器。
"""
import os
import sys
import json
import datetime
import numpy as np
import pyvista as pv
from PIL import Image
from pyvistaqt import QtInteractor
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QRadioButton, QButtonGroup, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QStyleFactory,
)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, QObject, Signal
from PySide6.QtGui import QFont, QPalette, QColor

from core.dataset_loader import PREALIGNED_TRIPLET, SHARED_TRIPLET, load_dataset_triplet
from core.obj_loader import OBJLoader
from core.track import TrackSystem
from utils.screenshot import ScreenshotManager
from utils.model_folder import allocate_next_model_folder


class InteractionMode:
    """交互模式枚举 - 三大模式互斥"""
    VIEWING = "viewing"           # 默认检视模式：左键拖拽旋转相机视角
    MODEL_ADJUSTING = "model"     # 调整模型模式：左键拖拽旋转模型自身
    TRACK_ADJUSTING = "track"     # 调整轨道模式：左键点轨道圆环拖拽调半径


# ── 轨道配色 ───────────────────────────────────────────────────────────
TRACK_COLORS = {
    0: (0.90, 0.22, 0.21),   # 红色 - 水平
    1: (0.12, 0.53, 0.90),   # 蓝色 - 竖直
    2: (0.26, 0.63, 0.28),   # 绿色 - 左斜
    3: (0.98, 0.55, 0.00),   # 橙色 - 右斜
}
TRACK_NAMES = ["水平", "竖直", "左斜", "右斜"]

# ── 导入姿态 / 观察参数 ────────────────────────────────────────────────
#
# 模型本体朝向固定由 _obj_to_pv_coords() 落地：
#     OBJLoader +Z 顶面方向 -> 软件 +X 四轨拍摄出发点方向。
#     OBJLoader XY 底面     -> 软件 YZ 灰色虚拟平面。
#
# 导入后的初始观察相机只影响屏幕显示，不改变模型姿态或拍摄轨道。
APP_MODEL_TOP_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
APP_GRAY_PLANE_NORMAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)
BOUNDARY_CAP_SCORE_THRESHOLD = 0.95
BOUNDARY_CAP_MIN_AREA_RATIO = 0.03
BOUNDARY_CAP_CANDIDATE_ALIGNMENT = 0.80

# ── 渲染参数 ──────────────────────────────────────────────────────────
CAMERA_FOV         = 45.0   # 视场角（度）——标准人眼/镜头视角，避免广角畸变与长焦平板感
CAMERA_NEAR        = 0.01   # 近裁剪面（极小，近距离不穿模）
CAMERA_FAR         = 1000.0 # 远裁剪面——轨道半径增大后，确保模型远端不被切掉
CAPTURE_SIZE_PX    = 1024  # 拍照输出分辨率（优化2：1024提升速度；可选改为2048追求工业高清）
INITIAL_ELEV       = 22.0   # 初始俯仰角
INITIAL_AZIM       = 135.0  # 初始方位角
ELEV_MIN           = -88.0  # 俯仰角下限
ELEV_MAX           = 88.0   # 俯仰角上限
INITIAL_OVERVIEW_ELEV = 24.0
INITIAL_OVERVIEW_AZIM = 128.0
INITIAL_OVERVIEW_DISTANCE_FACTOR = 3.0
MODEL_FILL_RATIO   = 0.80   # 模型占视图的比例
MAX_TRIANGLES      = 80000  # 三角面数上限（超采后自动抽稀）
ORBIT_MARGIN       = 1.8    # 轨道半径 = 模型最大外接球半径 × 此系数（FOV 缩小后轨道自动推远，此系数进一步微调画面占比）


# ═══════════════════════════════════════════════════════════════════════
#  PyVista 3D 视口（基于 pyvistaqt.QtInteractor）
# ═══════════════════════════════════════════════════════════════════════

class PyVistaView(QtInteractor):
    """
    PyVista 3D 渲染视口，继承自 pyvistaqt.QtInteractor（本身是 QWidget）。
    嵌入 PySide6 布局，无需手动管理窗口嵌入逻辑。
    """

    def __init__(self, main_window):
        self._main_window = main_window
        self._started = False

        # 数据
        self._mesh        = None   # pv.PolyData，当前模型（底层顶点数据）
        self._faces       = []     # 原始面数据（OBJ faces 列表，供 MODEL_ADJUSTING 重建）
        self._verts_base  = None   # 模型原始顶点（用于 MODEL_ADJUSTING 旋转）
        self._track_sys   = TrackSystem()  # 轨道系统

        # 渲染节流
        self._render_queued  = False
        self._redraw_pending = False

        # 相机极坐标（必须在 _init_plotter 之前初始化，_get_camera_position 依赖它们）
        self._elev = INITIAL_ELEV
        self._azim = INITIAL_AZIM

        super().__init__()

        self._init_plotter()

        # ── 交互状态（由 MainWindow._set_interaction_mode 同步写入）──────────
        self._mode           = InteractionMode.VIEWING
        self._active_track   = None
        self._model_rot_elev = 0.0
        self._model_rot_azim = 0.0
        self._adjust_last_xy = None   # MODEL_ADJUSTING 模式：鼠标拖拽起点（像素）
        self._model_dragging = False
        # ── 模型 Actor 追踪（仅用于当前 mesh 的显示引用）──────────────────────
        self._model_actor            = None    # 当前模型的 Actor 引用（pickable=True）
        self._model_transform_applied = False  # 标记：视觉姿态是否已固化到顶点
        self._model_center_lock_timer = QTimer(self)
        self._model_center_lock_timer.setInterval(16)
        self._model_center_lock_timer.timeout.connect(self._lock_model_actor_center)
        self._last_import_pose_report = ""
        self._loader_normalization_report = {}
        self._dataset_triplet = None
        self._dataset_meshes = {}
        self._dataset_mode = None
        self._dataset_plotters = {}
        self._dataset_parallel_scale = None
        self._capture_subject = "cad"

        # ── 垂直基准面（UI 墙，X 轴脚底板对齐）──────────────────────────────
        self._ground_actor = None   # 垂直半透明圆盘 actor
        self._bottom_x     = None   # 模型 X 轴最低点（脚底板，绘制 UI 墙的参考面）
        self._orbit_radius = 8.0    # 轨道半径缓存（由 compute_orbit_radius 设置）

    def _init_plotter(self):
        """初始化 PyVista Plotter 并设置相机参数。"""
        # QtInteractor 本身（继承链上）就是 Plotter，用 self 直接引用
        # 注意：_init_plotter() 在 super().__init__() 之后调用，此时 Plotter 已就绪
        self.setMouseTracking(True)
        self.hide_axes()
        self.enable_terrain_style()

        # ── 相机参数：透视投影 + 标准FOV(45°) + Near=0.01 ─────────────────
        cam = self.camera
        cam.focal_point    = (0.0, 0.0, 0.0)
        cam.position       = self._get_camera_position()
        cam.up             = (0.0, 0.0, 1.0)
        cam.view_angle     = CAMERA_FOV
        cam.clipping_range = (CAMERA_NEAR, CAMERA_FAR)

        # 注册默认视图旋转。MODEL_ADJUSTING 交给 PyVista 原生 TrackballActor。
        self.iren.add_observer("MouseMoveEvent", self._on_mouse_move)

        # 初始渲染
        self._render_scene()
        self._started = True

    def _on_mouse_move(self, obj, event):
        """鼠标移动：左键拖拽。

        - MODEL_ADJUSTING：由 PyVista TrackballActor 原生处理模型拖拽
        - VIEWING / TRACK_ADJUSTING：旋转相机视角
        """
        iren = obj
        xy = iren.GetEventPosition()

        if getattr(self, "_mode", InteractionMode.VIEWING) == InteractionMode.MODEL_ADJUSTING:
            QTimer.singleShot(0, self._lock_model_actor_center)
            return

        if not (QApplication.mouseButtons() & Qt.LeftButton):
            self._last_xy = None
            return

        # ── 调整轨道 / 默认检视：旋转相机视角 ──────────────────────────
        if self._last_xy is not None:
            dx = xy[0] - self._last_xy[0]
            dy = xy[1] - self._last_xy[1]
            self.orbit_camera(dx * 0.5, -dy * 0.5)
        self._last_xy = (float(xy[0]), float(xy[1]))

    _last_xy = None

    def _reset_model_actor_transform(self):
        """确保模型 actor 不再叠加任何临时姿态。"""
        actor = self._model_actor
        if actor is None:
            return
        try:
            actor.SetOrientation(0.0, 0.0, 0.0)
            actor.SetPosition(0.0, 0.0, 0.0)
            actor.SetScale(1.0, 1.0, 1.0)
        except Exception:
            pass
        try:
            actor.user_matrix = np.eye(4, dtype=np.float64)
        except Exception:
            pass

    def _get_track_center(self):
        """四条轨道的共同中心。"""
        center = getattr(self._track_sys, "center", None)
        if center is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(center, dtype=np.float64).reshape(3)

    def _notify_model_mesh_changed(self):
        if self._mesh is None:
            return
        try:
            self._mesh.GetPoints().Modified()
        except Exception:
            pass
        try:
            self._mesh.Modified()
        except Exception:
            pass
        if self._model_actor is not None:
            try:
                self._model_actor.GetMapper().Modified()
            except Exception:
                pass

    def _center_model_mesh_on_tracks(self):
        """把 mesh 几何中心固定到四条轨道中心。"""
        if self._mesh is None:
            return
        target = self._get_track_center()
        current = np.asarray(self._mesh.center, dtype=np.float64)
        delta = target - current
        if np.linalg.norm(delta) <= 1e-7:
            self._bottom_x = self._mesh.bounds[0]
            return

        self._mesh.translate(delta, inplace=True)
        if self._verts_base is not None:
            self._verts_base = np.asarray(self._verts_base, dtype=np.float64) + delta
        self._bottom_x = self._mesh.bounds[0]
        self._notify_model_mesh_changed()

    def _validate_import_pose(self):
        """导入后姿态验收：中心、底面平行、顶面朝四轨出发点。"""
        if self._mesh is None or self._track_sys is None:
            return False

        center = self.get_model_center()
        track_center = self._get_track_center()
        center_err = float(np.linalg.norm(center - track_center))

        # _obj_to_pv_coords 将 OBJLoader 的 +Z 顶面方向固定映射为软件 +X。
        imported_top_dir = _obj_to_pv_coords(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64))
        imported_top_dir = imported_top_dir[1] - imported_top_dir[0]
        imported_top_dir /= max(float(np.linalg.norm(imported_top_dir)), 1e-12)

        anchor = self._track_sys.get_intersection_point()
        camera_start_dir = np.asarray(anchor, dtype=np.float64) - track_center
        camera_start_dir /= max(float(np.linalg.norm(camera_start_dir)), 1e-12)

        gray_plane_normal = APP_GRAY_PLANE_NORMAL
        top_axis_error = float(np.linalg.norm(imported_top_dir - APP_MODEL_TOP_AXIS))
        top_alignment = float(np.dot(imported_top_dir, camera_start_dir))
        plane_parallel = float(abs(np.dot(imported_top_dir, gray_plane_normal)))
        mesh_parallel, top_faces, bottom_faces, boundary_state = self._validate_boundary_faces_parallel()
        boundary_hard_fail = boundary_state in {"bottom_fail", "complete_fail"}
        loader_report = dict(getattr(self, "_loader_normalization_report", {}) or {})
        loader_status = loader_report.get("status", "unknown")
        fallback_used = bool(loader_report.get("fallback_used", True))
        ground_axis_alignment = float(loader_report.get("ground_axis_alignment_before", 0.0) or 0.0)
        ground_source = loader_report.get("normalizer_source", "n/a")
        boundary_bottom_trusted = boundary_state in {"complete_ok", "bottom_ok"}
        normalization_trusted = (
            loader_status == "ground_normalized"
            and not fallback_used
            and ground_axis_alignment >= 0.95
        )
        if ground_source == "mesh_bottom_boundary" and boundary_bottom_trusted:
            pose_trust = "strong_bottom"
        elif ground_source == "mesh_bottom_boundary":
            pose_trust = "loader_bottom"
        elif boundary_bottom_trusted:
            pose_trust = "boundary_confirmed"
        else:
            pose_trust = "loader_only"

        ok = (
            center_err <= 1e-5
            and top_axis_error <= 1e-6
            and top_alignment >= 0.999
            and plane_parallel >= 0.999
            and normalization_trusted
            and not boundary_hard_fail
        )
        self._last_import_pose_report = (
            f"center_err={center_err:.6g} | "
            f"top_axis_err={top_axis_error:.6g} | "
            f"top_to_camera={top_alignment:.6f} | "
            f"bottom_parallel_gray={plane_parallel:.6f} | "
            f"loader_status={loader_status} | "
            f"fallback={int(fallback_used)} | "
            f"ground_source={ground_source} | "
            f"ground_inliers={loader_report.get('ground_inliers', 'n/a')} | "
            f"ground_axis={ground_axis_alignment:.6f} | "
            f"pose_trust={pose_trust} | "
            f"mesh_boundary_parallel={mesh_parallel:.6f} "
            f"(top_faces={top_faces}, bottom_faces={bottom_faces}, boundary_status={boundary_state})"
        )
        print(f"[导入姿态验收] {self._last_import_pose_report} | {'OK' if ok else 'FAIL'}")
        return ok

    def _clear_loaded_model_state(self):
        """导入失败时清理半加载状态，避免错误姿态继续用于拍摄。"""
        self._mesh = None
        self._faces = []
        self._verts_base = None
        self._model_actor = None
        self._bottom_x = None
        self._loader_normalization_report = {}
        self._dataset_triplet = None
        self._dataset_meshes = {}
        self._dataset_mode = None
        self._destroy_dataset_plotters()
        self._dataset_parallel_scale = None
        self._track_sys = TrackSystem()
        self._clear_scene()

    def _validate_boundary_faces_parallel(self):
        """从真实三角面片反查导入后的顶/底边界面是否平行灰色平面。

        OBJ buildings cut from a larger scene are often open meshes. Missing a cap
        face is not a pose error, so incomplete boundary evidence is reported but
        does not block import.
        """
        if self._mesh is None or not self._faces:
            return 1.0, 0, 0, "no_faces"

        pts = np.asarray(self._mesh.points, dtype=np.float64)
        if pts.size == 0:
            return 1.0, 0, 0, "no_points"

        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
        z_min = float(np.min(pts[:, 2]))
        z_max = float(np.max(pts[:, 2]))
        x_extent = max(x_max - x_min, 1e-9)
        yz_area2 = max(2.0 * (y_max - y_min) * (z_max - z_min), 1e-9)
        tol = max(x_extent * 0.03, 1e-5)

        top_scores = []
        bottom_scores = []
        for face in self._faces:
            for i0, i1, i2 in _face_to_tri_indices(face, len(pts)):
                p0, p1, p2 = pts[i0], pts[i1], pts[i2]
                normal = np.cross(p1 - p0, p2 - p0)
                area2 = float(np.linalg.norm(normal))
                if area2 <= 1e-12:
                    continue
                normal /= area2
                centroid_x = float((p0[0] + p1[0] + p2[0]) / 3.0)
                alignment = float(abs(np.dot(normal, APP_GRAY_PLANE_NORMAL)))
                weighted = alignment * area2
                if centroid_x >= x_max - tol:
                    top_scores.append((weighted, area2))
                if centroid_x <= x_min + tol:
                    bottom_scores.append((weighted, area2))

        def side_stats(items):
            if not items:
                return 0.0, 0.0, False
            cap_items = [(w, a) for w, a in items if (w / max(a, 1e-12)) >= BOUNDARY_CAP_CANDIDATE_ALIGNMENT]
            if not cap_items:
                return 0.0, 0.0, False
            weighted_sum = sum(w for w, _ in cap_items)
            area_sum = sum(a for _, a in cap_items)
            score = float(weighted_sum / max(area_sum, 1e-12))
            area_ratio = float(area_sum / yz_area2)
            trusted = area_ratio >= BOUNDARY_CAP_MIN_AREA_RATIO
            return score, area_ratio, trusted

        top_score, top_area_ratio, top_trusted = side_stats(top_scores)
        bottom_score, bottom_area_ratio, bottom_trusted = side_stats(bottom_scores)
        top_aligned = top_score >= BOUNDARY_CAP_SCORE_THRESHOLD
        bottom_aligned = bottom_score >= BOUNDARY_CAP_SCORE_THRESHOLD

        if bottom_trusted:
            if bottom_aligned:
                state = "complete_ok" if (top_trusted and top_aligned) else "bottom_ok"
                return bottom_score, len(top_scores), len(bottom_scores), state
            return bottom_score, len(top_scores), len(bottom_scores), "bottom_fail"

        if top_trusted:
            state = "partial_ok" if top_aligned else "partial_untrusted"
            return top_score, len(top_scores), len(bottom_scores), state

        if top_scores or bottom_scores:
            weak_score = max(top_score, bottom_score)
            return weak_score, len(top_scores), len(bottom_scores), "weak_boundary_evidence"
        return 1.0, 0, 0, "no_boundary_faces"

    def _actor_transformed_point(self, point):
        actor = self._model_actor
        if actor is None:
            return None
        vtk_mat = actor.GetMatrix()
        p = np.asarray(point, dtype=np.float64).reshape(3)
        return np.array(
            [
                vtk_mat.GetElement(0, 0) * p[0] + vtk_mat.GetElement(0, 1) * p[1] + vtk_mat.GetElement(0, 2) * p[2] + vtk_mat.GetElement(0, 3),
                vtk_mat.GetElement(1, 0) * p[0] + vtk_mat.GetElement(1, 1) * p[1] + vtk_mat.GetElement(1, 2) * p[2] + vtk_mat.GetElement(1, 3),
                vtk_mat.GetElement(2, 0) * p[0] + vtk_mat.GetElement(2, 1) * p[1] + vtk_mat.GetElement(2, 2) * p[2] + vtk_mat.GetElement(2, 3),
            ],
            dtype=np.float64,
        )

    def _lock_model_actor_center(self, force=False):
        """调整模型时只允许旋转，视觉中心始终锁在轨道中心。"""
        if self._model_actor is None or self._mesh is None:
            return
        if not force and getattr(self, "_mode", InteractionMode.VIEWING) != InteractionMode.MODEL_ADJUSTING:
            return

        target = self._get_track_center()
        visual_center = self._actor_transformed_point(target)
        if visual_center is None:
            return
        delta = target - visual_center
        if np.linalg.norm(delta) <= 1e-6:
            return

        try:
            pos = np.asarray(self._model_actor.GetPosition(), dtype=np.float64)
            self._model_actor.SetPosition(*(pos + delta))
            self.render()
        except Exception:
            pass

    def _get_camera_position(self):
        """根据当前 elev/azim 计算相机球面坐标。"""
        el = np.radians(self._elev)
        az = np.radians(self._azim)
        r  = self._get_camera_radius()
        x  = r * np.cos(el) * np.sin(az)
        y  = r * np.cos(el) * np.cos(az)
        z  = r * np.sin(el)
        return (float(x), float(y), float(z))

    def _get_camera_radius(self):
        """UI 相机球面半径：模型最大尺寸 × 1.5，与轨道物理半径完全一致。"""
        base = 8.0
        if self._mesh is not None:
            b = self._mesh.bounds
            x_len = b[1] - b[0]
            y_len = b[3] - b[2]
            z_len = b[5] - b[4]
            ext = max(x_len, y_len, z_len)
            base = max(8.0, ext * 3.5)
        return min(base, 30.0)

    def _snapshot_camera_state(self):
        """保存当前视口相机状态，用于重建场景后保持用户缩放/视角。"""
        cam = self.camera
        state = {
            "position": tuple(cam.position),
            "focal_point": tuple(cam.focal_point),
            "up": tuple(cam.up),
            "view_angle": float(cam.view_angle),
            "clipping_range": tuple(cam.clipping_range),
        }
        try:
            state["parallel_projection"] = bool(cam.GetParallelProjection())
            state["parallel_scale"] = float(cam.GetParallelScale())
        except Exception:
            pass
        return state

    def _restore_camera_state(self, state):
        """恢复 _snapshot_camera_state() 保存的相机状态。"""
        if not state:
            return
        cam = self.camera
        try:
            cam.position = state["position"]
            cam.focal_point = state["focal_point"]
            cam.up = state["up"]
            cam.view_angle = state["view_angle"]
            cam.clipping_range = state["clipping_range"]
            if "parallel_projection" in state:
                if state["parallel_projection"]:
                    cam.ParallelProjectionOn()
                else:
                    cam.ParallelProjectionOff()
            if "parallel_scale" in state:
                cam.SetParallelScale(state["parallel_scale"])
        except Exception:
            pass

    def _clear_scene(self):
        """清空 Plotter 所有对象。"""
        if False:   # QtInteractor 初始化后 Plotter 必定存在
            return
        try:
            self.clear()
        except Exception:
            pass

    def _render_scene(self):
        """完整重建 3D 场景：垂直基准面 + 轨道 + 模型 + 相机锚点。

        环境元素全部 pickable=False；模型 actor 只作为当前 mesh 的显示引用。
        """
        if False:
            return
        self._clear_scene()

        # ── 垂直基准面（UI 墙，YZ 平面，pickable=False）────────────────────────
        # direction=(1,0,0) 法线朝 X 轴 → YZ 垂直平面，center 的 X = bottom_x（脚底板）
        # reset_camera=False 防止大平面拉远相机，撑爆模型视野
        if self._mesh is not None and self._bottom_x is not None:
            plane_size = self._orbit_radius * 3.0
            vertical_ground = pv.Plane(
                center=(self._bottom_x, 0.0, 0.0),
                direction=tuple(float(x) for x in APP_GRAY_PLANE_NORMAL),
                i_size=plane_size,
                j_size=plane_size,
            )
            self._ground_actor = self.add_mesh(
                vertical_ground,
                color="lightgray",
                opacity=0.4,
                pickable=False,
                reset_camera=False,
                name="ui_vertical_ground",
            )

        # ── 轨道（建筑模式显示平行虚拟地面的水平环 + 原三条上半轨）────────
        if self._mesh is not None and self._track_sys is not None:
            building_preview = (
                self._capture_subject == "building"
                and not self.has_dataset()
            )
            building_points = (
                self._track_sys.get_building_track_points_for_rendering(256)
                if building_preview
                else None
            )
            for i, track in enumerate(self._track_sys.tracks):
                pts = building_points[i] if building_points is not None else track.sample_circle(256)
                color = TRACK_COLORS.get(i, (1.0, 1.0, 1.0))
                self.add_mesh(
                    pts,
                    color=color,
                    line_width=2.5 if i == self._active_track else 1.8,
                    opacity=0.85,
                    style="wireframe",
                    name=f"track_{i}",
                    pickable=False,
                )

        # ── 模型实体（pickable=True，调整模式下由 TrackballActor 原生旋转）──
        mesh_to_show = self._get_model_mesh_for_display()
        if mesh_to_show is not None:
            actor = self.add_mesh(
                mesh_to_show,
                color="#c8cdd2",
                style="surface",
                show_edges=False,
                smooth_shading=True,
                ambient=0.58,
                diffuse=0.38,
                specular=0.05,
                specular_power=30.0,
                name="model_mesh",
                pickable=True,
            )
            try:
                actor.SetOrigin(*self._get_track_center())
                actor.SetPosition(0.0, 0.0, 0.0)
            except Exception:
                pass
            self._model_actor = actor

        # ── 相机锚点（pickable=False）────────────────────────────────────
        if self._mesh is not None and self._track_sys is not None:
            anchor = self._track_sys.get_intersection_point()
            anchor_arr = np.array(anchor, dtype=np.float64).reshape(1, 3)
            self.add_mesh(
                pv.PolyData(anchor_arr),
                color="#FF6D00",
                point_size=14.0,
                render_points_as_spheres=True,
                name="cam_anchor",
                pickable=False,
            )
            self._add_frustum_visualization()

        # ── 计算对焦范围 ──────────────────────────────────────────────────
        focus_bounds = None
        if self._mesh is not None and self._track_sys is not None:
            tb = self._track_sys.tracks[0].get_bounds() if self._track_sys.tracks else None
            if tb is not None:
                for track in self._track_sys.tracks[1:]:
                    tb = _merge_bounds(tb, track.get_bounds())
                mb = self._mesh.bounds
                focus_bounds = _merge_bounds(tb, mb)
            else:
                focus_bounds = self._mesh.bounds

        self.reset_camera(bounds=focus_bounds)
        self.camera.zoom(1.3)
        self.camera.view_angle = CAMERA_FOV
        self._render()

    def _get_model_mesh_for_display(self):
        """返回要显示的模型。

        调整模式下由 PyVista 原生 TrackballActor 实时旋转 actor；
        固化时再把 actor 矩阵写回当前 mesh。
        """
        if self._mesh is None:
            return None
        return self._mesh

    def orbit_camera(self, d_azim, d_elev):
        """鼠标拖拽轨道旋转相机。"""
        self._azim = float(self._azim + d_azim)
        self._elev = float(np.clip(self._elev + d_elev, ELEV_MIN, ELEV_MAX))

        # ── 动态对焦：仅对齐主体（模型+轨道），忽略外围网格 ─────────────
        focus_bounds = None
        if self._mesh is not None and self._track_sys is not None:
            tb = self._track_sys.tracks[0].get_bounds() if self._track_sys.tracks else None
            if tb is not None:
                for track in self._track_sys.tracks[1:]:
                    tb = _merge_bounds(tb, track.get_bounds())
                mb = self._mesh.bounds
                focus_bounds = _merge_bounds(tb, mb)
            else:
                focus_bounds = self._mesh.bounds

        self.reset_camera(bounds=focus_bounds)
        self.camera.zoom(1.3)

    def hide_gizmos(self):
        """隐藏所有非模型辅助元素（轨道、网格地面、相机锚点、坐标轴、垂直基准面等）。
        返回原始可见性字典（用于 restore_gizmos 恢复）。
        """
        visibility = {}
        try:
            renderer = self.renderer
            gizmo_exact = ("ui_vertical_ground", "frustum_wireframe")
            gizmo_prefixes = ("track_", "cam_anchor")
            for name, actor in self.actors.items():
                if name == "model_mesh":
                    continue
                if name in gizmo_exact:
                    pass
                elif any(name.startswith(p) for p in gizmo_prefixes):
                    pass
                else:
                    continue
                visibility[name] = actor.GetVisibility()
                actor.SetVisibility(False)
        except Exception:
            pass
        return visibility

    def restore_gizmos(self, visibility):
        """恢复辅助元素的可见性。visibility 为 hide_gizmos 返回的字典。"""
        if not visibility:
            return
        try:
            renderer = self.renderer
            for name, was_visible in visibility.items():
                try:
                    actor = self.actors[name]
                    actor.SetVisibility(was_visible)
                except KeyError:
                    pass
        except Exception:
            pass

    def get_model_center(self):
        """返回模型的几何中心（世界坐标系），用于设置相机焦点。"""
        if self._mesh is not None:
            b = self._mesh.bounds
            cx = (b[0] + b[1]) * 0.5
            cy = (b[2] + b[3]) * 0.5
            cz = (b[4] + b[5]) * 0.5
            return np.array([cx, cy, cz], dtype=np.float64)
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def get_model_extent(self):
        """返回模型最大外接球半径（米级）。"""
        if self._mesh is None:
            return 1.0
        b = self._mesh.bounds
        ext = max(
            float(b[1] - b[0]),
            float(b[3] - b[2]),
            float(b[5] - b[4]),
        )
        return max(ext, 1e-4)

    def compute_orbit_radius(self):
        """纯粹几何轨道半径：模型最大尺寸的 3.5 倍，与 FOV 解耦。

        不再使用 tan(FOV) 反推——轨道物理位置与相机 FOV 完全独立。
        """
        extent = self.get_model_extent()
        orbit_radius = extent * 3.5
        self._orbit_radius = float(orbit_radius)
        print(f"[轨道] 模型最大尺寸: {extent:.4f}  |  轨道半径: {orbit_radius:.4f}")
        return self._orbit_radius

    def _set_initial_overview_camera(self):
        """导入后默认给一个能同时看到模型、轨道和 +X 出发点的概览视角。"""
        if self._mesh is None or self._track_sys is None:
            return

        self._elev = INITIAL_OVERVIEW_ELEV
        self._azim = INITIAL_OVERVIEW_AZIM

        radius = max(float(self._orbit_radius), self.get_model_extent(), 1.0)
        distance = max(radius * INITIAL_OVERVIEW_DISTANCE_FACTOR, 24.0)
        el = np.radians(self._elev)
        az = np.radians(self._azim)
        focus = self._get_track_center()
        cam_pos = focus + np.array(
            [
                distance * np.cos(el) * np.sin(az),
                distance * np.cos(el) * np.cos(az),
                distance * np.sin(el),
            ],
            dtype=np.float64,
        )

        self.camera_position = [
            tuple(float(x) for x in cam_pos),
            tuple(float(x) for x in focus),
            (0.0, 0.0, 1.0),
        ]
        self.camera.view_angle = CAMERA_FOV
        self.camera.clipping_range = (CAMERA_NEAR, CAMERA_FAR)
        try:
            self.reset_camera_clipping_range()
        except Exception:
            pass

    def zoom_at(self, factor):
        """滚轮缩放：factor < 1 为放大，> 1 为缩小。"""
        cam_pos = np.array(self.camera.position)
        dist = np.linalg.norm(cam_pos)
        new_dist = float(np.clip(dist * factor, CAMERA_NEAR * 10, 200.0))
        dir_vec = cam_pos / (dist + 1e-9)
        self.camera.position = tuple(dir_vec * new_dist)
        self.render()

    def _render(self):
        if not self._render_queued:
            self._render_queued = True
            QTimer.singleShot(16, self._do_render)

    def _do_render(self):
        self._render_queued = False
        try:
            self.render()
        except Exception:
            pass

    def _schedule_scene_rebuild(self):
        if not self._redraw_pending:
            self._redraw_pending = True
            QTimer.singleShot(16, self._do_scene_rebuild)

    def _do_scene_rebuild(self):
        self._redraw_pending = False
        self._render_scene()

    def load_obj(self, filepath):
        """加载 OBJ 文件，并转入本软件拍摄坐标系。

        核心原则：OBJLoader 先把模型规范成 MeshLab 语义坐标（顶面方向 +Z）。
        这里只负责：
            1) 把 +Z 顶面方向映射到四轨拍摄出发点 +X。
            2) 让底面平行于灰色 YZ 虚拟平面。
            3) TrackSystem 原生轨道 + 固定正面机位。
        """
        self._dataset_triplet = None
        self._dataset_meshes = {}
        self._dataset_mode = None
        self._destroy_dataset_plotters()
        self._dataset_parallel_scale = None
        loader = OBJLoader()
        verts, _, faces = loader.load(filepath)
        self._loader_normalization_report = loader.get_normalization_report()
        if not verts:
            return False

        # ── 坐标系落地：MeshLab 顶面(+Z) -> 本软件拍摄出发点(+X) ────────────
        # 变换后：底面为 YZ 平面，平行灰色虚拟平面；顶面朝四轨交叉相机点。
        display_verts = _obj_to_pv_coords(verts)

        self._verts_base = display_verts.copy()
        self._faces      = faces

        self._mesh = build_pyvista_mesh(display_verts, faces, MAX_TRIANGLES)
        if self._mesh is None:
            return False

        # ── 提取脚底板坐标（用于垂直基准面 UI 墙）────────────────────────────
        # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        # x_min 是建筑在 -X 方向的最远端，作为 YZ 垂直基准面的参考 X 坐标。
        self._bottom_x = self._mesh.bounds[0]

        # ── 轨道系统（原生，不旋转，不偏移）──────────────────────────────────
        self._track_sys = TrackSystem()
        self._track_sys.set_center([0.0, 0.0, 0.0])
        self._center_model_mesh_on_tracks()
        orbit_radius = self.compute_orbit_radius()
        for t in self._track_sys.tracks:
            t.set_radius(orbit_radius)
        self._track_sys.camera_distance = orbit_radius
        if not self._validate_import_pose():
            report = self._last_import_pose_report
            self._clear_loaded_model_state()
            raise RuntimeError(
                "导入姿态验收失败：模型底面/顶面没有按要求对齐。\n"
                f"{report}\n"
                "请检查 OBJ 是否满足：MeshLab 默认打开时顶面朝屏幕使用者，且模型底面为同一水平面。"
            )

        # ── 重置模型旋转状态 ───────────────────────────────────────────────
        self._model_actor = None
        self._model_transform_applied = False
        self._model_rot_elev = 0.0
        self._model_rot_azim = 0.0

        radii = self._track_sys.snapshot_radii()
        print(f"[MeshLab 复刻] 轨道半径（extent × 3.5）: {radii}")
        print(f"[MeshLab 复刻] 脚底板 bottom_x: {self._bottom_x:.4f}")

        # ── 重建场景（垂直基准面 + 轨道圆环）────────────────────────────────
        self._render_scene()

        # ── 初始观察视角：自动拉远，确保模型、轨道和 +X 出发点同时可见 ─────
        self._set_initial_overview_camera()

        # ── 强制立即渲染 ───────────────────────────────────────────────────
        self.render()

        print(
            f"[初始视角] 相机已设置  |  "
            f"overview elev={self._elev:.1f} azim={self._azim:.1f}  |  "
            f"pos={self.camera.position}  up={self.camera.up}"
        )
        return True

    def load_dataset_triplet(self, folder, mode=PREALIGNED_TRIPLET):
        """Load complete/incomplete/removed OBJ files for Dataset Mode."""
        triplet = load_dataset_triplet(folder, mode=mode)
        converted = {}
        meshes = {}
        faces_by_role = {}
        for role, mesh_data in (
            ("complete", triplet.complete),
            ("incomplete", triplet.incomplete),
            ("removed", triplet.removed),
        ):
            display_verts = _obj_to_pv_coords(mesh_data.vertices)
            converted[role] = display_verts
            faces_by_role[role] = mesh_data.faces
            meshes[role] = build_pyvista_mesh(display_verts, mesh_data.faces, max_tris=None)
            if meshes[role] is None:
                raise RuntimeError(f"无法构建 {role}.obj 的 PyVista mesh")

        self._dataset_triplet = triplet
        self._dataset_mode = mode
        self._dataset_meshes = meshes

        # Preview the incomplete input, but keep complete/removed in the same world frame.
        self._mesh = meshes["incomplete"]
        self._faces = faces_by_role["incomplete"]
        self._verts_base = converted["incomplete"].copy()
        self._bottom_x = meshes["complete"].bounds[0]
        self._loader_normalization_report = {
            "status": mode,
            "fallback_used": False,
            "reason": (
                "Dataset triplet uses one shared alignment from complete.obj."
                if mode == SHARED_TRIPLET
                else "Dataset triplet is frozen/aligned before Blender and loaded without transform."
            ),
        }

        self._track_sys = TrackSystem()
        self._track_sys.set_center([0.0, 0.0, 0.0])
        complete_extent = _mesh_max_extent(meshes["complete"])
        orbit_radius = max(complete_extent * 3.5, 1.0)
        self._dataset_parallel_scale = max(complete_extent * 0.75, 1.0)
        for track in self._track_sys.tracks:
            track.set_radius(orbit_radius)
        self._track_sys.camera_distance = orbit_radius
        self._orbit_radius = orbit_radius

        self._model_actor = None
        self._model_transform_applied = False
        self._model_rot_elev = 0.0
        self._model_rot_azim = 0.0
        self._last_import_pose_report = (
            f"{mode}: loaded frozen triplet; Dataset Mode does not compute mesh-dependent transforms"
        )

        self._render_scene()
        self._set_initial_overview_camera()
        self.render()
        return True

    def load_prealigned_dataset(self, folder):
        return self.load_dataset_triplet(folder, mode=PREALIGNED_TRIPLET)

    def load_shared_dataset(self, folder):
        return self.load_dataset_triplet(folder, mode=SHARED_TRIPLET)

    def export_aligned_complete(self, filepath, meta_path=None):
        """Export the current normal-mode aligned model in OBJ semantic coordinates."""
        if self._mesh is None:
            raise RuntimeError("No model is loaded")
        if self.has_dataset():
            raise RuntimeError("Dataset Mode cannot export aligned_complete.obj")

        self._solidify_model_transform()

        points_pv = np.asarray(self._mesh.points, dtype=np.float64)
        if points_pv.size == 0:
            raise RuntimeError("Current model has no vertices")
        if not self._faces:
            raise RuntimeError("Current model has no faces")

        points_obj = _pv_to_obj_coords(points_pv)
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write("# aligned_complete.obj exported by Python_3D_Scanner normal mode\n")
            f.write("# coordinate_space: obj_semantic_after_alignment\n")
            f.write("# inverse_fixed_conversion: obj=(pv_y,pv_z,pv_x)\n")
            for x, y, z in points_obj:
                f.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
            for face in self._faces:
                valid = [int(idx) for idx in face if 0 <= int(idx) < len(points_obj)]
                if len(valid) >= 3:
                    f.write("f " + " ".join(str(idx + 1) for idx in valid) + "\n")

        if meta_path is None:
            stem = os.path.splitext(os.path.basename(filepath))[0]
            meta_path = os.path.join(out_dir, f"{stem}_meta.json")
        else:
            meta_dir = os.path.dirname(os.path.abspath(meta_path))
            if meta_dir:
                os.makedirs(meta_dir, exist_ok=True)
        meta = {
            "exported_from_normal_mode": True,
            "coordinate_space": "obj_semantic_after_alignment",
            "inverse_fixed_conversion": "obj=(pv_y,pv_z,pv_x)",
            "intended_next_step": (
                "Blender cuts complete/incomplete/removed, then Dataset Mode "
                "loads prealigned triplet"
            ),
            "obj_path": os.path.abspath(filepath),
            "vertex_count": int(len(points_obj)),
            "face_count": int(len(self._faces)),
        }
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return filepath, meta_path

    def _add_frustum_visualization(self):
        """在模型加载后在主视口显示初始取景视锥体（静态，仅在 load_obj 后调用一次）。

        几何逻辑：
        - 相机位置 = 四条轨道的公共交点（水平轨道 X+ 端点），与拍摄第一帧完全一致
        - 焦平面 = 原点 (0,0,0)
        - 颜色固定为黑色，与 UI 斜视角解耦
        - 视锥矩形 = 以原点为中心、边长 = 2 * dist * tan(FOV/2) 的正方形
          （1:1 画幅，矩形宽高相等）
        """
        # 轨道公共交点：X+ 端点，与拍摄起点严格对齐
        cam_pos = self._track_sys.get_intersection_point()
        focal   = np.array([0.0, 0.0, 0.0])

        # 相机→焦平面的方向向量（归一化）
        cam_dir = focal - cam_pos
        dist    = np.linalg.norm(cam_dir)
        if dist < 1e-6:
            return
        cam_dir /= dist

        # 构建焦平面局部坐标系（right, up）
        # 优先以 world_up 为参考，若 cam_dir 与 world_up 共线则换备用轴
        world_up = np.array([0.0, 0.0, 1.0])
        ref_vec  = world_up

        # 检测 cam_dir 是否与 ref_vec 共线（gimbal lock 保护）
        #   - cam_pos=(r,0,0) → cam_dir=(-1,0,0)，与 Z 轴正交，安全
        #   - 若未来改回 Z 轴极点，需要此保护
        is_collinear = (abs(np.dot(cam_dir, ref_vec)) > 0.9999)
        if is_collinear:
            # cam_dir 几乎平行/反平行于 ref_vec，换 (1,0,0) 为参考轴
            ref_vec = np.array([1.0, 0.0, 0.0])
            if is_collinear and abs(np.dot(cam_dir, ref_vec)) > 0.9999:
                ref_vec = np.array([0.0, 1.0, 0.0])

        right    = np.cross(cam_dir, ref_vec)
        rn       = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([0.0, 0.0, 0.0])  # fallback，概率极低
        else:
            right /= rn
        up = np.cross(right, cam_dir)
        up /= np.linalg.norm(up)

        # 正方形取景框边长（1:1 画幅，矩形宽=高）
        half = dist * np.tan(np.radians(CAMERA_FOV * 0.5))

        # 焦平面矩形 4 个角（沿 cam_dir 正方向前移，悬浮于模型前方避免深度遮挡）
        depth = half * 0.05
        rect = [
            focal + right * (-half) + up * (-half) + cam_dir * depth,
            focal + right * (+half) + up * (-half) + cam_dir * depth,
            focal + right * (+half) + up * (+half) + cam_dir * depth,
            focal + right * (-half) + up * (+half) + cam_dir * depth,
        ]

        # 构建线段几何体：4 条射线 + 4 条矩形边，共 8 条线段
        # PyVista lines 格式：[n_verts, v0, v1, v2, v3, ...]
        cam_np = np.array(cam_pos)
        pts    = np.vstack([cam_np] + rect)       # 5 个点：index 0=相机, 1-4=矩形角

        seg_count = 8
        lines_arr = np.empty(seg_count * 3, dtype=np.int32)
        for i in range(4):
            lines_arr[i * 3:(i + 1) * 3] = [2, 0, i + 1]         # 射线
        for i in range(4):
            j = i + 1
            k = (i + 1) % 4 + 1
            lines_arr[(4 + i) * 3:(5 + i) * 3] = [2, j, k]       # 矩形边

        mesh = pv.PolyData(pts, lines=lines_arr)
        self.add_mesh(
            mesh,
            color="black",
            line_width=2,
            name="frustum_wireframe",
            pickable=False,
        )

    def _ensure_cap_plotter(self):
        """Step 1: 创建前绝对清理 + Step 2: 深拷贝 mesh。

        每次拍摄开始时调用一次，确保旧 plotter 被销毁，新 plotter 使用 mesh 深拷贝。
        """
        # Step 1: 创建前绝对清理
        if hasattr(self, "_cap_plotter") and self._cap_plotter is not None:
            try:
                self._cap_plotter.close()
            except Exception as e:
                print(f"[WARN] 离屏渲染器创建前清理失败 (可忽略): {e}")
                try:
                    if getattr(sys, 'frozen', False):
                        base_dir = os.path.dirname(sys.executable)
                    else:
                        base_dir = os.getcwd()
                    log_path = os.path.join(base_dir, "crash.log")
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{now}] [WARN] 离屏渲染器创建前清理失败: {e}\n")
                except Exception:
                    pass
            self._cap_plotter = None

        w = CAPTURE_SIZE_PX
        h = CAPTURE_SIZE_PX
        try:
            self._cap_plotter = pv.Plotter(off_screen=True, window_size=(w, h))

            # Step 2: 深拷贝（防止 VTK 底层内存交叉污染）
            mesh_for_render = self._mesh.copy(deep=True)
            self._cap_plotter.add_mesh(
                mesh_for_render,
                color="#c8cdd2",
                style="surface",
                smooth_shading=False,
                ambient=0.10,
                diffuse=0.90,
                specular=0.10,
                specular_power=50.0,
                reset_camera=False,
            )
            self._cap_plotter.disable_anti_aliasing()
            self._cap_plotter.enable_lightkit()
        except Exception as e:
            print(f"[致命错误] _ensure_cap_plotter 初始化失败: {e}")
            self._cap_plotter = None
            raise RuntimeError(f"[致命错误] _ensure_cap_plotter 初始化失败: {e}")

    def _destroy_cap_plotter(self):
        if hasattr(self, "_cap_plotter") and self._cap_plotter is not None:
            try:
                self._cap_plotter.close()
            except Exception as e:
                # 1. 正常控制台输出（开发环境可见）
                print(f"[WARN] 离屏渲染器销毁失败 (可忽略，防静默崩溃): {e}")

                # 2. 动态获取绝对路径并写入 crash.log（完美适配 exe 快捷方式等场景）
                try:
                    if getattr(sys, 'frozen', False):
                        base_dir = os.path.dirname(sys.executable)
                    else:
                        base_dir = os.getcwd()

                    log_path = os.path.join(base_dir, "crash.log")
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{now}] [WARN] 离屏渲染器销毁失败: {e}\n")
                except:
                    pass  # 写日志若失败则彻底静默，绝不阻塞主程序

            self._cap_plotter = None

    def _destroy_dataset_plotters(self):
        plotters = getattr(self, "_dataset_plotters", {}) or {}
        for name, plotter in list(plotters.items()):
            try:
                plotter.close()
            except Exception as e:
                print(f"[WARN] Dataset renderer cleanup failed for {name}: {e}")
        self._dataset_plotters = {}

    def _camera_model_dict(self):
        return {
            "focal_point": [0.0, 0.0, 0.0],
            "view_up": [0.0, 0.0, 1.0],
            "fov": float(CAMERA_FOV),
            "near": float(CAMERA_NEAR),
            "far": float(CAMERA_FAR),
            "parallel_projection": True,
            "parallel_scale": float(self._dataset_parallel_scale or 1.0),
            "image_size": [int(CAPTURE_SIZE_PX), int(CAPTURE_SIZE_PX)],
        }

    def _apply_capture_camera(self, plotter, cam_pos, parallel_scale=None):
        pos = np.asarray(cam_pos, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"capture camera expected position shape (3,), got {pos.shape}")
        cam = plotter.camera
        cam.position = tuple(float(x) for x in pos)
        cam.focal_point = (0.0, 0.0, 0.0)
        cam.up = (0.0, 0.0, 1.0)
        cam.view_angle = CAMERA_FOV
        cam.clipping_range = (CAMERA_NEAR, CAMERA_FAR)
        cam.ParallelProjectionOn()
        if parallel_scale is not None:
            cam.SetParallelScale(float(parallel_scale))
        return cam

    def _make_dataset_plotter(self, layers, background, mask):
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(CAPTURE_SIZE_PX, CAPTURE_SIZE_PX),
        )
        plotter.set_background(background)
        for mesh, color in layers:
            if mesh is None:
                continue
            plotter.add_mesh(
                mesh.copy(deep=True),
                color=color,
                style="surface",
                smooth_shading=False,
                ambient=1.0 if mask else 0.10,
                diffuse=0.0 if mask else 0.90,
                specular=0.0 if mask else 0.10,
                lighting=not mask,
                reset_camera=False,
            )
        plotter.disable_anti_aliasing()
        if not mask:
            plotter.enable_lightkit()
        return plotter

    def _ensure_dataset_plotters(self):
        if self._dataset_plotters:
            return
        meshes = self._dataset_meshes
        specs = {
            "complete_rgb": ([(meshes["complete"], "#c8cdd2")], "white", False),
            "incomplete_rgb": ([(meshes["incomplete"], "#c8cdd2")], "white", False),
            "missing_mask": ([(meshes["removed"], "white"), (meshes["incomplete"], "black")], "black", True),
            "visible_mask": ([(meshes["incomplete"], "white")], "black", True),
        }
        try:
            self._dataset_plotters = {
                name: self._make_dataset_plotter(layers, background, is_mask)
                for name, (layers, background, is_mask) in specs.items()
            }
        except Exception:
            self._destroy_dataset_plotters()
            raise

    def capture_dataset_layer(self, layer_name, cam_pos):
        if not self._dataset_plotters:
            raise RuntimeError("Dataset renderers are not initialized")
        if layer_name not in self._dataset_plotters:
            raise KeyError(f"Unknown dataset layer: {layer_name}")
        plotter = self._dataset_plotters[layer_name]
        self._apply_capture_camera(
            plotter,
            cam_pos,
            parallel_scale=self._dataset_parallel_scale,
        )
        plotter.render()
        self._apply_capture_camera(
            plotter,
            cam_pos,
            parallel_scale=self._dataset_parallel_scale,
        )
        img = plotter.screenshot(
            transparent_background=False,
            return_img=True,
        )
        if img is None:
            raise RuntimeError(f"Dataset renderer returned no image for {layer_name}")
        img = img[:, :, :3].astype(np.uint8)
        if layer_name.endswith("_mask"):
            gray = img.mean(axis=2)
            img = np.repeat((gray > 127).astype(np.uint8)[:, :, None] * 255, 3, axis=2)
        return img, plotter.camera.position

    def capture_frame(self, cam_pos, view_up=None):
        """用完全独立的隐藏 Plotter 渲染截图，UI 视图零干扰。

        规则：
        - 1024×1024 中等分辨率（优化3可选：若需工业级高清画质，将上方
          CAPTURE_SIZE_PX 全局常量改为 2048，关闭下方 "关闭渲染抗锯齿" 注释）
        - 关闭渲染抗锯齿（优化3可选：若需抗锯齿，删除下一行代码）
        - 相机 Position = 轨道点坐标（不做任何距离调整）
        - 相机 FocalPoint = (0,0,0)（与视锥体严格对齐）
        - Up 由调用方传入；CAD默认 +Z，建筑水平轨使用模型高度轴 +X
        - 只渲染模型，不含任何辅助元素
        - 返回 (img_array, real_cam_pos)，调用方负责分离

        Returns:
            tuple: (img_array, real_cam_pos) 或 (white_img, None) on error
        """
        if cam_pos is None:
            raise ValueError("capture_frame: cam_pos must not be None")

        pos = np.asarray(cam_pos, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"capture_frame: expected cam_pos shape (3,), got {pos.shape}")
        up = np.asarray(
            [0.0, 0.0, 1.0] if view_up is None else view_up,
            dtype=np.float64,
        ).reshape(3)
        up_norm = float(np.linalg.norm(up))
        if up_norm <= 1e-9:
            raise ValueError("capture_frame: view_up must be non-zero")
        up /= up_norm

        if self._mesh is None:
            return np.full((CAPTURE_SIZE_PX, CAPTURE_SIZE_PX, 3), 255, dtype=np.uint8), None

        # Step 1 guard: ensure plotter exists before use
        if not hasattr(self, "_cap_plotter") or self._cap_plotter is None:
            err = "_cap_plotter is None — call _ensure_cap_plotter before capture starts"
            print(f"[Capture] FATAL: {err}")
            raise RuntimeError(f"[Capture] FATAL: {err}")

        plotter = self._cap_plotter
        try:
            cam = plotter.camera
            cam.position        = tuple(float(x) for x in pos)
            cam.focal_point     = (0.0, 0.0, 0.0)
            cam.up              = tuple(float(x) for x in up)
            cam.view_angle      = CAMERA_FOV
            cam.clipping_range  = (CAMERA_NEAR, CAMERA_FAR)
            cam.ParallelProjectionOn()

            plotter.render()

            # re-apply to override render() internal state machine
            cam.position        = tuple(float(x) for x in pos)
            cam.focal_point     = (0.0, 0.0, 0.0)
            cam.up              = tuple(float(x) for x in up)
            cam.view_angle      = CAMERA_FOV
            cam.clipping_range  = (CAMERA_NEAR, CAMERA_FAR)
            cam.ParallelProjectionOn()

            img = plotter.screenshot(
                transparent_background=False,
                return_img=True,
            )
            real_pos = plotter.camera.position

            if img is not None:
                return img.astype(np.uint8), real_pos
            return np.full((CAPTURE_SIZE_PX, CAPTURE_SIZE_PX, 3), 255, dtype=np.uint8), None

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e)
            print(f"[Capture] EXCEPTION: {err_msg}")
            # 渲染器在整个拍摄循环中必须存活，不在此销毁；异常抛给外层 _capture_one 处理
            raise RuntimeError(f"[Capture] 渲染异常: {err_msg}")

    def capture_mesh_layers(self, cam_pos, layers, background="white", mask=False):
        """Deprecated one-shot renderer kept only for fallback/debug use.

        Dataset Mode uses reusable plotters from _ensure_dataset_plotters()
        so large batches do not create a new VTK renderer for every output.
        """
        if cam_pos is None:
            raise ValueError("capture_mesh_layers: cam_pos must not be None")
        pos = np.asarray(cam_pos, dtype=np.float64)
        if pos.shape != (3,):
            raise ValueError(f"capture_mesh_layers: expected cam_pos shape (3,), got {pos.shape}")

        plotter = None
        try:
            plotter = pv.Plotter(
                off_screen=True,
                window_size=(CAPTURE_SIZE_PX, CAPTURE_SIZE_PX),
            )
            plotter.set_background(background)
            for mesh, color in layers:
                if mesh is None:
                    continue
                plotter.add_mesh(
                    mesh.copy(deep=True),
                    color=color,
                    style="surface",
                    smooth_shading=False,
                    ambient=1.0 if mask else 0.10,
                    diffuse=0.0 if mask else 0.90,
                    specular=0.0 if mask else 0.10,
                    lighting=not mask,
                    reset_camera=False,
                )
            plotter.disable_anti_aliasing()
            if not mask:
                plotter.enable_lightkit()

            cam = self._apply_capture_camera(
                plotter,
                pos,
                parallel_scale=self._dataset_parallel_scale,
            )
            plotter.render()

            img = plotter.screenshot(
                transparent_background=False,
                return_img=True,
            )
            if img is None:
                img = np.full((CAPTURE_SIZE_PX, CAPTURE_SIZE_PX, 3), 0 if mask else 255, dtype=np.uint8)
            img = img[:, :, :3].astype(np.uint8)
            if mask:
                gray = img.mean(axis=2)
                img = np.repeat((gray > 127).astype(np.uint8)[:, :, None] * 255, 3, axis=2)
            return img, plotter.camera.position
        finally:
            if plotter is not None:
                try:
                    plotter.close()
                except Exception:
                    pass

    def _solidify_model_transform(self):
        """将 TrackballActor 的视觉姿态固化到 mesh 顶点数据。"""
        actor = self._model_actor
        if actor is None or self._mesh is None:
            return

        camera_state = self._snapshot_camera_state()
        self._lock_model_actor_center(force=True)

        try:
            vtk_mat = actor.GetMatrix()
            mat64 = np.array(
                [[vtk_mat.GetElement(i, j) for j in range(4)] for i in range(4)],
                dtype=np.float64,
            )
        except Exception:
            mat = None
            try:
                mat = actor.user_matrix
            except Exception:
                mat = None
            mat64 = np.eye(4, dtype=np.float64) if mat is None else np.array(mat, dtype=np.float64)

        if np.allclose(mat64, np.eye(4), atol=1e-6):
            self._center_model_mesh_on_tracks()
            self._reset_model_actor_transform()
            self._restore_camera_state(camera_state)
            return

        rt = mat64[:3, :]
        t = rt[:, 3]

        pts = self._mesh.points.copy()
        self._mesh.points[:] = (rt[:, :3] @ pts.T).T + t

        if self._verts_base is not None:
            self._verts_base = (rt[:, :3] @ self._verts_base.T).T + t

        self._reset_model_actor_transform()
        self._model_rot_elev = 0.0
        self._model_rot_azim = 0.0
        self._model_transform_applied = True

        self._center_model_mesh_on_tracks()

        self._render_scene()
        self._restore_camera_state(camera_state)
        self._render()

    # ── 交互模式切换（显式鼠标事件驱动）──────────────────────────────────

    def _enter_model_adjusting_mode(self):
        """进入 MODEL_ADJUSTING：启用 PyVista 原生 TrackballActor。"""
        self._adjust_last_xy = None
        self._model_dragging = False
        self._center_model_mesh_on_tracks()
        self._reset_model_actor_transform()
        self._freeze_environment_pickable()
        if self._model_actor is not None:
            try:
                self._model_actor.SetOrigin(*self._get_track_center())
                self._model_actor.SetPosition(0.0, 0.0, 0.0)
                self._model_actor.pickable = True
            except Exception:
                pass
        try:
            self.enable_trackball_actor_style()
        except Exception:
            pass
        self._model_center_lock_timer.start()
        self._lock_model_actor_center(force=True)
        self._model_transform_applied = False

    def _enter_viewing_mode(self):
        """进入 VIEWING 模式：恢复全局相机旋转。

        切换前的模型固化由 MainWindow._set_interaction_mode 根据 previous_mode 统一处理。
        """
        self._adjust_last_xy = None
        self._model_dragging = False
        self._model_center_lock_timer.stop()
        # 恢复相机旋转（包含滚轮缩放）——enable_trackball_style 必须保留
        self.enable_trackball_style()
        self._restore_environment_pickable()

    def _enter_track_adjusting_mode(self):
        """进入 TRACK_ADJUSTING 模式：类似 VIEWING，相机自由旋转。"""
        self._adjust_last_xy = None
        self._model_dragging = False
        self._model_center_lock_timer.stop()
        self.enable_trackball_style()
        self._restore_environment_pickable()

    def _freeze_environment_pickable(self):
        """强制所有非模型 actor 为 pickable=False（轨道、网格、锚点、视锥体）。"""
        try:
            for name, actor in self.actors.items():
                if name == "model_mesh":
                    continue
                actor.pickable = False
        except Exception:
            pass

    def _restore_environment_pickable(self):
        """恢复环境 actor 的 pickable 状态（轨道恢复为可交互）。"""
        try:
            for name, actor in self.actors.items():
                if name == "model_mesh":
                    continue
                actor.pickable = True
        except Exception:
            pass

    def set_mode(self, mode):
        self._mode = mode

    def set_active_track(self, idx):
        self._active_track = idx
        self._render_scene()

    def has_model(self):
        return self._mesh is not None

    def has_dataset(self):
        return self._dataset_triplet is not None and bool(self._dataset_meshes)

    def set_capture_subject(self, subject):
        value = "building" if subject == "building" else "cad"
        if self._capture_subject == value:
            return
        if self._mesh is not None:
            self._solidify_model_transform()
        self._capture_subject = value
        if self._mesh is not None:
            self._render_scene()

    def get_track_system(self):
        return self._track_sys

    def get_elev_azim(self):
        return float(self._elev), float(self._azim)

    def get_look_target(self):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def get_plotter(self):
        return self  # QtInteractor 本身就是 Plotter


# ═══════════════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════════════

def _obj_to_pv_coords(verts_obj):
    """把 OBJLoader 的 MeshLab 语义坐标转成软件拍摄坐标。

    输入：OBJLoader 已经完成规范化，建筑顶面/底面法向为 +Z/-Z。
    输出：PyVista/拍摄坐标 (z, x, y)：
        - 原 +Z 顶面方向 -> 新 +X，正对四轨交叉拍摄出发点。
        - 原 XY 底面 -> 新 YZ 平面，平行软件灰色虚拟平面。
        - 保持右手坐标系：old X -> new Y, old Y -> new Z, old Z -> new X。

    这是导入时的固定姿态落地，不影响后续用户手动模型调整。
    """
    v = np.asarray(verts_obj, dtype=np.float64)
    return np.column_stack([v[:, 2], v[:, 0], v[:, 1]])


def _pv_to_obj_coords(verts_pv):
    """Inverse of _obj_to_pv_coords: OBJ semantic coords = (pv_y, pv_z, pv_x)."""
    v = np.asarray(verts_pv, dtype=np.float64)
    return np.column_stack([v[:, 1], v[:, 2], v[:, 0]])


def _rotate_verts_around_center(verts, center, elev_deg, azim_deg):
    """绕任意中心点旋转顶点（弧度顺序：X轴 → Z轴）。"""
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]], dtype=np.float64)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]], dtype=np.float64)
    R  = Rx @ Rz
    p  = np.asarray(verts, dtype=np.float64)
    return (p - center) @ R.T + center


def _face_to_tri_indices(face, n_verts):
    """多边形面扇形三角化，返回 (i0, i1, i2) 元组列表。"""
    idxs = [int(i) for i in face if 0 <= int(i) < n_verts]
    if len(idxs) < 3:
        return []
    if len(idxs) == 3:
        return [tuple(idxs)]
    return [(idxs[0], idxs[k], idxs[k + 1]) for k in range(1, len(idxs) - 1)]


def build_pyvista_mesh(verts, faces, max_tris=MAX_TRIANGLES):
    """将 OBJ 数据转换为 PyVista PolyDataMesh。"""
    n_verts = len(verts)
    tris = []
    for face in faces:
        tris.extend(_face_to_tri_indices(face, n_verts))
    if not tris:
        return None
    tri_idx = np.array(tris, dtype=np.int64)
    if max_tris is not None and max_tris > 0 and len(tri_idx) > max_tris:
        idx = np.linspace(0, len(tri_idx) - 1, max_tris, dtype=np.int64)
        tri_idx = tri_idx[idx]
    cells = np.concatenate([np.full((len(tri_idx), 1), 3), tri_idx], axis=1)
    pts = np.asarray(verts, dtype=np.float64)
    return pv.PolyData(pts, cells)


def _merge_bounds(a, b):
    """合并两组 bounds：(xmin,xmax,ymin,ymax,zmin,zmax)。"""
    return (min(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), max(a[3], b[3]),
            min(a[4], b[4]), max(a[5], b[5]))


def _mesh_max_extent(mesh):
    if mesh is None:
        return 1.0
    b = mesh.bounds
    return max(
        float(b[1] - b[0]),
        float(b[3] - b[2]),
        float(b[5] - b[4]),
        1e-4,
    )


# ═══════════════════════════════════════════════════════════════════════
#  拍照线程（主线程渲染架构）
#  - CaptureWorker：纯计算线程（QObject + QThread）
#    仅负责角度/节奏计算，通过 pyqtSignal 发送数据到主线程。
#  - 主线程槽函数：接收信号后执行相机更新 + 渲染 + 保存图片。
#  - QTimer：驱动拍摄循环节拍，主线程逐帧完成拍摄。
# ═══════════════════════════════════════════════════════════════════════

class CaptureWorker(QObject):
    """纯计算型拍摄线程，通过信号与主线程通信。"""

    tick = Signal(object, int, int, int)  # (pos, t_idx, p_idx, cur)
    finished = Signal(str)
    stopped = Signal()

    def __init__(self, pv_view, output_dir, photos_per_track=16, interval_ms=150):
        super().__init__()
        self._pv    = pv_view
        self.out_dir   = output_dir
        self.photos    = photos_per_track
        self.interval  = interval_ms
        self._run      = True

    def run(self):
        """子线程入口：只负责遍历位置并发射信号，渲染由主线程完成。"""
        try:
            mgr = ScreenshotManager(self.out_dir)
            mgr.ensure_output_dir()
            el, az = self._pv.get_elev_azim()
            track_sys = self._pv.get_track_system()
            all_pos = track_sys.get_all_camera_positions(
                elev_deg=el, azim_deg=az, num_positions=self.photos)

            flat = []
            for t_idx, block in enumerate(all_pos):
                positions = block.get("positions", []) if isinstance(block, dict) else block
                for p_idx, pos in enumerate(positions):
                    flat.append((t_idx, p_idx, pos))

            cur = 0
            for t_idx, p_idx, pos in flat:
                if not self._run:
                    self.stopped.emit()
                    return
                cur += 1
                self.tick.emit(pos, t_idx, p_idx, cur)
                QThread.msleep(self.interval)

            self.finished.emit(
                f"拍摄完成！共 {len(flat)} 张\n保存至:\n{mgr.get_output_dir()}")
        except Exception as ex:
            import traceback; traceback.print_exc()
            self.finished.emit(f"拍摄出错: {ex}")

    def stop(self):
        self._run = False


# ═══════════════════════════════════════════════════════════════════════
#  主窗口（PySide6）
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scanner - 模型拍摄工具")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 980)

        # 交互状态
        # 批量拍摄状态（_capture_one 递归驱动，after 非阻塞）
        self._capture_out_dir  = None
        self._capture_flat     = []
        self._capture_idx      = 0
        self._capture_mgr      = None
        self._capture_stopping = False
        self._camera_pose_log  = []  # [(img_name, target_pos, real_cam_pos, view_up), ...]
        self._active_capture_profile = None
        self._dataset_pose_log = []
        self._capture_is_dataset = False
        self._dataset_save_futures = []
        self._dataset_failed_outputs = []
        self.save_parent_dir    = None
        self.last_obj_dir       = os.getcwd()   # 导入 OBJ 的独立记忆路径
        self.last_save_dir      = os.getcwd()   # 保存位置的独立记忆路径
        self.export_parent_dir  = None
        self.last_export_dir    = os.getcwd()   # 导出 aligned_complete.obj 的独立记忆路径
        self._active_track     = None
        self._photo_count      = 32
        self._capture_subject  = "cad"
        self._capture_queue    = []   # [(t_idx, p_idx, pos, target), ...]
        self._capture_index    = 0
        self._capture_mgr      = None
        # ── 拍摄线程池（异步存图，避免 I/O 阻塞渲染）───────────────────────
        self._capture_executor = ThreadPoolExecutor(max_workers=2)

        # ── 主布局：水平分栏（左侧 3D 视口，右侧控制面板）──────────────
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 左侧：3D 视口（PyVistaView = QtInteractor = QWidget）────────
        self._pv_view = PyVistaView(self)
        main_layout.addWidget(self._pv_view, 1)  # stretch=1，占满剩余空间

        # ── 右侧：控制面板 ─────────────────────────────────────────────
        self._setup_panel()
        main_layout.addWidget(self._panel, 0)     # stretch=0，固定宽度

        # 初始模式
        self._set_interaction_mode(InteractionMode.VIEWING)
        self._update_buttons()
        self.setFocusPolicy(Qt.StrongFocus)

    # ── 3D 视口属性（兼容旧代码） ───────────────────────────────────
    @property
    def _pv_frame(self):
        return self._pv_view

    # ── 右侧面板 ──────────────────────────────────────────────────────

    def _setup_panel(self):
        self._panel = QFrame()
        self._panel.setObjectName("panel")
        self._panel.setFixedWidth(340)
        self._panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        palette = self._panel.palette()
        palette.setColor(QPalette.Window, QColor("#E8E8E8"))
        self._panel.setPalette(palette)
        self._panel.setAutoFillBackground(True)

        layout = QVBoxLayout(self._panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._make_project_group())
        layout.addWidget(self._make_mode_group())
        layout.addWidget(self._make_capture_group())
        layout.addStretch(1)
        layout.addWidget(self._make_status_bar())

    def _make_project_group(self):
        """Project 分组：文件信息 + 导入按钮 + 保存位置按钮。"""
        group = QFrame()
        group.setObjectName("section")

        vl = QVBoxLayout(group)
        vl.setContentsMargins(8, 6, 8, 6)
        vl.setSpacing(4)

        # 标题栏
        title_lbl = QLabel("Project", group)
        title_lbl.setObjectName("sectionTitle")
        title_lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        title_lbl.setStyleSheet(
            "QLabel#sectionTitle { background: #D0D0D0; color: #111111; "
            "padding: 4px 8px; font-weight: bold; }")
        title_lbl.setFixedHeight(22)
        vl.addWidget(title_lbl)

        # 内容区
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(6, 6, 6, 6)
        body_layout.setSpacing(4)
        vl.addLayout(body_layout)

        self.lbl_info = QLabel("未加载模型", group)
        self.lbl_info.setFont(QFont("Consolas", 10))
        self.lbl_info.setStyleSheet("color: #333333; background: transparent;")
        self.lbl_info.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setMinimumHeight(48)
        body_layout.addWidget(self.lbl_info)

        self.btn_import = self._make_btn_big("导入 OBJ", "#4B8CC8", group=group)
        self.btn_import.clicked.connect(self._on_import)
        body_layout.addWidget(self.btn_import)

        self.btn_import_dataset = self._make_btn_big("导入数据集", "#00897B", group=group)
        self.btn_import_dataset.clicked.connect(self._on_import_dataset)
        body_layout.addWidget(self.btn_import_dataset)

        self.btn_export_aligned = self._make_btn_big("导出已对齐完整模型", "#546E7A", group=group)
        self.btn_export_aligned.clicked.connect(self._on_export_aligned)
        self.btn_export_aligned.setEnabled(False)
        body_layout.addWidget(self.btn_export_aligned)

        self.btn_export_folder = self._make_btn_big("导出位置", "#607D8B", group=group)
        self.btn_export_folder.clicked.connect(self._on_export_folder)
        body_layout.addWidget(self.btn_export_folder)

        self.btn_folder = self._make_btn_big("保存位置", "#7B7B7B", group=group)
        self.btn_folder.clicked.connect(self._on_folder)
        body_layout.addWidget(self.btn_folder)

        return group

    def _make_mode_group(self):
        """调整模式分组：调整轨道 + 调整模型 + 轨道选择。"""
        group = QFrame()
        group.setObjectName("section")

        vl = QVBoxLayout(group)
        vl.setContentsMargins(8, 6, 8, 6)
        vl.setSpacing(4)

        title_lbl = QLabel("调整模式", group)
        title_lbl.setObjectName("sectionTitle")
        title_lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        title_lbl.setStyleSheet(
            "QLabel#sectionTitle { background: #D0D0D0; color: #111111; "
            "padding: 4px 8px; font-weight: bold; }")
        title_lbl.setFixedHeight(22)
        vl.addWidget(title_lbl)

        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(6, 6, 6, 6)
        body_layout.setSpacing(4)
        vl.addLayout(body_layout)

        self.btn_track = self._make_btn_big("调整轨道", "#4A148C", group=group)
        self.btn_track.clicked.connect(self._toggle_track_adjust_mode)
        self.btn_track.setFixedHeight(52)
        body_layout.addWidget(self.btn_track)

        self.btn_inspect = self._make_btn_big("调整模型", "#1B5E20", group=group)
        self.btn_inspect.clicked.connect(self._toggle_model_adjust)
        self.btn_inspect.setFixedHeight(52)
        body_layout.addWidget(self.btn_inspect)

        lbl = QLabel("选择轨道 (1-4)：在轨道模式下，点住对应圆环拖拽可调大小", group)
        lbl.setFont(QFont("Microsoft YaHei", 9))
        lbl.setStyleSheet("color: #333333; background: transparent;")
        lbl.setWordWrap(True)
        lbl.setFixedHeight(32)
        body_layout.addWidget(lbl)

        grid = QWidget(group)
        grid_layout = QHBoxLayout(grid)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(6)
        self.track_btns = []
        names = ["1 水平", "2 竖直", "3 左斜", "4 右斜"]
        for i in range(4):
            c = TRACK_COLORS.get(i, (1.0, 1.0, 1.0))
            hex_c = "#{:02X}{:02X}{:02X}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            b = QPushButton(names[i], grid)
            b.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(36)
            b.setStyleSheet(
                f"QPushButton {{ background: {hex_c}; color: white; "
                f"border: 1px solid #444; border-radius: 3px; font-weight: bold; }}"
                f"QPushButton:hover {{ background: {self._lighten(hex_c)}; }}"
                f"QPushButton:disabled {{ background: #999; color: #ddd; }}")
            b.clicked.connect(lambda _, idx=i: self._toggle_track(idx))
            grid_layout.addWidget(b)
            self.track_btns.append(b)
        body_layout.addWidget(grid)

        return group

    def _make_capture_group(self):
        """Capture 分组：照片数量选择 + 拍摄按钮 + 停止按钮 + 进度条。"""
        group = QFrame()
        group.setObjectName("section")

        vl = QVBoxLayout(group)
        vl.setContentsMargins(8, 6, 8, 6)
        vl.setSpacing(4)

        title_lbl = QLabel("Capture", group)
        title_lbl.setObjectName("sectionTitle")
        title_lbl.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        title_lbl.setStyleSheet(
            "QLabel#sectionTitle { background: #D0D0D0; color: #111111; "
            "padding: 4px 8px; font-weight: bold; }")
        title_lbl.setFixedHeight(22)
        vl.addWidget(title_lbl)

        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(6, 6, 6, 6)
        body_layout.setSpacing(4)
        vl.addLayout(body_layout)

        lbl_subject = QLabel("拍摄对象：", group)
        lbl_subject.setFont(QFont("Microsoft YaHei", 9))
        lbl_subject.setStyleSheet("color: #333333; background: transparent;")
        body_layout.addWidget(lbl_subject)

        self._capture_subject_group = QButtonGroup(group)
        self._capture_subject_buttons = {}
        for value, text in [
            ("cad", "CAD模型：四轨360度"),
            ("building", "建筑物模型：水平环绕360度，其余上半轨180度"),
        ]:
            rb = QRadioButton(text, group)
            rb.setFont(QFont("Microsoft YaHei", 9))
            rb.setStyleSheet("color: #333333; background: transparent;")
            rb.setCursor(Qt.PointingHandCursor)
            if value == "cad":
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, v=value: self._on_capture_subject_toggled(v, checked))
            self._capture_subject_group.addButton(rb)
            self._capture_subject_buttons[value] = rb
            body_layout.addWidget(rb)

        lbl_count = QLabel("选择照片数量：", group)
        lbl_count.setFont(QFont("Microsoft YaHei", 9))
        lbl_count.setStyleSheet("color: #333333; background: transparent;")
        body_layout.addWidget(lbl_count)

        self._photo_group = QButtonGroup(group)
        self._photo_buttons = {}
        for val, text in [(32, "32张 (每轨8张)"), (64, "64张 (每轨16张)"), (96, "96张 (每轨24张)")]:
            rb = QRadioButton(text, group)
            rb.setFont(QFont("Microsoft YaHei", 9))
            rb.setStyleSheet("color: #333333; background: transparent;")
            rb.setCursor(Qt.PointingHandCursor)
            if val == 32:
                rb.setChecked(True)
            rb.toggled.connect(lambda checked, v=val: self._on_photo_toggled(v, checked))
            self._photo_group.addButton(rb, val)
            self._photo_buttons[val] = rb
            body_layout.addWidget(rb)

        self.btn_capture = self._make_btn_big("拍摄所有照片 (32张)", "#1565C0", group=group)
        self.btn_capture.clicked.connect(self._on_capture)
        body_layout.addWidget(self.btn_capture)

        self.btn_stop = self._make_btn_big("停止", "#B71C1C", group=group)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        body_layout.addWidget(self.btn_stop)

        self.prog_lbl = QLabel("进度: 0 / 32", group)
        self.prog_lbl.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        self.prog_lbl.setStyleSheet("color: #333333; background: transparent;")
        self.prog_lbl.setAlignment(Qt.AlignLeft)
        body_layout.addWidget(self.prog_lbl)

        self.prog_bar = QProgressBar(group)
        self.prog_bar.setMaximum(32)
        self.prog_bar.setValue(0)
        self.prog_bar.setFixedHeight(16)
        self.prog_bar.setTextVisible(True)
        self.prog_bar.setFormat("%p%")
        self.prog_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #aaa; border-radius: 4px; text-align: center; }"
            "QProgressBar::chunk { background: #1565C0; border-radius: 3px; }")
        body_layout.addWidget(self.prog_bar)

        return group

    def _make_btn_big(self, text, color_hex, group=None):
        btn = QPushButton(text, group)
        btn.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(50)
        btn.setStyleSheet(
            f"QPushButton {{ background: {color_hex}; color: white; "
            f"border: 2px solid #444; border-radius: 4px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {self._lighten(color_hex)}; }}"
            f"QPushButton:pressed {{ background: {self._darken(color_hex)}; }}"
            f"QPushButton:disabled {{ background: #888; color: #ccc; }}")
        return btn

    def _lighten(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        fac = 1.2
        return f"#{min(int(r*fac),255):02X}{min(int(g*fac),255):02X}{min(int(b*fac),255):02X}"

    def _darken(self, hex_color):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        fac = 0.85
        return f"#{max(int(r*fac),0):02X}{max(int(g*fac),0):02X}{max(int(b*fac),0):02X}"

    def _set_mode_button_style(self, btn, color_hex, selected=False):
        border = "2px solid #1a1a1a" if selected else "2px solid #444"
        btn.setStyleSheet(
            f"QPushButton {{ background: {color_hex}; color: white; border: {border}; "
            f"border-radius: 4px; font-weight: bold; }}"
            f"QPushButton:hover:enabled {{ background: {self._lighten(color_hex)}; }}"
            f"QPushButton:pressed:enabled {{ background: {self._darken(color_hex)}; }}"
            "QPushButton:disabled { background: #888; color: #ccc; border: 2px solid #666; }"
        )

    def _make_status_bar(self):
        bar = QFrame()
        bar.setObjectName("statusbar")
        bar.setFixedHeight(28)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bar.setStyleSheet("QFrame#statusbar { background: #1c1c1c; }")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)
        self.status_lbl = QLabel("导入 OBJ 后在右侧操作")
        self.status_lbl.setFont(QFont("Microsoft YaHei", 9))
        self.status_lbl.setStyleSheet("color: #aaaaaa; background: transparent; border: none;")
        layout.addWidget(self.status_lbl)
        return bar

    # ── 模式切换 ──────────────────────────────────────────────────────

    def _toggle_model_adjust(self):
        if not self._pv_view.has_model():
            self.status_lbl.setText("请先导入 OBJ 模型，再调整模型")
            return
        if self._pv_view.has_dataset():
            self.status_lbl.setText("Dataset Mode 禁止单独调整模型，避免破坏三元组共享坐标系")
            return
        if self._pv_view._mode == InteractionMode.MODEL_ADJUSTING:
            self._set_interaction_mode(InteractionMode.VIEWING)
        else:
            self._set_interaction_mode(InteractionMode.MODEL_ADJUSTING)

    def _toggle_track_adjust_mode(self):
        if not self._pv_view.has_model():
            self.status_lbl.setText("请先导入 OBJ 模型，再调整轨道")
            return
        if self._pv_view._mode == InteractionMode.TRACK_ADJUSTING:
            self._set_interaction_mode(InteractionMode.VIEWING)
        else:
            self._active_track = None
            self._set_interaction_mode(InteractionMode.TRACK_ADJUSTING)

    def _toggle_track(self, idx):
        if not self._pv_view.has_model():
            self.status_lbl.setText("请先导入 OBJ 模型，再选择轨道")
            return
        if self._pv_view._mode == InteractionMode.TRACK_ADJUSTING and self._active_track == idx:
            self._set_interaction_mode(InteractionMode.VIEWING)
        else:
            self._active_track = idx
            self._set_interaction_mode(InteractionMode.TRACK_ADJUSTING)

    def _set_interaction_mode(self, mode):
        previous_mode = self._pv_view._mode

        # 退出“调整模型”时把 TrackballActor 姿态写回 mesh，并保持当前相机缩放/视角。
        if previous_mode == InteractionMode.MODEL_ADJUSTING and mode != InteractionMode.MODEL_ADJUSTING:
            self._pv_view._solidify_model_transform()

        self._pv_view._mode = mode
        self._pv_view._active_track = self._active_track

        # ── 触发 PyVista 交互模式切换 ─────────────────────────────────────
        if mode == InteractionMode.MODEL_ADJUSTING:
            self._pv_view._enter_model_adjusting_mode()
        elif mode == InteractionMode.VIEWING:
            self._pv_view._enter_viewing_mode()
        elif mode == InteractionMode.TRACK_ADJUSTING:
            self._pv_view._enter_track_adjusting_mode()

        # ── 按钮高亮样式 ───────────────────────────────────────────────────
        self._set_mode_button_style(
            self.btn_inspect,
            "#2E7D32" if mode == InteractionMode.MODEL_ADJUSTING else "#1B5E20",
            selected=(mode == InteractionMode.MODEL_ADJUSTING),
        )
        self._set_mode_button_style(
            self.btn_track,
            "#6A1B9A" if mode == InteractionMode.TRACK_ADJUSTING else "#4A148C",
            selected=(mode == InteractionMode.TRACK_ADJUSTING),
        )

        self._refresh_track_btns()
        self._update_status_bar(mode)

    def _update_status_bar(self, mode):
        if mode == InteractionMode.VIEWING:
            self.status_lbl.setText("默认检视：左键拖拽旋转视角，滚轮缩放")
        elif mode == InteractionMode.MODEL_ADJUSTING:
            self.status_lbl.setText("调整模型：左键拖拽旋转模型自身，滚轮缩放")
        elif mode == InteractionMode.TRACK_ADJUSTING:
            names = TRACK_NAMES
            idx = self._active_track if self._active_track is not None else 0
            self.status_lbl.setText(
                f"调整轨道{idx+1}（{names[idx]}）：左键在圆环上拖拽调整半径；空白处旋转视角")

    def _refresh_track_btns(self):
        sel = self._active_track if self._pv_view._mode == InteractionMode.TRACK_ADJUSTING else None
        for i, b in enumerate(self.track_btns):
            c = TRACK_COLORS.get(i, (1.0, 1.0, 1.0))
            hex_c = "#{:02X}{:02X}{:02X}".format(int(c[0]*255), int(c[1]*255), int(c[2]*255))
            if i == sel:
                b.setStyleSheet(
                    f"QPushButton {{ background: {hex_c}; color: white; "
                    f"border: 2px solid #111; border-radius: 3px; font-weight: bold; }}"
                    "QPushButton:disabled { background: #888; color: #ccc; border: 1px solid #666; }")
            else:
                b.setStyleSheet(
                    f"QPushButton {{ background: {hex_c}; color: white; "
                    f"border: 1px solid #444; border-radius: 3px; font-weight: bold; }}"
                    f"QPushButton:hover:enabled {{ background: {self._lighten(hex_c)}; }}"
                    "QPushButton:disabled { background: #888; color: #ccc; border: 1px solid #666; }")

    # ── 按钮回调 ──────────────────────────────────────────────────────

    def _on_capture_subject_toggled(self, value, checked):
        if checked:
            self._capture_subject = value
            self._pv_view.set_capture_subject(value)

    def _on_photo_toggled(self, val, checked):
        if checked:
            self._photo_count = val
            self.btn_capture.setText(f"拍摄所有照片 ({val}张)")
            self.prog_lbl.setText(f"进度: 0 / {val}")
            self.prog_bar.setMaximum(val)

    def _normal_capture_profile(self):
        if getattr(self, "_capture_subject", "cad") == "building":
            return {
                "profile_version": "building-v9-horizontal-parallel-ground",
                "capture_subject": "building",
                "model_top_axis": [1.0, 0.0, 0.0],
                "model_pose_rule": "roof_plus_x_points_to_legacy_track_anchor",
                "model_pose_anchor_source": "legacy_four_track_common_anchor_plus_x",
                "ground_rule": "model_bottom_on_virtual_yz_plane",
                "underside_rule": "camera_x_greater_than_or_equal_to_track_center_x",
                "camera_roll_rule": "track1_fixed_model_up_x_tracks234_fixed_world_up_z",
                "tracks": {
                    "track1": "horizontal_yz_parallel_ground_360",
                    "track2": "original_vertical_upper_half_180",
                    "track3": "original_left_diagonal_upper_half_180",
                    "track4": "original_right_diagonal_upper_half_180",
                },
            }
        return {
            "capture_subject": "cad",
            "tracks": {
                "track1": "legacy_circle_360",
                "track2": "legacy_circle_360",
                "track3": "legacy_circle_360",
                "track4": "legacy_circle_360",
            },
        }

    def _on_import(self):
        self._capture_stopping = True
        self._restore_gizmos()
        fp, _ = QFileDialog.getOpenFileName(
            self, "选择 OBJ 文件", self.last_obj_dir,
            "OBJ 文件 (*.obj);;所有文件 (*.*)")
        if not fp:
            return
        self.last_obj_dir = os.path.dirname(fp)
        try:
            loaded = self._pv_view.load_obj(fp)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            loaded = False
            self.status_lbl.setText(f"导入失败: {str(ex)[:80]}")
            QMessageBox.critical(self, "导入失败", f"加载 OBJ 时出错:\n{ex}")

        if loaded:
            # 导入成功后默认进入 TRACK_ADJUSTING，让用户可以自由环视全局
            self._active_track = None
            self._set_interaction_mode(InteractionMode.TRACK_ADJUSTING)
            self._refresh_track_btns()
            self.prog_bar.setValue(0)
            self.prog_lbl.setText(f"进度: 0 / {self._photo_count}")
            name  = os.path.basename(fp)
            verts = len(self._pv_view._verts_base) if self._pv_view._verts_base is not None else 0
            faces = self._pv_view._mesh.n_cells if self._pv_view._mesh else 0
            self.lbl_info.setText(f"{name}\n顶点数: {verts}\n面数: {faces}")
            self.status_lbl.setText(f"已加载: {name}  |  轨道已自动适配模型大小")
        else:
            QMessageBox.warning(self, "错误", "无法加载该 OBJ 文件")
        self._update_buttons()

    def _on_import_dataset(self):
        self._capture_stopping = True
        self._restore_gizmos()
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择冻结三元组文件夹（complete/incomplete/removed）",
            self.last_obj_dir,
        )
        if not folder:
            return
        self.last_obj_dir = folder
        try:
            loaded = self._pv_view.load_prealigned_dataset(folder)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            loaded = False
            self.status_lbl.setText(f"数据集导入失败: {str(ex)[:80]}")
            QMessageBox.critical(self, "数据集导入失败", f"加载三元组时出错:\n{ex}")

        if loaded:
            self._active_track = None
            self._set_interaction_mode(InteractionMode.TRACK_ADJUSTING)
            self._refresh_track_btns()
            self.prog_bar.setValue(0)
            self.prog_lbl.setText(f"进度: 0 / {self._photo_count}")
            triplet = self._pv_view._dataset_triplet
            self.lbl_info.setText(
                "Dataset Mode\n"
                f"mode: {triplet.mode}\n"
                f"complete: {len(triplet.complete.vertices)} vertices\n"
                f"incomplete: {len(triplet.incomplete.vertices)} vertices\n"
                f"removed: {len(triplet.removed.vertices)} vertices\n"
                f"warnings: {len(triplet.validation_warnings)}"
            )
            if triplet.validation_warnings:
                warning_text = "\n".join(f"- {w}" for w in triplet.validation_warnings)
                QMessageBox.warning(
                    self,
                    "数据集坐标检查警告",
                    "三元组已加载，但坐标 sanity check 发现风险：\n\n"
                    f"{warning_text}\n\n"
                    "软件不会自动修正坐标，请确认这组三元组确实来自同一个冻结后的 aligned_complete.obj。",
                )
            self.status_lbl.setText(
                "已加载冻结三元组：Dataset Mode 不 normalize/recenter/rescale/reorient"
            )
        else:
            QMessageBox.warning(self, "错误", "无法加载该数据集三元组")
        self._update_buttons()

    def _on_export_aligned(self):
        if not self._pv_view.has_model():
            QMessageBox.warning(self, "提示", "请先导入普通 OBJ 模型")
            return
        if self._pv_view.has_dataset():
            QMessageBox.warning(self, "提示", "Dataset Mode 不导出 aligned_complete.obj，请在普通模式导出")
            return

        out_dir = self.export_parent_dir or self.last_export_dir or os.getcwd()
        obj_dir = os.path.join(out_dir, "aligned_complete")
        meta_dir = os.path.join(out_dir, "aligned_complete_json")
        os.makedirs(obj_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        index = 1
        while True:
            filepath = os.path.join(obj_dir, f"{index}.obj")
            meta_path = os.path.join(meta_dir, f"{index}.json")
            if not os.path.exists(filepath) and not os.path.exists(meta_path):
                break
            index += 1

        try:
            obj_path, meta_path = self._pv_view.export_aligned_complete(filepath, meta_path)
        except Exception as ex:
            QMessageBox.critical(self, "导出失败", f"导出 aligned_complete.obj 时出错:\n{ex}")
            return

        self.export_parent_dir = out_dir
        self.last_export_dir = self.export_parent_dir
        self.status_lbl.setText(f"已导出已对齐完整模型: {obj_path}")
        QMessageBox.information(
            self,
            "导出完成",
            f"已导出:\n{obj_path}\n\n元数据:\n{meta_path}",
        )
        self._update_buttons()

    def _on_export_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择 aligned_complete.obj 导出目录", self.last_export_dir)
        if folder:
            self.export_parent_dir = folder
            self.last_export_dir = folder
            self.status_lbl.setText(f"导出目录: {folder}")

    def _on_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择保存根目录", self.last_save_dir)
        if folder:
            self.save_parent_dir = folder
            self.last_save_dir   = folder
            self.status_lbl.setText(f"保存根目录: {folder}")

    def _on_capture(self):
        if not self._pv_view.has_model():
            QMessageBox.warning(self, "提示", "请先导入 OBJ 模型")
            return
        if self._pv_view.has_dataset():
            self._on_capture_dataset()
            return

        # 如果旧线程池还在（比如上轮拍摄已 shutdown），重新实例化
        if hasattr(self, '_capture_executor') and self._capture_executor is not None:
            try:
                self._capture_executor.shutdown(wait=False)
            except Exception:
                pass
        self._capture_executor = ThreadPoolExecutor(max_workers=4)

        base = self.save_parent_dir if self.save_parent_dir else os.getcwd()
        try:
            out_dir = allocate_next_model_folder(base)
        except OSError as e:
            QMessageBox.critical(self, "错误", f"无法创建保存目录:\n{e}")
            return

        self._capture_stopping = False
        self._disable_ui_for_capture()
        self.prog_bar.setValue(0)
        total = self._photo_count
        self.prog_lbl.setText(f"进度: 0 / {total}")

        # ── 同步实时编辑后的模型顶点缓存，确保拍摄使用当前 mesh ───────────
        self._pv_view._solidify_model_transform()

        # ── 隐藏辅助元素（轨道、网格、锚点）──────────────────────────────
        self._gizmo_visibility = self._pv_view.hide_gizmos()

        self._capture_mgr = ScreenshotManager(out_dir)
        self._capture_mgr.ensure_output_dir()

        # 预先计算所有拍摄位置（同步完成，不阻塞主线程）
        # 相机朝向 = (1,0,0)，与 get_intersection_point / _add_frustum_visualization 完全一致
        track_sys = self._pv_view.get_track_system()
        photos_per_track = self._photo_count // 4
        self._active_capture_profile = self._normal_capture_profile()
        if self._capture_subject == "building":
            all_pos = track_sys.get_building_camera_positions(
                num_positions=photos_per_track,
            )
        else:
            all_pos = track_sys.get_all_camera_positions(
                elev_deg=0.0,
                azim_deg=0.0,
                num_positions=photos_per_track,
            )

        flat = []
        for t_idx, block in enumerate(all_pos):
            positions = block.get("positions", []) if isinstance(block, dict) else block
            for p_idx, pos in enumerate(positions):
                flat.append((t_idx, p_idx, pos))

        self._capture_out_dir = out_dir
        self._capture_flat   = flat
        self._capture_total  = len(flat)
        self._capture_idx    = 0
        self._camera_pose_log = []

        self.status_lbl.setText(f"正在拍摄... 保存至: {out_dir}")

        # Step 1: 创建前绝对清理（新模型拍摄开始时强制重建离屏 plotter）
        self._pv_view._ensure_cap_plotter()

        # 彻底屏蔽鼠标/键盘事件穿透 3D 视图（防弹玻璃策略）
        self._pv_view.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._pv_view.setFocusPolicy(Qt.NoFocus)

        try:
            self._pv_view._ensure_cap_plotter()
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"[Capture] FATAL: _ensure_cap_plotter 初始化失败: {err_msg}")
            traceback.print_exc()
            self.status_lbl.setText(f"[致命错误] {err_msg[:80]}")
            self._restore_gizmos()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            return

        self._capture_one()

    def _on_capture_dataset(self):
        if not self._pv_view.has_dataset():
            QMessageBox.warning(self, "提示", "请先导入数据集三元组")
            return

        if hasattr(self, '_capture_executor') and self._capture_executor is not None:
            try:
                self._capture_executor.shutdown(wait=False)
            except Exception:
                pass
        self._capture_executor = ThreadPoolExecutor(max_workers=4)

        base = self.save_parent_dir if self.save_parent_dir else os.getcwd()
        try:
            out_dir = allocate_next_model_folder(base)
        except OSError as e:
            QMessageBox.critical(self, "错误", f"无法创建保存目录:\n{e}")
            return

        subdirs = {
            "complete_rgb": os.path.join(out_dir, "complete_rgb"),
            "incomplete_rgb": os.path.join(out_dir, "incomplete_rgb"),
            "missing_mask": os.path.join(out_dir, "missing_mask"),
            "visible_mask": os.path.join(out_dir, "visible_mask"),
        }
        for path in subdirs.values():
            os.makedirs(path, exist_ok=True)

        track_sys = self._pv_view.get_track_system()
        photos_per_track = self._photo_count // 4
        all_pos = track_sys.get_all_camera_positions(
            elev_deg=0.0, azim_deg=0.0, num_positions=photos_per_track)

        flat = []
        for t_idx, block in enumerate(all_pos):
            positions = block.get("positions", []) if isinstance(block, dict) else block
            for p_idx, pos in enumerate(positions):
                flat.append((t_idx, p_idx, pos))

        self._capture_stopping = False
        self._capture_is_dataset = True
        self._dataset_pose_log = []
        self._dataset_save_futures = []
        self._dataset_failed_outputs = []
        self._capture_out_dir = out_dir
        self._dataset_output_dirs = subdirs
        self._capture_flat = flat
        self._capture_total = len(flat)
        self._capture_idx = 0
        self._capture_mgr = ScreenshotManager(out_dir)

        self._disable_ui_for_capture()
        self.prog_bar.setValue(0)
        self.prog_lbl.setText(f"进度: 0 / {self._capture_total}")
        self._gizmo_visibility = self._pv_view.hide_gizmos()
        self._pv_view.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._pv_view.setFocusPolicy(Qt.NoFocus)
        self.status_lbl.setText(f"Dataset Mode 正在生成数据集... 保存至 {out_dir}")

        try:
            self._pv_view._destroy_dataset_plotters()
            self._pv_view._ensure_dataset_plotters()
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Dataset Mode 错误", f"无法创建数据集离屏渲染器:\n{e}")
            self._restore_gizmos()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            return

        self._capture_dataset_one()

    def _capture_dataset_one(self):
        if getattr(self, "_capture_stopping", False):
            self._restore_gizmos()
            self.status_lbl.setText("正在保存已生成的数据集图片...")
            QApplication.processEvents()
            self._capture_executor.shutdown(wait=True)
            self._collect_dataset_save_results()
            self._write_dataset_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            self.status_lbl.setText("Dataset Mode 已停止")
            return

        idx = self._capture_idx
        flat = self._capture_flat
        if idx >= len(flat):
            self._restore_gizmos()
            self.status_lbl.setText("正在保存 Dataset Mode 输出...")
            QApplication.processEvents()
            self._capture_executor.shutdown(wait=True)
            failures = self._collect_dataset_save_results()
            self._write_dataset_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            self.status_lbl.setText("Dataset Mode 拍摄完成" if not failures else "Dataset Mode 完成，但有图片保存失败")
            QMessageBox.information(
                self,
                "Dataset Mode 完成",
                f"{self._capture_total} 个相机位姿已生成四路输出\n"
                f"保存失败: {len(failures)}\n\n保存至:\n{self._capture_out_dir}",
            )
            return

        t_idx, p_idx, pos = flat[idx]
        img_name = f"track{t_idx + 1}_{p_idx + 1:02d}.png"
        self.prog_bar.setValue(idx + 1)
        self.prog_lbl.setText(f"进度: {idx + 1} / {self._capture_total}  (轨道{t_idx + 1} 第{p_idx + 1}张)")
        QApplication.processEvents()

        specs = ["complete_rgb", "incomplete_rgb", "missing_mask", "visible_mask"]

        outputs = {}
        real_pos = None
        try:
            for subdir in specs:
                img, rp = self._pv_view.capture_dataset_layer(subdir, pos)
                if rp is None:
                    raise RuntimeError("Dataset Capture camera position is missing")
                if real_pos is None:
                    real_pos = rp
                elif np.linalg.norm(np.asarray(rp, dtype=np.float64) - np.asarray(real_pos, dtype=np.float64)) > 1e-6:
                    raise RuntimeError("Dataset Capture camera mismatch across output layers")
                filepath = os.path.join(self._dataset_output_dirs[subdir], img_name)
                outputs[subdir] = os.path.join(subdir, img_name).replace("\\", "/")
                future = self._capture_executor.submit(
                    self._save_dataset_image_async,
                    img,
                    filepath,
                    self._capture_mgr,
                )
                self._dataset_save_futures.append((future, filepath))
            self._dataset_pose_log.append({
                "frame_id": img_name,
                "track_index": int(t_idx + 1),
                "photo_index": int(p_idx + 1),
                "target_pos": [float(v) for v in pos],
                "real_pos": [float(v) for v in real_pos] if real_pos else None,
                "camera": self._pv_view._camera_model_dict(),
                "outputs": outputs,
            })
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"[Dataset Capture] ERROR at track={t_idx + 1} pos={p_idx + 1}: {err_msg}")
            traceback.print_exc()
            self.status_lbl.setText(f"[Dataset Capture 异常] {err_msg[:80]}")
            self._restore_gizmos()
            QApplication.processEvents()
            self._capture_executor.shutdown(wait=True)
            self._collect_dataset_save_results()
            self._write_dataset_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            return

        self._capture_idx += 1
        QTimer.singleShot(10, self._capture_dataset_one)

    def _disable_ui_for_capture(self):
        self.btn_capture.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_import_dataset.setEnabled(False)
        self.btn_export_aligned.setEnabled(False)
        self.btn_export_folder.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_track.setEnabled(False)
        self.btn_inspect.setEnabled(False)
        for b in getattr(self, "_capture_subject_buttons", {}).values():
            b.setEnabled(False)
        for b in getattr(self, "_photo_buttons", {}).values():
            b.setEnabled(False)
        for b in self.track_btns:
            b.setEnabled(False)

    def _on_capture_tick(self, pos, t_idx, p_idx, cur):
        # 旧接口，已废弃，保留防止遗漏引用
        pass

    def _capture_one(self):
        """递归拍摄一张 + 刷新进度条 + after 调度下一张（非阻塞）。"""
        if getattr(self, "_capture_stopping", False):
            self._restore_gizmos()
            self.status_lbl.setText("正在保存已拍照片...")
            QApplication.processEvents()
            self._capture_executor.shutdown(wait=True)
            self._write_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            self.status_lbl.setText("拍摄已停止")
            return

        idx = self._capture_idx
        flat = self._capture_flat

        if idx >= len(flat):
            # 全部完成，先等待所有存图任务写完，再恢复交互
            self._restore_gizmos()
            self.status_lbl.setText("正在保存...")
            QApplication.processEvents()
            # 等待线程池中所有待提交的存图任务完成
            self._capture_executor.shutdown(wait=True)
            # ── 写入 camera_poses.txt ──────────────────────────────────────
            self._write_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            self.status_lbl.setText("拍摄完成")
            response = QMessageBox.question(
                self, "拍摄完成", f"{self._capture_total} 张照片已保存\n\n是否继续加载新的模型进行拍摄？",
                QMessageBox.Yes | QMessageBox.No)
            if response == QMessageBox.Yes:
                self._on_import()
            return

        t_idx, p_idx, pos = flat[idx]

        # ── 刷新进度条（立即可见）──────────────────────────
        self.prog_bar.setValue(idx + 1)
        self.prog_lbl.setText(f"进度: {idx + 1} / {self._capture_total}  (轨道{t_idx + 1} 第{p_idx + 1}张)")
        QApplication.processEvents()

        # ── 拍摄 & 保存截图（渲染在主线程，存图在后台线程）─────────────────
        img_name = f"track{t_idx + 1}_{p_idx + 1:02d}.png"
        try:
            if getattr(self, "_capture_subject", "cad") == "building":
                view_up = self._pv_view.get_track_system().get_building_camera_view_up(
                    t_idx
                )
            else:
                view_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            img, real_pos = self._pv_view.capture_frame(pos, view_up=view_up)
            if img is not None:
                self._camera_pose_log.append((img_name, pos, real_pos, view_up))
                print(f"[Verify] {img_name} | Target Pos: {pos} | Real Cam Pos: {real_pos}")
                # 立即提交存图任务到线程池，不等待 I/O 完成
                mgr = self._capture_mgr
                self._capture_executor.submit(
                    self._save_image_async,
                    img, t_idx + 1, p_idx + 1, mgr)
                print(f"[Capture] {img_name} queued")
            else:
                print(f"[Capture] WARNING: capture_frame returned None for {img_name}")
        except Exception as e:
            import traceback
            err_msg = str(e)
            print(f"[Capture] ERROR at track={t_idx + 1} pos={p_idx + 1}: {err_msg}")
            traceback.print_exc()
            self.status_lbl.setText(f"[拍摄异常] {err_msg[:80]}")
            self._restore_gizmos()
            QApplication.processEvents()
            self._capture_executor.shutdown(wait=True)
            self._write_camera_poses()
            self._cleanup_capture()
            self._restore_mouse_interaction()
            return

        self._capture_idx += 1
        QTimer.singleShot(10, self._capture_one)

    def _restore_gizmos(self):
        """安全恢复辅助元素可见性，包含异常保护。"""
        try:
            if hasattr(self, "_gizmo_visibility") and self._gizmo_visibility:
                self._pv_view.restore_gizmos(self._gizmo_visibility)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            self._gizmo_visibility = {}

        try:
            self._pv_view._render_scene()
        except Exception:
            pass

    def _save_image_async(self, img_array, track_idx, photo_idx, mgr):
        """在线程池中执行：将 numpy 数组保存为 PNG 文件。"""
        try:
            if mgr is None:
                print(f"[Capture] SKIP track{track_idx}_{photo_idx:02d} (manager already cleaned up)")
                return
            saved = mgr.save_screenshot_from_array(img_array, track_idx, photo_idx)
            print(f"[Capture] track{track_idx}_{photo_idx:02d} -> {saved}")
        except Exception:
            import traceback; traceback.print_exc()

    def _save_dataset_image_async(self, img_array, filepath, mgr):
        if mgr is None:
            raise RuntimeError(f"Screenshot manager already cleaned up for {filepath}")
        saved = mgr.save_array_to_path(img_array, filepath)
        print(f"[Dataset Capture] -> {saved}")
        return saved

    def _collect_dataset_save_results(self):
        failures = []
        for future, filepath in getattr(self, "_dataset_save_futures", []):
            try:
                future.result()
            except Exception as ex:
                failures.append({
                    "path": os.path.abspath(filepath),
                    "error": str(ex),
                })
        self._dataset_failed_outputs = failures
        self._dataset_save_futures = []
        if failures:
            print(f"[Dataset Capture] ERROR: {len(failures)} image save task(s) failed")
            for item in failures[:10]:
                print(f"  - {item['path']}: {item['error']}")
        return failures

    def _write_camera_poses(self):
        """将所有拍摄的相机位姿记录写入 camera_poses.json。"""
        if not self._camera_pose_log:
            return
        out_dir = getattr(self, "_capture_out_dir", None)
        if out_dir is None:
            print("[Capture] WARNING: _capture_out_dir is None, skipping camera_poses.json")
            return
        json_path = os.path.join(out_dir, "camera_poses.json")
        try:
            frames = []
            for img_name, target_pos, real_pos, view_up in self._camera_pose_log:
                tp = [float(v) for v in target_pos]
                rp = [float(v) for v in real_pos] if real_pos else None
                vu = [float(v) for v in view_up]
                frames.append({
                    "file_path": img_name,
                    "target_pos": tp,
                    "real_pos": rp,
                    "view_up": vu,
                })
            data = {
                "capture_profile": dict(self._active_capture_profile or {}),
                "frames": frames,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"[Capture] camera_poses.json saved ({len(frames)} entries) -> {json_path}")
        except Exception:
            import traceback; traceback.print_exc()
            print(f"[Capture] ERROR: failed to write camera_poses.json")
        finally:
            self._camera_pose_log.clear()

    def _write_dataset_camera_poses(self):
        if not self._dataset_pose_log:
            return
        out_dir = getattr(self, "_capture_out_dir", None)
        triplet = self._pv_view._dataset_triplet
        if out_dir is None or triplet is None:
            print("[Dataset Capture] WARNING: missing output dir or triplet, skipping metadata")
            return

        poses_path = os.path.join(out_dir, "camera_poses.json")
        meta_path = os.path.join(out_dir, "metadata.json")
        try:
            with open(poses_path, "w", encoding="utf-8") as f:
                json.dump({
                    "camera_model": self._pv_view._camera_model_dict(),
                    "failed_outputs": list(getattr(self, "_dataset_failed_outputs", [])),
                    "frames": self._dataset_pose_log,
                }, f, indent=2)

            meta = {
                "dataset_mode": triplet.mode,
                "coordinate_rule": (
                    "shared_triplet computes one rigid alignment from complete.obj only, "
                    "then applies the same rotation and translation to complete/incomplete/removed. "
                    "Scale is never changed, and incomplete/removed never compute their own transform."
                    if triplet.mode == SHARED_TRIPLET
                    else "prealigned_triplet is the default Dataset Mode. Normal mode first prepares "
                         "and freezes aligned_complete.obj; Blender splits that frozen model into "
                         "complete/incomplete/removed while preserving one world coordinate system; "
                         "Dataset Mode only reads the frozen triplet and applies no normalize/recenter/"
                         "rescale/reorient transform."
                ),
                "alignment_report": dict(getattr(triplet, "alignment_report", {}) or {}),
                "app_coordinate_mapping": (
                    "Before rendering, all three meshes are equally mapped from aligned OBJ "
                    "semantic coordinates to app capture coordinates by _obj_to_pv_coords: "
                    "(x, y, z) -> (z, x, y). This is a shared coordinate convention conversion, "
                    "not a per-mesh normalization/rescale step."
                ),
                "camera_pose_coordinate_system": "app_capture_coordinates_after_shared_obj_to_pv_mapping",
                "camera_model": self._pv_view._camera_model_dict(),
                "visible_mask_source": "incomplete",
                "missing_mask_rule": (
                    "Render removed as white with incomplete as black occluder on a black background, "
                    "so the mask represents the currently visible missing region."
                ),
                "validation_warnings": list(triplet.validation_warnings),
                "validation_scope": (
                    "Bounds checks are sanity checks only. They catch obvious coordinate-system "
                    "mismatches but do not mathematically prove that all parts came from the same source mesh."
                ),
                "source_root": triplet.root_dir,
                "source_files": {
                    "complete": triplet.complete.path,
                    "incomplete": triplet.incomplete.path,
                    "removed": triplet.removed.path,
                },
                "outputs": {
                    "complete_rgb": "complete_rgb",
                    "incomplete_rgb": "incomplete_rgb",
                    "missing_mask": "missing_mask",
                    "visible_mask": "visible_mask",
                },
                "frame_count": len(self._dataset_pose_log),
                "failed_outputs": list(getattr(self, "_dataset_failed_outputs", [])),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[Dataset Capture] metadata saved -> {out_dir}")
        except Exception:
            import traceback; traceback.print_exc()
            print("[Dataset Capture] ERROR: failed to write metadata")
        finally:
            self._dataset_pose_log.clear()

    def _cleanup_capture(self):
        # 四大退出路径 3/4: _cleanup_capture 是所有退出路径的共同终点
        # （正常完成/手动停止/异常）都在此统一销毁离屏 plotter
        self._pv_view._destroy_cap_plotter()
        self._pv_view._destroy_dataset_plotters()
        self._capture_flat   = []
        self._capture_idx    = 0
        self._capture_mgr    = None
        self._capture_is_dataset = False
        self._active_capture_profile = None
        self._enable_ui_after_capture()

    def _restore_mouse_interaction(self):
        """拍摄结束后安全恢复鼠标交互（防弹玻璃撤销）。"""
        try:
            self._pv_view.setAttribute(Qt.WA_TransparentForMouseEvents, False)
            self._pv_view.setFocusPolicy(Qt.StrongFocus)
        except Exception:
            import traceback; traceback.print_exc()

    def _enable_ui_after_capture(self):
        self.btn_capture.setEnabled(True)
        self.btn_track.setEnabled(bool(self._pv_view.has_model()))
        self.btn_inspect.setEnabled(bool(self._pv_view.has_model()) and not self._pv_view.has_dataset())
        self.btn_stop.setEnabled(False)
        self.btn_import.setEnabled(True)
        self.btn_import_dataset.setEnabled(True)
        self.btn_export_aligned.setEnabled(bool(self._pv_view.has_model()) and not self._pv_view.has_dataset())
        self.btn_export_folder.setEnabled(True)
        self.btn_folder.setEnabled(True)
        subject_enabled = not self._pv_view.has_dataset()
        for b in getattr(self, "_capture_subject_buttons", {}).values():
            b.setEnabled(subject_enabled)
        for b in getattr(self, "_photo_buttons", {}).values():
            b.setEnabled(True)
        for b in self.track_btns:
            b.setEnabled(bool(self._pv_view.has_model()))

    def _on_stop(self):
        self._capture_stopping = True
        self.btn_stop.setEnabled(False)
        self.status_lbl.setText("正在停止，等待已拍照片保存...")

    def _update_buttons(self):
        has = self._pv_view.has_model()
        cap = (getattr(self, "_capture_idx", 0) > 0) or (getattr(self, "_capture_flat", []) != [])
        self.btn_import.setEnabled(not cap)
        self.btn_import_dataset.setEnabled(not cap)
        self.btn_export_aligned.setEnabled(bool(has) and not cap and not self._pv_view.has_dataset())
        self.btn_export_folder.setEnabled(not cap)
        self.btn_folder.setEnabled(not cap)
        self.btn_capture.setEnabled(bool(has) and not cap)
        self.btn_track.setEnabled(bool(has) and not cap)
        self.btn_inspect.setEnabled(bool(has) and not cap and not self._pv_view.has_dataset())
        self.btn_stop.setEnabled(cap)
        subject_enabled = not cap and not self._pv_view.has_dataset()
        for b in getattr(self, "_capture_subject_buttons", {}).values():
            b.setEnabled(subject_enabled)
        for b in getattr(self, "_photo_buttons", {}).values():
            b.setEnabled(not cap)
        pick_s = bool(has) and not cap
        for b in self.track_btns:
            b.setEnabled(pick_s)

    def closeEvent(self, event):
        self._capture_stopping = True
        self._pv_view._destroy_cap_plotter()
        self._pv_view._destroy_dataset_plotters()
        self._restore_gizmos()
        self._restore_mouse_interaction()
        self._pv_view.close()
        super().closeEvent(event)


# ── QtInteractor 的 close() 扩展（关闭渲染窗口）──────────────────────
def _pvview_close(self):
    try:
        plotter = self  # QtInteractor 本身即 Plotter
        plotter.close()
    except Exception:
        pass

PyVistaView.close = _pvview_close


def run_app():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    run_app()
