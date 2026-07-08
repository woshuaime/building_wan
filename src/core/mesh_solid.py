"""OBJ 面片三角化与三角索引缓存（用于实体网格绘制）"""
import numpy as np


def _face_to_tri_indices(face, n_verts):
    idxs = [int(i) for i in face if 0 <= int(i) < n_verts]
    if len(idxs) < 3:
        return []
    if len(idxs) == 3:
        return [tuple(idxs)]
    return [(idxs[0], idxs[k], idxs[k + 1]) for k in range(1, len(idxs) - 1)]


def build_triangle_index_array(n_verts, faces, max_triangles=120000):
    """
    返回形状 (N,3) 的顶点索引数组，每行一个三角形。
    面数过多时均匀抽稀以减轻 matplotlib 负担。
    """
    n = int(n_verts)
    tris = []
    for face in faces:
        tris.extend(_face_to_tri_indices(face, n))
    if not tris:
        return None
    if len(tris) > max_triangles:
        idx = np.linspace(0, len(tris) - 1, max_triangles, dtype=np.int64)
        tris = [tris[i] for i in idx]
    return np.array(tris, dtype=np.int64)
