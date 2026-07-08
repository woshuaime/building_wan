# Python_3D_Scanner 项目上下文（给 ChatGPT 网页版使用）

## 项目目标

这个项目是一个面向“建筑模型补全数据集制造”的工具。核心流程是：

1. 输入完整 OBJ 建筑模型。
2. 软件自动摆正模型：模型中心在四轨中心，底面平行灰色虚拟平面，顶面朝四轨交叉/拍摄出发点。
3. 围绕模型用四条轨道拍摄多视角图片。
4. 后续计划从完整模型制造残缺模型，再用相同相机位姿渲染：
   - 完整图 `complete_rgb`
   - 残缺图 `incomplete_rgb`
   - 缺失区域 `missing_mask`
   - 可见区域 `visible_mask`
5. 训练中间图像补全模型，把残缺照片补好。
6. 将补好的多视角照片导入开源多视角重建软件，重建完整模型。

## 用户的关键业务标准

导入 OBJ 后必须满足：

- 模型底面平行灰色虚拟平面。
- 模型中心位于四条轨道中心。
- 模型顶面朝向四轨交叉的拍摄出发点。
- 导入后默认视图要能同时看到模型、轨道和出发点。
- 调整模型时要像 MeshLab 一样实时跟手。

## 当前已完成的关键修复

### 1. 导入姿态修复

文件：

- `src/core/obj_loader.py`
- `src/ui/tk_widget.py`

现在导入模型时会优先识别模型原始最低 Z 附近的真实底面边界，作为 `mesh_bottom_boundary`。

对 `26.obj` 的问题已经修复：

- 修复前：可能选择 `ransac_point_cloud` 的近似平面，导致底面略歪。
- 修复后：选择 `mesh_bottom_boundary`，`ground_axis=1.0`，`bottom_angle=0.0`。

批量验证：

- 路径：`D:\scj\down_load\三七\建筑物模型`
- OBJ 总数：`2390`
- 失败数：`0`
- `mesh_bottom_boundary`: `2380`
- `mesh_face`: `10`

弱证据但通过的模型：

- `341.obj`
- `742.obj`
- `1182.obj`
- `1635.obj`
- `1647.obj`
- `1701.obj`
- `1800.obj`
- `1905.obj`
- `2057.obj`
- `2248.obj`

这些不是失败，只是底面边界证据没有最强。

### 2. 导入默认视图修复

文件：

- `src/ui/tk_widget.py`

之前导入后相机硬编码在 `(0, -15, 0)`，距离太近，界面会变成模型撑满画面，看不到出发点。

现在改成自动概览视角：

- 按轨道半径自动计算相机距离。
- 默认斜向俯视。
- 能同时看到模型、轨道和橙色出发点。
- 只影响界面初始观察，不影响拍摄相机、不影响模型姿态、不影响实时调整。

### 3. 拍摄导出的 JSON 当前内容

拍摄完成后生成：

- `camera_poses.json`

当前格式：

```json
{
  "frames": [
    {
      "file_path": "track1_01.png",
      "target_pos": [x, y, z],
      "real_pos": [x, y, z]
    }
  ]
}
```

字段含义：

- `file_path`：图片文件名。
- `target_pos`：计划相机位置，也就是轨道采样点。
- `real_pos`：实际 PyVista 渲染相机位置。

当前还缺：

- 相机完整外参矩阵。
- 相机内参，例如 FOV、分辨率、near/far。
- focal point。
- view up。
- 模型 ID、原始 OBJ 路径、标准化信息。
- split 信息，例如 train/val/test。

## 数据集设计方向

用户现在只有完整模型，还没有残缺模型。因此推荐的数据制造流程是：

```text
完整 OBJ
  -> 参数化三维退化，制造残缺 OBJ
  -> 同一批相机位姿渲染完整图和残缺图
  -> 生成 missing_mask 和 visible_mask
  -> 训练图像补全模型
  -> 补好的多视角图像导入重建软件
  -> 得到完整模型
```

推荐残缺生成方法：

- 不建议只在 2D 图片上随机涂 mask。
- 应该在 3D 模型空间制造残缺，保证多视角一致。

可选退化方式：

1. 随机删连续面片。
2. 用球体、盒体、平面等 3D 几何体切掉一部分，推荐。
3. 结构化建筑破损，例如屋顶、墙面、边角优先，后续可做。

论文中建议称为：

- 基于三维几何约束的参数化退化方法。
- Parametric 3D Damage Simulation。
- Geometry-Guided Synthetic Incompletion。
- Controlled Random Degradation。

核心解释：

> 通过受控随机退化在三维空间中生成残缺区域，使同一残缺结构在所有视角下保持一致，避免二维随机遮挡导致的多视角不一致问题。

## mask 生成建议

最推荐保留三份模型：

```text
complete.obj      完整模型
incomplete.obj    残缺模型
removed.obj       被删除的缺失部分
```

然后每个相机视角渲染：

```text
complete_rgb/
incomplete_rgb/
missing_mask/
visible_mask/
camera_poses.json
metadata.json
```

最准确的缺失 mask：

```text
missing_mask = render_mask(removed.obj, same_camera)
```

备选方法：

```text
missing_mask = complete_mask & (~incomplete_mask)
```

但这个方法在残缺后背面露出来时可能漏掉缺失区域，所以 `removed.obj` 渲染 mask 更稳。

## 建议下一步开发

优先做“标准数据集生成模式”：

1. 导入完整 OBJ。
2. 自动制造一个或多个残缺 OBJ。
3. 保存 `complete.obj`、`incomplete.obj`、`removed.obj`。
4. 同一相机轨迹渲染：
   - `complete_rgb`
   - `incomplete_rgb`
   - `missing_mask`
   - `visible_mask`
5. 增强 `camera_poses.json`：
   - `file_path`
   - `camera_position`
   - `focal_point`
   - `view_up`
   - `fov`
   - `resolution`
   - `camera_to_world`
   - `world_to_camera`
6. 新增 `metadata.json`：
   - `model_id`
   - `source_obj`
   - `normalizer_source`
   - `ground_axis`
   - `scale`
   - `damage_type`
   - `damage_ratio`
   - `removed_face_count`

## 当前重要文件

- `src/ui/tk_widget.py`：主界面、导入、拍摄、相机、JSON 写入。
- `src/core/obj_loader.py`：OBJ 加载与规范化、底面识别。
- `src/core/track.py`：四轨相机位置生成。
- `src/utils/screenshot.py`：截图保存。
- `scripts/validate_import_pose.py`：只读导入姿态回归验证脚本。

## 给 ChatGPT 网页版的建议提问

可以把这份上下文粘到 ChatGPT 后问：

> 基于以上项目背景，请帮我设计一个“完整 OBJ -> 残缺 OBJ -> 多视角完整/残缺图像/mask -> 图像补全训练数据集”的实现方案，并给出适合写进论文的方法描述。

