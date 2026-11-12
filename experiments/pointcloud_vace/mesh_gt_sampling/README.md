# 全部OBJ离线生成完整几何GT点云

2026-11-13（UTC+09）。用户授权先下载作者代码、用临时构造网格调通程序，再后台启动全部OBJ转换；启动确认后交接，不等待全量结束，不自动验证或训练。

## 输入、输出与实现

- 数据项目：`/mnt/windowsE/chuanjun/building_wan/data/pointcloud_vace`。
- 输入：`sample-data-organized/mesh/`下全部OBJ，启动前实际计数15,890；不按pointcloud配对筛选，原数据只读。
- 输出：`mesh_gt_pointcloud/<原编号>.ply`；每栋16,384点，binary_little_endian，XYZ double、nx/ny/nz float。
- 作者代码克隆到数据项目的`third_party/APC2Mesh`，commit `6cb68e6dbf15e2aa69463440d1153f9abd1d7b8b`。阅读`dataset_pcc.py`的`__getitem__`及`point_ops`，没有导入或实例化会删除不配对mesh的CustomDataset。
- 沿用作者“50,000个均匀表面候选点→GPU FPS→同索引取点与法向”的方法；公开代码的16348在本项目统一为16,384。
- 作者旧构建包含sm_37等旧架构；当前CUDA12.4对该架构的兼容性编译调用报错。按本轮授权改用PyTorch3D `sample_farthest_points`官方CUDA后端。
- PyTorch3D源码位于数据项目的`third_party/pytorch3d`，commit `8698dfb41d95aae8bba89445bb34cedc7c2538f6`、版本0.7.9。仅编译官方FPS的CPU/CUDA源文件和所需Python接口，不安装其他算子；不改官方FPS算法。

`build_fps.py`把所需接口安装到数据盘的`mesh_gt_runtime/deps`，复用scj现有PyTorch2.11.0+cu128与CUDA12.4编译器，目标A6000 sm86。Ninja1.13.2和Trimesh5.2.0仅安装到该目录，不改现有Conda环境。依赖缓存、CUDA缓存、编译和临时文件、日志均在数据项目的`mesh_gt_runtime/`中。

## 采样和精度

Trimesh只读加载OBJ且关闭自动process；以精确顶点坐标和无序三角形顶点集合识别重复面，保留首次面绕序，去除零面积面。以剩余面的float64实际面积为权重，使用均匀重心坐标采50,000点；法向取对应三角面的单位法向。

数字建筑ID直接用其整数作为NumPy PCG64随机种子；非数字ID使用稳定BLAKE2b 64位种子。顶点与候选点均float64，先以float64包围盒中心平移，GPU仅接收局部float32坐标。FPS固定从候选索引0开始；其返回索引用于读取原float64世界坐标候选点及同批三角面法向，GPU坐标不作为输出来源。

`sample_mesh_gt.py`逐栋处理：先写同目录临时文件，关闭后原子替换正式PLY；恢复时跳过本程序已完整写出的同格式16,384点文件。单栋失败记录ID、错误和traceback，继续下一栋；日志和summary记录成功、跳过、失败数量。一次仅一个生成进程，复用项目GPU锁；没有VACE训练或推理。

## 已完成的程序调试

仅在程序内构造百万量级世界坐标上的简单立方体，附加一个重复面与一个零面积面，再临时写OBJ，完整执行一次50,000→16,384的CUDA FPS和二进制PLY读写。14面清理为12面，double XYZ和float法向读回与所选原数组一致。2026-11-13T02:23:32+09:00通过，耗时0.881秒，实际设备NVIDIA RTX A6000；调试网格/PLY随临时目录自动删除，记录在数据项目`mesh_gt_runtime/logs/self_test.json`。未从正式数据挑四栋，未做质量验收、预览或视频渲染。

## 后台启动与续跑

Python固定为`/home/shi/miniconda3/envs/scj/bin/python`。首次启动或任务明确结束/中断后的断点续跑入口：

```bash
/home/shi/miniconda3/envs/scj/bin/python -B experiments/pointcloud_vace/mesh_gt_sampling/scripts/launch.py
```

启动器创建独立tmux和数据盘日志，返回实际session、PID、日志、summary和输出路径。关闭Codex或SSH不影响生成；不要把启动入口当日常状态查询。批任务结束自动把summary更新为complete或complete_with_failures并写入真实计数；本次不持续监控、不等待结束、不自动执行后续验证或训练。完成/失败后的正式报告按项目规则留后续接手收尾。


## 本次实际启动

tmux `mesh_gt_20261113_022544`，PID1768992，启动时间2026-11-13T02:25:45+09。一次启动核查：02:26:30+09成功77/15,890、跳过0、失败0；进程正常。

- 日志：`/mnt/windowsE/chuanjun/building_wan/data/pointcloud_vace/mesh_gt_runtime/logs/generate_20261113_022544.log`
- 计数与终态：同目录`generate_20261113_022544.summary.json`
- 输出：`/mnt/windowsE/chuanjun/building_wan/data/pointcloud_vace/mesh_gt_pointcloud`

既有298项CPU回归实跑73.307秒，其中2项在临时目录清理时报DirectoryNotEmpty；错误原文在数据盘cpu_checks.log，遗留目录已清理。GT合成网格完整GPU调用、shell语法、Python编译和diff检查通过；没有把全局CPU回归写成全通过，没有GPU训练、正式数据抽样验收或视频渲染。后续以本批summary和日志读取状态，不重启已成功的输出。
