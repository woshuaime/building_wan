# BuildingFiller 源 465 交接与 VACE 最小实验准备

核查日期：2026-10-18。完成数据接收检查、CPU 图像/Mask 预处理检查与方案冻结。没有加载模型权重、运行 GPU 推理、下载模型或修改依赖。检查报告时间为 UTC，来源证据保留原时区。

## 1. 三台设备与执行位置

| 设备 | 职责 | 路径与访问范围 |
|---|---|---|
| 个人 Windows 电脑 | 独立 BuildingFiller 项目，制作三元组和对齐渲染 | 其历史 E 盘路径仅为来源证据 |
| 实验室 Windows 主机 | UU 远程控制，PowerShell/OpenSSH 上传 | 用户报告源目录为 `D:\P01_465` |
| 实验室 Linux 服务器 | SSH 运行 building_wan | 主机 `vccip`、用户 `scj`、项目 `/mnt/windowsE/chuanjun/building_wan` |

本次 Codex 实际在第三台设备执行，`uname=Linux`，SSH 会话存在。不能读取 Windows 主机的 `C:\Users\DELL\.ssh\config` 或 `D:\P01_465`。服务器 `/mnt/windowsD` 是空目录，没有挂载 Windows 主机 D 盘。SSH 会话中的地址不能证明 Windows 的 Host 别名、转发端口或认证方式。

服务器已存在下列三个文件，本轮直接验证，没有执行 SCP 或覆盖原件：

```text
data/buildingfiller_import/P01_465/
  BuildingFiller_P01_465_512_v1.zip
  BuildingFiller_P01_465_512_v1.zip.sha256
  Stage1_Closeout_Report.md
```

Windows 端源目录、OpenSSH 客户端和实际配置尚未在该端实测，检查命令见第 4 节。服务器已有同包，当前无需重复上传。

## 2. BuildingFiller 来源

BuildingFiller 是独立数据制作项目：对 2390 栋建筑 OBJ 筛选，按冻结 QF-1.3 制作 GT/Damaged/Removed 三元组，再渲染多视角 RGB、Exact，保存相机、元数据、哈希和质量证据。

用户报告和已接收收尾记录一致：8 个 Pilot 全部人工接受，A1/A2/A3/A4 为 3/2/2/1，Medium/Large 为 6/2。**本次在服务器独立验证的载荷仅为 P01 的 512 子集**，没有重新验证另外七例或 768/1024 图片。

P01：源/家族 465，`sample_0000`，稳定身份 `b465_2e6fc746`，A1 Medium，删除原表面积比例 `0.056792346128503535`。源 OBJ SHA256：`2e6fc7465b7e6af110563ae12a9ae5080b382a6657751e2c2aa5e7fffb1c1b52`。

`human_QA=accepted`、`generation_evidence_verified=true`、`training_ready=false` 同时保留。原 sample_meta 内旧 pending 是冻结快照，当前接受决定由独立证据给出，不修改原件来统一显示。人工接受不代表 VACE 补全成功。

## 3. 实际完整性检查

- ZIP：1,319,395 bytes，SHA256 `5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8`，匹配相邻清单和来源收尾报告。
- ZIP CRC 通过；包内 263 文件，`SHA256SUMS` 覆盖 262 文件，另一项为清单自身；manifest 的文件集合、大小、哈希均一致。
- 81 GT + 81 Damaged + 81 Exact，243 张 512×512 PNG 全部实际解码通过；RGB/RGB/L 通道正确，Exact 实际像素只有 0/255。
- manifest、CSV、相机索引、角度、时间、相机对原元数据的哈希绑定及三份 OBJ 哈希均一致。
- 81 帧包含 80 个不同位置，080 重复 000，三个角色的首尾图片哈希及相机闭合姿态通过。
- `046–072` 共 27 张 Exact 全黑，属于合法遮挡，保持原样。
- 已在新审计目录复制解压，并审阅、执行随包标准库校验器，结果 `status=passed`。
- 已查看 `000/020/040/060/046/005/013/021` 三联联系图，检查接收方向与遮挡关系；不作为模型效果验收。

新审计目录及可复查证据：

```text
outputs/buildingfiller_handoff/P01_465_20261018/
  audit.json
  supplier_verify.json
  environment_and_adapter_check.json
  frame_selection_17.json
  contact_sheet.png
  package/BuildingFiller_P01_465_512_v1/...
```

以下命令只读输入，不启动模型：

```bash
cd /mnt/windowsE/chuanjun/building_wan
/home/shi/miniconda3/envs/scj/bin/python -B scripts/audit_buildingfiller_package.py \
  data/buildingfiller_import/P01_465/BuildingFiller_P01_465_512_v1.zip \
  --expected-sha256 5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8
/home/shi/miniconda3/envs/scj/bin/python -B \
  outputs/buildingfiller_handoff/P01_465_20261018/package/BuildingFiller_P01_465_512_v1/verify_package.py
```

审计工具默认不写文件。加 `--output-dir <全新目录>` 才保存报告、联系图和解压副本，已有目录拒绝。它不执行 ZIP 内的代码，随包校验器在本次审阅后另行执行。

## 4. Windows PowerShell 检查与传输

**只在实验室 Windows 主机的 PowerShell 执行以下整段。** 复用现有 SSH Host 配置，不猜 IP/端口/账号；已有同包时只校验后退出。复制 ZIP 和校验清单，不移动或删除 Windows 原件。尚未在 Windows 实测这段代码。

```powershell
$ErrorActionPreference = 'Stop'
$SourceDir = 'D:\P01_465'
$ZipName = 'BuildingFiller_P01_465_512_v1.zip'
$ExpectedHash = '5a966a269866a1423c1efa38937fcce6ec6984f8fda78c35395abccab83d5bb8'
if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
    throw "Windows 源目录不存在：$SourceDir"
}
Get-ChildItem -LiteralPath $SourceDir -Force
Get-ChildItem -LiteralPath $SourceDir -Recurse -File |
    Group-Object Extension | Select-Object Name, Count
$SshExe = (Get-Command ssh.exe -CommandType Application -ErrorAction Stop).Source
$ScpExe = (Get-Command scp.exe -CommandType Application -ErrorAction Stop).Source
& $SshExe -V
$SshConfigPath = Join-Path $env:USERPROFILE '.ssh\config'
if (Test-Path -LiteralPath $SshConfigPath) {
    Select-String -LiteralPath $SshConfigPath -Pattern '^\s*Host\s+'
}
$ServerAlias = Read-Host '输入目前可正常连接服务器的 SSH Host 别名'
if ($ServerAlias -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') {
    throw '请填写 SSH 配置里的单个 Host 别名'
}
$EffectiveConfig = @(& $SshExe -G $ServerAlias)
if ($LASTEXITCODE -ne 0) { throw '无法解析 SSH 配置' }
$EffectiveConfig | Select-String '^(hostname|user|port|identityfile|identitiesonly|proxyjump|proxycommand|preferredauthentications) '
$ZipPath = Join-Path $SourceDir $ZipName
$ShaPath = "$ZipPath.sha256"
$ShaMatch = [regex]::Match((Get-Content -LiteralPath $ShaPath -Raw).Trim(), '^([a-fA-F0-9]{64})\s+\*?(.+)$')
if (-not $ShaMatch.Success -or $ShaMatch.Groups[2].Value -ne $ZipName) {
    throw 'SHA256 清单格式或文件名不符'
}
$LocalHash = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($LocalHash -ne $ExpectedHash -or $LocalHash -ne $ShaMatch.Groups[1].Value.ToLowerInvariant()) {
    throw '本机 ZIP 与冻结 P01 哈希不符，停止上传'
}
$ProjectDir = '/mnt/windowsE/chuanjun/building_wan'
$RemoteParent = "$ProjectDir/data/buildingfiller_import"
$RemoteCase = "$RemoteParent/P01_465"
& $SshExe -v $ServerAlias "test -d '$ProjectDir' && test -w '$ProjectDir/data'"
if ($LASTEXITCODE -ne 0) { throw 'SSH、项目路径或 data 写权限检查失败，查看上方具体错误' }
$RemoteState = @(& $SshExe $ServerAlias "if test -e '$RemoteCase'; then printf 'EXISTS\n'; else printf 'ABSENT\n'; fi")
if ($LASTEXITCODE -ne 0) { throw '无法检查接收目录' }
if ($RemoteState -contains 'EXISTS') {
    $RemoteHashLines = @(& $SshExe $ServerAlias "sha256sum -- '$RemoteCase/$ZipName'")
    if ($LASTEXITCODE -ne 0) { throw '同名接收目录已存在但 ZIP 不完整，保留现场，禁止覆盖' }
    $RemoteHash = (($RemoteHashLines -join "`n").Trim() -split '\s+')[0]
    if ($RemoteHash -ne $LocalHash) { throw '服务器同名 ZIP 哈希不同，禁止覆盖，应另用独立接收目录' }
    & $SshExe $ServerAlias "cd -- '$RemoteCase' && sha256sum -c -- '$ZipName.sha256'"
    if ($LASTEXITCODE -ne 0) { throw '服务器 SHA256 清单验证失败，保留现场' }
    Write-Host "服务器已有同一份数据，无需再次 SCP：$RemoteCase"
    return
}
& $SshExe $ServerAlias "mkdir -p -- '$RemoteParent' && mkdir -- '$RemoteCase'"
if ($LASTEXITCODE -ne 0) { throw '接收目录创建失败或已被占用，禁止继续上传' }
$UploadFiles = @($ZipPath, $ShaPath)
$CloseoutPath = Join-Path $SourceDir 'Stage1_Closeout_Report.md'
if (Test-Path -LiteralPath $CloseoutPath -PathType Leaf) { $UploadFiles += $CloseoutPath }
& $ScpExe @UploadFiles "${ServerAlias}:$RemoteCase/"
if ($LASTEXITCODE -ne 0) { throw 'SCP 失败，保留源数据和接收现场，不要直接重跑覆盖' }
& $SshExe $ServerAlias "cd -- '$RemoteCase' && sha256sum -c -- '$ZipName.sha256'"
if ($LASTEXITCODE -ne 0) { throw '传输后 ZIP 哈希不符，停止后续操作' }
Write-Host "已复制并检查 ZIP：$RemoteCase；下一步在 Linux 执行第 3 节检查"
```

`ssh -G` 给出实际主机、用户、端口、密钥和代理配置，包含 Include 最终结果；`ssh -v` 的 `Authenticated ... using ...` 给出实际认证方式。密码认证照常提示密码。

失败时先保留错误：客户端未找到查 `Get-Command`；无法解析名称查 Host/Include；超时或拒绝连接查有效地址/端口；`Permission denied` 区分 SSH 认证和目录写权限；SFTP 子系统错误保留日志再定位。不得自动安装软件、修改 SSH 服务或换协议。传输中断后保留接收目录，重新操作应先核对内容或改用全新目录。

## 5. VACE 环境与缺失项

| 项目 | 本次核查结果 |
|---|---|
| Python | `/home/shi/miniconda3/envs/scj/bin/python`，3.10 环境 |
| PyTorch/CUDA | `2.11.0+cu128` / 12.8，CUDA 可用 |
| GPU | 单张 RTX A6000，总显存约 47.4 GiB，探测时空闲约 47.1 GiB |
| 内存/磁盘 | 125 GiB RAM，探测时可用约 122 GiB；项目盘可用约 365 GiB |
| 驱动 | 内核模块 580.95.05；本次 PATH 无 nvidia-smi，但 CUDA 探测成功 |
| DiffSynth | 2.1.5，editable 指向本项目 `code/DiffSynth-Studio`，离线 pipeline 导入成功 |
| 版本追溯 | vendor 无独立 .git；父项目 HEAD `8a3720e82ab177543bd9428d2f43bbe5f304ea0f`；相关文件最后提交 `6c2324984124078f4722cabb3caec75bd263ce28`；文件 SHA256 记入环境报告 |
| 依赖 | torch/transformers/accelerate/safetensors/Pillow/numpy/imageio/einops/sentencepiece 等版本已记录，未改依赖 |
| VACE 权重 | **项目和可访问缓存未找到**；现有 T2V DiT 825 键不含 VACE 分支 |
| 公用组件 | T5 BF16 safetensors、Wan2.1 VAE safetensors、UMT5 tokenizer 已存在；未完整加载或重算大权重哈希 |
| 项目 VACE 执行器 | 尚未实现；当前项目训练/验证入口为 T2V，不直接支持 P01 条件补全 |

模型搜索范围：项目 `models/`，当前 `scj` 用户 Hugging Face/ModelScope 缓存。其他用户 `/home/shi/.cache` 无读取权限，不能断言整台服务器没有 VACE；如权重另放他处，需要提供可读实际路径。

第三方已有正式/低显存示例和 `vace_video`、`vace_video_mask`、`vace_reference_image`、`vace_scale` 接口。本地入口：

```text
code/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.1-VACE-1.3B.py
code/DiffSynth-Studio/examples/wanvideo/model_inference_low_vram/Wan2.1-VACE-1.3B.py
code/DiffSynth-Studio/diffsynth/pipelines/wan_video.py
```

示例会按模型 ID 下载权重/公用组件及示例数据，**不能直接执行**。未来项目执行器仅用 `ModelConfig(path=<明确本地文件>, skip_download=True)`，设置 `DIFFSYNTH_SKIP_DOWNLOAD=True`、`HF_HUB_OFFLINE=1`；加载后检查 `pipe.vace is not None` 和 DiT/T5/VAE 均存在。不能用 T2V 或 Preview 替代正式 VACE。硬件满足尝试 17 帧的前提，但真实加载、峰值显存、VAE 编码和推理尚未验证。

## 6. 可复现的 17 帧方案

冻结配置：[configs/vace_p01_465_17f_v1.json](../configs/vace_p01_465_17f_v1.json)。这是待执行方案，不是成功运行记录。

### 帧长与分辨率

本地 pipeline 宽高为 16 的倍数，帧数为 `4n+1`；不满足会向上调整。17=`4×4+1` 与 512×512 原样通过。官方 1.3B 常规推荐约为 81×480×832；方形 512 是本地代码接受的小样本测试形状，效果尚未验证。[VACE 官方说明](https://github.com/ali-vilab/VACE/blob/main/README.md)、[Wan 官方帧长合同](https://github.com/Wan-Video/Wan2.1/blob/main/wan/vace.py)。

取源 `005–021` 连续 17 帧，步长 1，重编号输出 `000–016`。确定性选择规则：在 80 个非重复位置中最大化连续 17 帧的 Exact 白像素总数，平局取最早窗口。本例为 204609，17 帧均有可见缺损；角度 22.5°–94.5°，源时间 0.3125–1.3125 秒，16 fps。短弧为 72°，不能据此验收完整环绕稳定性。

三个角色使用同一 manifest 帧对象与相机索引，禁止分别排序或均匀跳采。GT RGB 仅放独立评估清单，不传入参考图、视频初始化或模型条件。Exact 是 GT-oracle 标签，选择规则也用了该标签，结果必须披露 oracle-mask 条件，不能称为真实部署缺损检测。

### Exact → ModelMask v1

官方合同为白区生成、黑区保留上下文，缺失 RGB 区域用灰色。[VACE UserGuide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md) 明确 127；本方案据此固定填充值，并核对本地加载器，不能只凭历史 RGB127 合同认定兼容。

1. 原 Exact 保留为 L/0–255；另派生 `M=255 if Exact>127 else 0`，不反相、不膨胀、不缩放，保存 RGB 三个相同通道。
2. 从 Damaged 复制条件图，仅在 M 白区填 `RGB(127,127,127)`，黑区保持 Damaged，构造中不读取 GT RGB。
3. 给 `vace_video` 与 `vace_video_mask` 分别传同序 17 张 RGB PIL 列表，显式宽高 512、帧数 17，`vace_reference_image=None`，不传 GT 相关 `input_image/input_video`。
4. 本地 `preprocess_video` 按 HWC 读取，直接输入原 L 模式 Exact 实测会报 `EinopsError`；转换后 Mask 为 `[1,3,17,512,512]`、0/1，条件 RGB 范围为 -1 到 1。

所选真实帧的上述预处理已在 CPU 内存检查，形状、二值、通道和全黑/全白极性均通过；未保存派生 Mask/条件帧，未调用 VAE/DiT。检查仅证明预处理可接入当前代码。Exact 是像素中心首可见 Removed 标签，不是体积 Mask，抗锯齿混合像素可能在 Exact 外变化；将来若测试膨胀，单独建 v2，不改原件。

### 参数、输出与评价

初始参数：原始正式 VACE-1.3B、不加载 LoRA、BF16、seed 20261018、30 步、CFG 5、sigma_shift 5、vace_scale 1、tiled VAE、16 fps；中文内容 prompt 见 JSON。均为待实测参数。

每次新建 `outputs/vace/P01_465_17f_baseline_<unique_run_id>`，已有目录拒绝。未来执行器必须保存：

```text
run_config.json / environment.json / model_files_SHA256.json
frame_map.json                         源/输出编号、相机、角度、时间、哈希
inputs/damaged/ / inputs/exact/         所选原件的原样副本
inputs/model_mask/                     独立 Mask，转换版本与哈希
inputs/conditioning/                   Damaged+ModelMask 派生条件帧
evaluation/gt/                         仅供评估读取
prediction_raw/ / prediction_raw.mp4   先保存原始无损 PNG，再编码观看视频
comparison/                            可选评估/合成，不覆盖原始预测
stdout.log / stderr.log / exit_status.json / peak_memory.json
```

分别评估原始预测的 Mask 内 MAE/PSNR、Mask 外改变量、边界带误差与同视角视觉对比；全黑 Mask 的区域指标不适用，不能除零。原始预测保存后才能另存 `M*prediction+(1-M)*Damaged` 合成版，分别评价，不能用合成图冒充模型原始输出。黑区条件不意味着原始输出逐像素不变。17 帧像素结果不能证明三维恢复或完整环绕一致性。

## 7. 当前停止点

服务器数据交接验证完成；Windows 侧本机检查尚未执行。第一次实验仍需：

1. 用户确认可读正式 VACE 权重路径；如没有，另行授权下载，不能自动下载、换 Preview 或换框架。
2. 按冻结 JSON 实现项目层基线执行器，默认只读预检，显式运行开关、GPU 作业锁、分支/权重检查、独立输出及真实退出记录。
3. 在新实验目录落盘派生 Mask/条件帧、帧映射和模型文件哈希，GT 仅评估。CPU 内存检查不替代这些运行产物。
4. 等用户确认后才开始首次 GPU 推理，并记录真实显存和结果。

原 T2V 缓存仍为 1992 个 `.pth`。数据检查没有修改原数据、T2V 缓存/历史 LoRA/验证视频、模型文件、第三方核心、SSH 配置或 Windows workspace。2026-10-18 检查阶段没有提交、推送或拉取代码；后续 Git 同步按用户单独授权执行。

## 8. 本轮代码检查

`git diff --check`、训练/续训脚本 `bash -n`、训练/验证/新审计入口 `py_compile` 通过，`scripts/tests` 的 36 项 CPU 测试全部通过。新工具真实包只读复验通过；错误预期 SHA256 和已有输出目录都返回 1 拒绝，已有报告哈希未变；冻结 JSON 与实际 17 帧映射一致。PowerShell 命令仅人工审阅，尚未在 Windows 执行。

## 9. 版本控制交接（2026-10-19）

用户授权确认服务器 GitHub 账号为 woshuaime 后提交本轮准备内容。已通过实际 SSH 认证确认 `Hi woshuaime!`，本仓库 Git 提交身份也是 woshuaime；Linux 执行用户为 scj。沿用 origin 的 github-scj SSH 别名，不修改密钥或远程地址。

本轮发布范围为 P01 交接文档、17 帧 JSON 配置、只读包校验脚本及根目录的相关介绍章节。既有 T2V 审计和训练代码修改留在服务器工作区，不一并提交；原数据、ZIP、图片、模型、日志和 outputs 不入库。根目录文档的其余已提交内容按历史记录保留，当前目标与授权以 P01 章节为准。
