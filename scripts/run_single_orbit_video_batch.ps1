$ErrorActionPreference = "Stop"

$pythonPath = "C:\Users\DELL\anaconda3\python.exe"
$pythonwPath = "C:\Users\DELL\anaconda3\pythonw.exe"
$scriptPath = "D:\code\Python_3D_Scanner\scripts\build_single_orbit_mp4_videos.py"
$monitorPath = "D:\code\Python_3D_Scanner\scripts\monitor_single_orbit_video_progress.py"
$inputPath = "D:\code\single_orbit_image"
$outputPath = "D:\code\single_orbit_videos"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = "D:\code\single_orbit_videos_logs_$stamp"

foreach ($requiredPath in @($pythonPath, $pythonwPath, $scriptPath, $monitorPath, $inputPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "缺少必要文件或目录：$requiredPath"
    }
}

$modelDirs = @(Get-ChildItem -LiteralPath $inputPath -Directory | Where-Object Name -match '^\d+$' | Sort-Object { [int]$_.Name })
$total = $modelDirs.Count
if ($total -eq 0) { throw "输入目录中没有数字编号的模型文件夹：$inputPath" }

if (Test-Path -LiteralPath $outputPath) {
    $existing = @(Get-ChildItem -LiteralPath $outputPath -Force -Recurse -File)
    if ($existing.Count -gt 0) {
        throw "输出目录不是空目录，已停止以保护旧视频：$outputPath"
    }
} else {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}
New-Item -ItemType Directory -Path $logPath | Out-Null

$split = [int][Math]::Ceiling($total / 2.0)
$secondCount = $total - $split
$worker0Args = @(
    "-u", $scriptPath,
    "--input", $inputPath,
    "--output", $outputPath,
    "--start", "0", "--limit", "$split",
    "--skip-existing"
)
$worker1Args = @(
    "-u", $scriptPath,
    "--input", $inputPath,
    "--output", $outputPath,
    "--start", "$split", "--limit", "$secondCount",
    "--skip-existing"
)

$worker0Out = Join-Path $logPath "worker0.stdout.log"
$worker1Out = Join-Path $logPath "worker1.stdout.log"
$worker0Err = Join-Path $logPath "worker0.stderr.log"
$worker1Err = Join-Path $logPath "worker1.stderr.log"
$worker0 = Start-Process -FilePath $pythonPath `
    -ArgumentList $worker0Args `
    -WorkingDirectory "D:\code\Python_3D_Scanner" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $worker0Out `
    -RedirectStandardError $worker0Err `
    -PassThru

$worker1 = Start-Process -FilePath $pythonPath `
    -ArgumentList $worker1Args `
    -WorkingDirectory "D:\code\Python_3D_Scanner" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $worker1Out `
    -RedirectStandardError $worker1Err `
    -PassThru

$state = [ordered]@{
    started_at = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0
    total = $total
    workers = @(
        @{ name = "worker0"; pid = $worker0.Id; stdout = $worker0Out; stderr = $worker0Err; start = 1; count = $split },
        @{ name = "worker1"; pid = $worker1.Id; stdout = $worker1Out; stderr = $worker1Err; start = ($split + 1); count = $secondCount }
    )
}
$stateJson = $state | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText((Join-Path $logPath "batch_state.json"), $stateJson, [Text.UTF8Encoding]::new($false))

$monitorArgs = @(
    $monitorPath,
    "--input", $inputPath,
    "--output", $outputPath,
    "--log-dir", $logPath,
    "--total", "$total"
)
$monitor = Start-Process -FilePath $pythonwPath `
    -ArgumentList $monitorArgs `
    -WorkingDirectory "D:\code\Python_3D_Scanner" `
    -WindowStyle Normal `
    -PassThru

[PSCustomObject]@{
    TotalModels = $total
    Worker0_PID = $worker0.Id
    Worker0_Range = "1-$split"
    Worker1_PID = $worker1.Id
    Worker1_Range = "$(($split + 1))-$total"
    Monitor_PID = $monitor.Id
    Output = $outputPath
    Logs = $logPath
}
