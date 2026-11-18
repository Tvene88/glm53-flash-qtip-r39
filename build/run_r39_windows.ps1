# run_r39_windows.ps1 — Windows 11 PowerShell Launcher for APEX-FOLD Round 3.9 (1x RTX 5090 + 96 GB DDR5)
param(
    [string]$Profile = "r39_flagship_75gb_sub023",
    [string]$Stages = "default",
    [string]$ModelsRoot = "C:\models",
    [string]$PythonExe = "python",
    [switch]$Resume,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:R37_GPU_WINDOW = "1"
$env:R36_GPU_HANDOVER = "1"
$env:R36_FUSE_MHC = "1"
$env:R36_FP8_GEMV = "v2"
$env:R37_POOL_REG_TOPK = "6"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Driver = Join-Path $ScriptDir "run_r39_master_pipeline.py"

$CmdArgs = @("-u", $Driver, "--profile", $Profile, "--stages", $Stages, "--models-root", $ModelsRoot)
if ($Resume) { $CmdArgs += "--resume" }
if ($DryRun) { $CmdArgs += "--dry-run" }

& $PythonExe @CmdArgs
exit $LASTEXITCODE
