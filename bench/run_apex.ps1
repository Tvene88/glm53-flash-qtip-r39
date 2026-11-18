# Serves one APEX-FOLD archive ("new" = QTIP, "shipped" = TCQ33P5) with the settings of README.md section 3.1,
# runs the smoke test and then HumanEval + the GSM8K sample (README section 5), and stops the server.
# One CUDA process at a time: refuses to start while the GPU is in use.
param([Parameter(Mandatory = $true)][ValidateSet('new', 'shipped')][string]$Kind,
      [string]$Python = $env:R39_PYTHON, [string]$Root = (Split-Path -Parent $PSScriptRoot),
      [int]$LimitHe = 0, [int]$LimitGsm = 0, [switch]$SmokeOnly, [string]$PoolGib = '14', [switch]$Traffic, [string]$Tag = '',
      [string]$Models = $(if ($env:R39_MODELS) { $env:R39_MODELS } else { 'C:\models' }))
$Runtime = "$Root\runtime"; $Bench = "$Root\bench"
$Backbone = "$Models\GLM53-backbone-r34h"
$Fold = if ($Kind -eq 'new') { "$Models\GLM53-APEX-FOLD-R39-HYB-W16" } else { "$Models\GLM53-APEX-FOLD-TCQ33P5" }
$Name = if ($Kind -eq 'new') { 'apex-fold-r39-qtip' } else { 'apex-fold-tcq33p5' }
$VocabKeep = "$Root\configs\r34_vocab_keep.json"
$Port = 8188; $Url = "http://127.0.0.1:$Port/v1"; $Runs = "$Bench\runs"
New-Item -ItemType Directory -Force $Runs | Out-Null
if (Test-Path "$Runs\SKIP_$Kind") { "skipped: $Runs\SKIP_$Kind exists"; exit 0 }
$slog = "$Runs\server_$Kind.log"
if (-not (Test-Path $VocabKeep)) { "missing $VocabKeep"; exit 2 }
$used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if ($used -gt 4096) { "GPU busy ($used MiB): not started"; exit 2 }

Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]T?_' -and $_.Name -ne 'R39_PYTHON' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
$env:PYTHONUTF8 = '1'
$env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
$env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
$env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
$env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
$env:R35_DSA_QBLOCK_MIB = '64'; $env:R37_DENSE_MIN_PAIRS = '1000000000'; $env:R37_EMPTY_CACHE = '1'
$env:R37_VRAM_CAP_MIB = '31400'; $env:R37_PREFILL_CHUNK = '2048'
$env:R36_LONGCTX = '1'; $env:R36_LONGCTX_MAX = '65536'; $env:R37_LONGCTX_FAST = '1'; $env:R37_DUAL_GRAPHS = '1'
if ($Kind -eq 'new') { $env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16' }
else { $env:R37_TCQ_KERNEL = 'e_sh16'; $env:R37_TCQ_WARPS = '16'; $env:R37_TCQ_GX = '32' }
if ($Traffic) { $env:R39T_FAST_DEMAND = '2'; $env:R39T_PREEVICT = '4'; $env:R39T_LAZY_STAGING = '1' }
$Out = if ($Tag) { "${Kind}_$Tag" } else { $Kind }

$sargs = @('-u', '-W', 'ignore', "$Runtime\glm53_fold_generate.py",
  '--backbone-dir', $Backbone, '--fold-dir', $Fold, '--vocab-keep', $VocabKeep,
  '--backbone-mode', 'fp8', '--cold-mode', 'mmap', '--vram-budget-gib', '28', '--expert-pool-gib', $PoolGib,
  '--graphs', '--no-sync-timing', '--warm-cache', '--no-mtp-layer',
  '--serve', '--port', "$Port", '--serve-model-name', $Name, '--serve-early-token',
  '--serve-max-context', '65536', '--serve-max-tokens', '4096',
  '--serve-state-gib', '5', '--serve-cache-entries', '1', '--serve-cache-points', '2')
"START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') serve $Kind" | Set-Content $slog
$srv = Start-Process -FilePath $Python -ArgumentList $sargs -WorkingDirectory $Runtime -RedirectStandardOutput "$slog.out" -RedirectStandardError "$slog.err" -PassThru -WindowStyle Hidden
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
  Start-Sleep 10
  if ($srv.HasExited) { break }
  try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 | Out-Null; $ready = $true; break } catch {}
}
if (-not $ready) {
  "server not ready (exited=$($srv.HasExited))"
  Get-Content "$slog.out", "$slog.err" -ErrorAction SilentlyContinue | Select-Object -Last 40 | ForEach-Object { $_.Substring(0, [math]::Min(300, $_.Length)) }
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
  exit 3
}
"server ready $(Get-Date -Format 'HH:mm:ss'); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
try {
  $Common = @('--base-url', $Url, '--max-tokens', '1024', '--temperature', '0', '--thinking', 'off', '--reasoning-effort', 'low',
              '--extra-body', "@$Bench\extra_body.json", '--concurrency', '1')
  Remove-Item "$Runs\smoke_$Kind" -Recurse -Force -ErrorAction SilentlyContinue
  & $Python "$Bench\bench_openai.py" --task humaneval --ids-file "$Bench\data\humaneval_order.json" --limit 3 --model $Kind --out "$Runs\smoke_$Kind" @Common
  "smoke exit=$LASTEXITCODE"
  if ($LASTEXITCODE -eq 0 -and -not $SmokeOnly) {
    $he = @('--task', 'humaneval', '--ids-file', "$Bench\data\humaneval_order.json", '--model', $Kind, '--out', "$Runs\$Out\humaneval")
    if ($LimitHe -gt 0) { $he += @('--limit', "$LimitHe") }
    if (Test-Path "$Runs\$Out\humaneval\responses.jsonl") { $he += '--resume' }
    & $Python "$Bench\bench_openai.py" @he @Common; "humaneval exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
    $gs = @('--task', 'gsm8k', '--model', $Kind, '--out', "$Runs\$Out\gsm8k")
    if ($LimitGsm -gt 0) { $gs += @('--limit', "$LimitGsm") }
    if (Test-Path "$Runs\$Out\gsm8k\responses.jsonl") { $gs += '--resume' }
    & $Python "$Bench\bench_openai.py" @gs @Common; "gsm8k exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
  }
}
finally {
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
  Start-Sleep 8
  "server stopped; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
foreach ($t in 'humaneval', 'gsm8k') {
  $sj = "$Runs\$Out\$t\summary.json"
  if (Test-Path $sj) { "--- $Kind $t"; Get-Content $sj -Raw | ForEach-Object { $_.Substring(0, [math]::Min(1500, $_.Length)) } }
}
