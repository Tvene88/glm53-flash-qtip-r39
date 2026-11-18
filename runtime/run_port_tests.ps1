# GPU tests of the QTIP port into the Round 3.8 runtime (claude_r39_speed). One CUDA process at a time.
#   -Step selftest   : shipped QTIP self-test, port self-test, port self-test under the production flags
#   -Step exact      : forced-decode logits, old runtime vs new (flags off) vs new (production flags), bitwise compare
#   -Step speed      : QTIP archive, speed only, pool 18 GiB, production flags
#   -Step quality    : QTIP archive, qbench quality + speed, pool 16 GiB, production flags
#   -Step control    : shipped TCQ33P5, qbench quality + speed, pool 16 GiB, production flags
#   -Step speedab    : QTIP speed A/B (REG_TOPK unset; WARPS 16 / GX 32; shipped QTIP pool)
param([Parameter(Mandatory = $true)][ValidateSet('selftest', 'exact', 'speed', 'quality', 'control', 'speedab')][string]$Step)
# [release] machine-specific paths replaced. R39_PYTHON = python.exe of the environment (default: python on PATH).
$PY = if ($env:R39_PYTHON) { $env:R39_PYTHON } else { 'python' }
# R39_OLD_RUNTIME = the ORIGINAL package's runtime folder (not part of this repository); only -Step exact needs it.
$OLDRT = $env:R39_OLD_RUNTIME
$RT = $PSScriptRoot; $OUT = "$RT\out_r39s"
$Q = 'C:\models\GLM53-APEX-FOLD-R39-HYB-W16'; $P5 = 'C:\models\GLM53-APEX-FOLD-TCQ33P5'
$BB = 'C:\models\GLM53-backbone-r34h'; $XBB = 'C:\models\GLM53-backbone'; $EVX = 'C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt'
New-Item -ItemType Directory -Force $OUT | Out-Null
Set-Location $RT
$log = "$OUT\port_$Step.log"
"START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') step $Step" | Add-Content $log

function Clear-R {
  Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
  $env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1'
}
function Set-Prod {
  Clear-R
  $env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
  $env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
  $env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
  $env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
  $env:R37_TCQ_KERNEL = 'e_sh16'; $env:R37_TCQ_WARPS = '16'; $env:R37_TCQ_GX = '32'   # TCQ only; QTIP ignores them
}
function Run-One([string]$name, [string[]]$cmd) {
  Start-Sleep 6
  $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if ($used -gt 4096) { "GPU busy ($used MiB): $name not run" | Add-Content $log; return $false }
  $envs = (Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
  "--- $name $(Get-Date -Format 'HH:mm:ss') | $envs" | Add-Content $log
  & $PY -u -W ignore @cmd *>> $log
  "exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
  return ($LASTEXITCODE -eq 0)
}
$qargs = @('--backbone', $BB, '--exact-backbone', $XBB, '--evx', $EVX)

switch ($Step) {
  'selftest' {
    Clear-R; if (-not (Run-One 'shipped selftest' @("$RT\glm53_fold_qtip_runtime.py", '--selftest'))) { break }
    Clear-R; if (-not (Run-One 'port selftest, flags off' @("$RT\glm53_fold_qtip_runtime.py", '--selftest-port'))) { break }
    Set-Prod; Run-One 'port selftest, production flags' @("$RT\glm53_fold_qtip_runtime.py", '--selftest-port') | Out-Null
  }
  'exact' {
    if (-not $OLDRT) { 'R39_OLD_RUNTIME is not set: the old-runtime record needs the original package runtime' | Add-Content $log; break }
    Clear-R; if (-not (Run-One 'record old runtime' @("$RT\port_exact_r39s.py", 'record', '--runtime-dir', $OLDRT, '--fold', $Q, '--pool-gib', '16', '--register-gib', '42', '--out', "$OUT\pe_qtip_old.pt"))) { break }
    Clear-R; if (-not (Run-One 'record new runtime, flags off' @("$RT\port_exact_r39s.py", 'record', '--fold', $Q, '--pool-gib', '16', '--register-gib', '42', '--force', "$OUT\pe_qtip_old.pt", '--eager-steps', '16', '--out', "$OUT\pe_qtip_new_off.pt"))) { break }
    Set-Prod; if (-not (Run-One 'record new runtime, production flags' @("$RT\port_exact_r39s.py", 'record', '--fold', $Q, '--pool-gib', '16', '--register-gib', '68', '--force', "$OUT\pe_qtip_old.pt", '--out', "$OUT\pe_qtip_new_prod.pt"))) { break }
    Clear-R; $env:CUDA_VISIBLE_DEVICES = '-1'
    "--- compare old vs new (flags off)" | Add-Content $log; & $PY -u "$RT\port_exact_r39s.py" compare "$OUT\pe_qtip_old.pt" "$OUT\pe_qtip_new_off.pt" *>> $log
    "--- compare old vs new (production flags)" | Add-Content $log; & $PY -u "$RT\port_exact_r39s.py" compare "$OUT\pe_qtip_old.pt" "$OUT\pe_qtip_new_prod.pt" *>> $log
    Remove-Item Env:CUDA_VISIBLE_DEVICES
  }
  'speed' {
    Set-Prod; Run-One 'QTIP speed, pool 18, production flags' (@("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Q) + $qargs + @('--qbench-only', '--speed-only', '--pool-gib', '18', '--out', "$OUT\eval_qtip_w16_speed_pool18.json")) | Out-Null
  }
  'quality' {
    Set-Prod; Run-One 'QTIP qbench + speed, pool 16, production flags' (@("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Q) + $qargs + @('--qbench-only', '--speed-gate', '--pool-gib', '16', '--out', "$OUT\eval_qtip_w16_qbench_pool16.json")) | Out-Null
  }
  'control' {
    Set-Prod; Run-One 'TCQ33P5 qbench + speed, pool 16, production flags' (@("$RT\quality_eval_r39s.py", '--runner', 'tcq', '--fold', $P5) + $qargs + @('--qbench-only', '--speed-gate', '--pool-gib', '16', '--out', "$OUT\eval_p5_qbench_pool16.json")) | Out-Null
  }
  'speedab' {
    $sp = @("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Q) + $qargs + @('--qbench-only', '--speed-only', '--pool-gib', '18')
    Set-Prod; Remove-Item Env:R37_POOL_REG_TOPK; Run-One 'QTIP speed, REG_TOPK unset' ($sp + @('--out', "$OUT\eval_qtip_speed_noregtopk.json")) | Out-Null
    Set-Prod; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_GX = '32'; Run-One 'QTIP speed, WARPS 16 GX 32' ($sp + @('--out', "$OUT\eval_qtip_speed_w16gx32.json")) | Out-Null
    Set-Prod; $env:R39_POOL = 'qtip'; Run-One 'QTIP speed, shipped QTIP pool' ($sp + @('--out', "$OUT\eval_qtip_speed_qtippool.json")) | Out-Null
  }
}
Get-Content $log | Where-Object { $_ -notmatch '^\[eval-r39\w*\] row|^\s+\[pool\]' } | Select-Object -Last 45 | ForEach-Object { $_.Substring(0, [math]::Min(500, $_.Length)) }
