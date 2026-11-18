# GPU tests of the QTIP expert kernel v2 (R39_QTIP_KERNEL=v2). One CUDA process at a time; waits for a free GPU.
#   -Step selftest : --selftest-kernel on synthetic slabs, then with real rows of one layer (bitwise v1 == v2 + timing)
#   -Step sweep    : --selftest-kernel --kbench-sweep on real rows (grid of v2 settings)
#   -Step exact    : forced-decode logits with v2 under the production flags, compared bitwise with the v1 record
#   -Step speed    : speed gate with v2, production flags, pool 18 GiB  (-Extra 'NAME=VALUE,...' adds env variables)
param([Parameter(Mandatory = $true)][ValidateSet('selftest', 'sweep', 'exact', 'speed')][string]$Step, [string]$Extra = '', [string]$Tag = 'k2')
# [release] machine-specific paths replaced. R39_PYTHON = python.exe of the environment (default: python on PATH).
$PY = if ($env:R39_PYTHON) { $env:R39_PYTHON } else { 'python' }
$RT = $PSScriptRoot; $OUT = "$RT\out_r39s"
$Q = 'C:\models\GLM53-APEX-FOLD-R39-HYB-W16'
New-Item -ItemType Directory -Force $OUT | Out-Null
Set-Location $RT
$log = "$OUT\kernel_$Step.log"
"START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') step $Step extra '$Extra' tag $Tag" | Add-Content $log

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
  $env:R39_QTIP_KERNEL = 'v2'
  foreach ($kv in ($Extra -split ',' | Where-Object { $_ })) { $k, $v = $kv -split '=', 2; Set-Item "Env:$k" $v }
}
function Wait-Gpu {
  for ($i = 0; $i -lt 90; $i++) {
    $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($used -le 4096) { return $true }
    Start-Sleep 10
  }
  return $false
}
function Run-One([string]$name, [string[]]$cmd) {
  Start-Sleep 5
  if (-not (Wait-Gpu)) { "GPU busy: $name not run" | Add-Content $log; return $false }
  $envs = (Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
  "--- $name $(Get-Date -Format 'HH:mm:ss') | $envs" | Add-Content $log
  & $PY -u -W ignore @cmd *>> $log
  "exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
  return ($LASTEXITCODE -eq 0)
}

switch ($Step) {
  'selftest' {
    Clear-R; if (-not (Run-One 'kernel selftest, synthetic' @("$RT\glm53_fold_qtip_runtime.py", '--selftest-kernel'))) { break }
    Clear-R; Run-One 'kernel selftest, real rows' @("$RT\glm53_fold_qtip_runtime.py", '--selftest-kernel', '--fold', $Q) | Out-Null
  }
  'sweep' {
    Clear-R; Run-One 'kernel sweep, real rows' @("$RT\glm53_fold_qtip_runtime.py", '--selftest-kernel', '--fold', $Q, '--kbench-sweep') | Out-Null
  }
  'exact' {
    Set-Prod; if (-not (Run-One 'record new runtime, production flags, kernel v2' @("$RT\port_exact_r39s.py", 'record', '--fold', $Q, '--pool-gib', '16', '--register-gib', '68', '--force', "$OUT\pe_qtip_old.pt", '--out', "$OUT\pe_qtip_new_prod_$Tag.pt"))) { break }
    Clear-R; $env:CUDA_VISIBLE_DEVICES = '-1'
    "--- compare kernel v1 record vs kernel v2 record (both production flags)" | Add-Content $log
    & $PY -u "$RT\port_exact_r39s.py" compare "$OUT\pe_qtip_new_prod.pt" "$OUT\pe_qtip_new_prod_$Tag.pt" *>> $log
    Remove-Item Env:CUDA_VISIBLE_DEVICES
  }
  'speed' {
    Set-Prod; Run-One "QTIP speed, pool 18, production flags, kernel v2 ($Tag)" @("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Q, '--backbone', 'C:\models\GLM53-backbone-r34h', '--exact-backbone', 'C:\models\GLM53-backbone', '--evx', 'C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt', '--qbench-only', '--speed-only', '--pool-gib', '18', '--compare-ids', "$OUT\eval_qtip_speed_qtippool.json", '--out', "$OUT\eval_qtip_w16_speed_pool18_$Tag.json") | Out-Null
  }
}
Get-Content $log | Where-Object { $_ -notmatch '^\s+\[pool\]' } | Select-Object -Last 60 | ForEach-Object { $_.Substring(0, [math]::Min(420, $_.Length)) }
