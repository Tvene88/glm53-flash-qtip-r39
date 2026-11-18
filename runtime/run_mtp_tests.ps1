# GPU tests of the lossless speed work on the MTP (speculative decoding) path, tag [R3.9M].
# One CUDA process at a time; every run waits for a free GPU. Every MTP run is compared with the plain reference run
# (--compare-ids): generated ids of every round AND the bits of the logits behind every generated token (R39_MTP_FP=1).
#   -Step reference  : plain decode (no MTP), pool -PoolGib (16) -> out_r39s\mtp_reference.json. Run this first.
#   -Step exact      : MTP, rows exact, default policy -> mtp_exact.json (also records which op classes run per row)
#   -Step batchsweep : R35_MTP_BATCH, each op class alone, then the cumulative set of the classes that stayed
#                      identical. Classes that are no-ops in this configuration are skipped (from mtp_exact.json).
#                      -Classes 'norm,conv' runs only those.
#   -Step policy     : draft policy (never changes a token): tau 0.6 / 0.7 / 0.8, n-gram 3, pin45 all
#   -Step variants   : the new flag R39_MTP_PF_ROWS=1, alone, with REG_TOPK 8 / 4, and with pool 17 / 17.5 GiB
#   -Step kernel2    : kernel self-test (synthetic rows, about 1 min each) at several WARPS / PF settings; prints the
#                      N = 2 decode_core time of each, to choose R39_QTIP_N2_WARPS / R39_QTIP_N2_PF
#   -Step one        : one MTP run: -Tag name -Extra 'NAME=VALUE,...' [-PoolGib g]   (add -Plain for a plain run;
#                      -Extra applies to every step; a single R35_MTP_BATCH list needs a trailing ';': 'R35_MTP_BATCH=norm,conv;')
#   -Step table      : print the table of every report written so far (no GPU)
# Logs: out_r39s\mtp_<step>.log. Reports: out_r39s\mtp_<tag>.json. A run is about 3.5 minutes (load + 12 x 128 tokens).
param(
  [Parameter(Mandatory = $true)][ValidateSet('reference', 'exact', 'batchsweep', 'policy', 'variants', 'kernel2', 'one', 'table')][string]$Step,
  [string]$Fold = 'C:\models\GLM53-APEX-FOLD-R39-HYB-W16',
  [double]$PoolGib = 16,
  [string]$Extra = '',
  [string]$Tag = '',
  [string]$Classes = '',
  [switch]$Plain,
  [switch]$IdsOnly,
  [int]$WaitMinutes = 15
)
# [release] machine-specific paths replaced. R39_PYTHON = python.exe of the environment (default: python on PATH).
$PY = if ($env:R39_PYTHON) { $env:R39_PYTHON } else { 'python' }
$RT = $PSScriptRoot; $OUT = "$RT\out_r39s"
$BB = 'C:\models\GLM53-backbone-r34h'; $XBB = 'C:\models\GLM53-backbone'; $EVX = 'C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt'
$REF = "$OUT\mtp_reference.json"
New-Item -ItemType Directory -Force $OUT | Out-Null
Set-Location $RT
$log = "$OUT\mtp_$Step.log"
if ($Step -ne 'table') { "START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') step $Step pool $PoolGib extra '$Extra' tag '$Tag'" | Add-Content $log }

function Clear-R {
  Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
  $env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1'
}
function Set-Prod {
  # the production flags of run_kernel_tests.ps1 / run_port_tests.ps1 + expert kernel v2 at its measured best setting
  Clear-R
  $env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
  $env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
  $env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
  $env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
  $env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
  # bookkeeping of this test only (no effect on any token): per-step statistics and logit fingerprints
  $env:R39_MTP_STATS = '1'; $env:R39_MTP_FP = '1'
}
function Set-Vars([hashtable]$vars) {
  if ($vars) { foreach ($k in $vars.Keys) { Set-Item "Env:$k" ([string]$vars[$k]) } }
  # -Extra 'A=1,B=2'; with ';' between the pairs a value may hold commas: -Extra 'R35_MTP_BATCH=norm,conv;R39_MTP_PF_ROWS=1'
  $sep = if ($Extra -match ';') { ';' } else { ',' }
  foreach ($kv in ($Extra -split $sep | Where-Object { $_ })) { $k, $v = $kv -split '=', 2; Set-Item "Env:$($k.Trim())" $v }
}
function Wait-Gpu {
  for ($i = 0; $i -lt ($WaitMinutes * 6); $i++) {
    $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($used -le 4096) { return $true }
    Start-Sleep 10
  }
  return $false
}
function Run-One([string]$name, [string[]]$cmd, [string]$logPath = $log) {
  Start-Sleep 5
  if (-not (Wait-Gpu)) { "GPU busy: $name not run" | Add-Content $logPath; Write-Host "GPU busy after $WaitMinutes min: $name not run"; return $false }
  $envs = (Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]_' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
  "--- $name $(Get-Date -Format 'HH:mm:ss') | $envs" | Add-Content $logPath
  & $PY -u -W ignore @cmd *>> $logPath
  "exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')" | Add-Content $logPath
  return ($LASTEXITCODE -eq 0)
}
function Eval-Args([double]$pool, [string]$outPath) {
  return @("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Fold, '--backbone', $BB, '--exact-backbone', $XBB, '--evx', $EVX,
           '--qbench-only', '--speed-only', '--pool-gib', "$pool", '--out', $outPath)
}
# one plain run (no MTP layer loaded, the N = 1 graphs only)
function Run-Plain([string]$tag, [hashtable]$vars, [double]$pool = $PoolGib, [bool]$compare = $true) {
  $o = "$OUT\mtp_$tag.json"; $t0 = Get-Date
  Set-Prod; Set-Vars $vars
  $a = Eval-Args $pool $o
  if ($compare) { $a += @('--compare-ids', $REF) }
  $ok = Run-One "plain decode, pool $pool ($tag)" $a
  return ($ok -and (Test-Path $o) -and ((Get-Item $o).LastWriteTime -ge $t0))
}
# one MTP run on the graph path, compared with the reference
function Run-Mtp([string]$tag, [hashtable]$vars, [double]$pool = $PoolGib) {
  if (-not (Test-Path $REF)) { Write-Host "no reference report ($REF): run -Step reference first"; return $false }
  $o = "$OUT\mtp_$tag.json"; $t0 = Get-Date
  Set-Prod; $env:R35_MTP_GRAPHS = '1'; Set-Vars $vars
  $a = (Eval-Args $pool $o) + @('--mtp', '--compare-ids', $REF)
  $ok = Run-One "MTP, pool $pool ($tag)" $a
  return ($ok -and (Test-Path $o) -and ((Get-Item $o).LastWriteTime -ge $t0))
}
function Read-Report([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  try { return (Get-Content $path -Raw | ConvertFrom-Json) } catch { return $null }
}
# 'yes' / 'NO' / 'n/a' for the two lossless checks of a report (ids of every round; logit bits of every token)
function Get-Lossless($j) {
  $ids = '-'; $lg = '-'
  $ivr = $j.speed_gate.ids_vs_reference
  if ($ivr) {
    if ($ivr.r39m) {
      $ids = if ($ivr.r39m.ids_all) { 'yes' } else { 'NO' }
      $lg = if ($null -eq $ivr.r39m.logits_all) { 'n/a' } elseif ($ivr.r39m.logits_all) { 'yes' } else { 'NO' }
    } else {
      $all = $true
      foreach ($p in $ivr.prompts.PSObject.Properties) { if (-not $p.Value.identical) { $all = $false } }
      $ids = if ($all) { 'yes (round 1)' } else { 'NO' }
    }
  }
  return @($ids, $lg)
}
function Test-Lossless([string]$path) {
  $j = Read-Report $path
  if ($null -eq $j) { return $false }
  $r = Get-Lossless $j
  if ($IdsOnly) { return ($r[0] -like 'yes*') }
  return (($r[0] -like 'yes*') -and ($r[1] -eq 'yes'))
}
function Get-Row([string]$setting, [string]$path, $refMs) {
  $j = Read-Report $path
  if ($null -eq $j -or $null -eq $j.speed_gate) { return [pscustomobject]@{ setting = $setting; 'tok/s' = 'no report' } }
  $sg = $j.speed_gate; $pp = $sg.per_prompt_mean_tok_s; $s = $sg.r39m_summary; $r = Get-Lossless $j
  $row = [ordered]@{ setting = $setting; 'tok/s' = $sg.overall_mean_tok_s; code = $pp.code; capital = $pp.capital; math = $pp.math; systems = $pp.systems
                     'ids=' = $r[0]; 'logits=' = $r[1]; acc = ''; 'tok/step' = ''; 'pair ms' = ''; 'n1 ms' = ''; 'draft ms' = ''; 'pair/plain' = ''
                     'miss/tok' = ''; 'pf/tok' = ''; 'GiB/tok' = ''; 'free GiB' = $sg.vram_free_gib_after }
  if ($s) {
    $row['miss/tok'] = $s.misses_per_token; $row['pf/tok'] = $s.pf_issued_per_token; $row['GiB/tok'] = $s.gib_h2d_per_token
    if ($s.mtp) {
      $row['acc'] = $s.mtp.accept_rate; $row['tok/step'] = $s.mtp.tokens_per_step
      $k = $s.mtp.kinds; $n = 0; $ms = 0.0
      foreach ($kind in @('pair_acc', 'pair_rej')) { if ($k.$kind) { $n += $k.$kind.n; $ms += $k.$kind.n * $k.$kind.ms_mean } }
      if ($n -gt 0) { $row['pair ms'] = [math]::Round($ms / $n, 2) }
      if ($k.n1) { $row['n1 ms'] = $k.n1.ms_mean }
      if ($k.draft) { $row['draft ms'] = $k.draft.ms_mean }
      # cost of one verify step + its draft in units of one plain N = 1 step of the reference run (same pool size)
      if ($n -gt 0 -and $refMs -and $k.draft) { $row['pair/plain'] = [math]::Round((($ms / $n) + $k.draft.ms_mean) / $refMs, 3) }
    } else { $row['n1 ms'] = $s.ms_per_token }
  }
  return [pscustomobject]$row
}
function Show-Table {
  $refJ = Read-Report $REF
  $refMs = if ($refJ -and $refJ.speed_gate.r39m_summary) { $refJ.speed_gate.r39m_summary.ms_per_token } else { $null }
  $rows = @()
  $today = "$OUT\eval_qtip_w16_speed_pool18_k2_w16pf16.json"
  if (Test-Path $today) { $rows += Get-Row 'plain, pool 18 (kernel test of 2026-11-17)' $today $refMs }
  $rows += Get-Row 'reference = plain, this pool' $REF $refMs
  Get-ChildItem $OUT -Filter 'mtp_*.json' | Where-Object { $_.Name -ne 'mtp_reference.json' } | Sort-Object LastWriteTime | ForEach-Object {
    $j = Read-Report $_.FullName
    $e = if ($j) { $j.speed_gate.env } else { $null }
    $desc = $_.BaseName -replace '^mtp_', ''
    if ($e) {
      $bits = @()
      foreach ($n in @('R35_MTP_ROWS', 'R35_MTP_BATCH', 'R35_MTP_TAU', 'R35_MTP_NGRAM', 'R35_MTP_PIN45', 'R39_MTP_PF_ROWS', 'R37_POOL_REG_TOPK', 'R39_QTIP_N2_WARPS', 'R39_QTIP_N2_PF')) {
        if ($e.$n -and -not ($n -eq 'R37_POOL_REG_TOPK' -and $e.$n -eq '6')) { $bits += "$($n -replace '^R3[0-9]_(MTP_|POOL_|QTIP_)?', '')=$($e.$n)" }
      }
      $slots = $j.runtime.pool_slots
      $desc = "$desc [" + (($bits + "slots=$slots") -join ' ') + ']'
    }
    $rows += Get-Row $desc $_.FullName $refMs
  }
  $rows | Format-Table -Property * -AutoSize | Out-String -Width 360 | Write-Host
  Write-Host "ids= : generated ids of every round equal to the reference's.  logits= : the bits of the logits behind every generated token equal to the reference's"
  Write-Host "       (the lossless check proper; 'n/a' = no fingerprints in one of the two reports).  pair/plain = (verify step + draft) / plain ms per token."
  Write-Host "       A gain needs pair/plain < tok/step.  miss/tok, pf/tok, GiB/tok: demand misses, prefetches and bytes copied per generated token (decode only)."
}

switch ($Step) {
  'reference' {
    $t = if ($Tag) { $Tag } else { 'reference' }
    Run-Plain $t @{} $PoolGib $false | Out-Null
    Show-Table
  }
  'exact' {
    Run-Mtp 'exact' @{} | Out-Null
    Show-Table
  }
  'batchsweep' {
    # order: classes whose rows are provably computed independently first (norm, conv), then the cuBLAS classes
    # (core, attn, route, lin: only a run can tell), then head / mhc (no-ops when their kernels are row-local),
    # then final and rms (expected NOT bit-identical: they are the negative control of the logit check)
    $all = @('norm', 'conv', 'core', 'attn', 'route', 'lin', 'head', 'mhc', 'final', 'rms')
    $cls = if ($Classes) { @($Classes -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) } else { $all }
    $ex = Read-Report "$OUT\mtp_exact.json"
    if ($null -eq $ex) { Write-Host 'no mtp_exact.json: run -Step exact first'; break }
    $ops = $ex.speed_gate.mtp_info.pair_info.ops_per_step
    $good = @()
    foreach ($c in $cls) {
      $counted = @('rms', 'norm', 'lin', 'head', 'route', 'attn', 'final') -contains $c
      $noop = ($counted -and $ops -and -not (@($ops.PSObject.Properties.Name) -contains "${c}_per_row")) -or ($c -eq 'mhc' -and $ex.speed_gate.mhc_row_local)
      if ($noop) { "class ${c}: nothing runs per row in this configuration (no-op), not run" | Tee-Object -FilePath $log -Append | Write-Host; continue }
      if (-not (Run-Mtp "batch_$c" @{ R35_MTP_BATCH = $c })) { continue }
      if (Test-Lossless "$OUT\mtp_batch_$c.json") { $good += $c }
    }
    "identical classes: $($good -join ',')" | Tee-Object -FilePath $log -Append | Write-Host
    if ($good.Count -ge 2) { Run-Mtp 'batch_cum' @{ R35_MTP_BATCH = ($good -join ',') } | Out-Null }
    Show-Table
  }
  'policy' {
    foreach ($tau in @('0.6', '0.7', '0.8')) { Run-Mtp "tau$($tau -replace '\.', '')" @{ R35_MTP_TAU = $tau } | Out-Null }
    Run-Mtp 'ngram3' @{ R35_MTP_NGRAM = '3' } | Out-Null
    Run-Mtp 'pin45' @{ R35_MTP_PIN45 = 'all' } | Out-Null
    Show-Table
  }
  'variants' {
    # per-row prefetch gate (each row's top 6, as two N = 1 steps); then the gate one step looser and tighter (bytes
    # copied for nothing against demand misses); then the pool sizes the MTP sets leave room for (16 GiB left 3.2 GiB
    # free on 2026-11-17, plain decode with 18 GiB left 1.72: watch the 'free GiB' column)
    Run-Mtp 'pfrows' @{ R39_MTP_PF_ROWS = '1' } | Out-Null
    Run-Mtp 'pfrows_reg8' @{ R39_MTP_PF_ROWS = '1'; R37_POOL_REG_TOPK = '8' } | Out-Null
    Run-Mtp 'pfrows_reg4' @{ R39_MTP_PF_ROWS = '1'; R37_POOL_REG_TOPK = '4' } | Out-Null
    Run-Mtp 'pfrows_p170' @{ R39_MTP_PF_ROWS = '1' } 17.0 | Out-Null
    Run-Mtp 'pfrows_p175' @{ R39_MTP_PF_ROWS = '1' } 17.5 | Out-Null
    Show-Table
  }
  'kernel2' {
    $res = @()
    foreach ($g in @(@(16, 16), @(16, 8), @(8, 16), @(8, 8), @(4, 16))) {
      $w = $g[0]; $p = $g[1]; $kl = "$OUT\mtp_kernel2_w${w}_pf${p}.log"
      Clear-R; $env:R39_QTIP_WARPS = "$w"; $env:R39_QTIP_PF = "$p"
      "START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') warps $w pf $p" | Add-Content $kl
      $ok = Run-One "kernel selftest, synthetic rows, warps $w pf $p" @("$RT\glm53_fold_qtip_runtime.py", '--selftest-kernel') $kl
      $n1 = ''; $n2 = ''
      if ($ok) {
        $tail = Get-Content $kl | Select-Object -Last 60
        $m1 = $tail | Select-String -Pattern 'rule A .*graph replay, N=1, decode_core\s+\[v2\]: median\s+([0-9.]+) us' | Select-Object -Last 1
        $m2 = $tail | Select-String -Pattern 'rule A .*graph replay, N=2, decode_core\s+\[v2\]: median\s+([0-9.]+) us' | Select-Object -Last 1
        if ($m1) { $n1 = $m1.Matches[0].Groups[1].Value }
        if ($m2) { $n2 = $m2.Matches[0].Groups[1].Value }
      }
      $res += [pscustomobject]@{ warps = $w; pf = $p; passed = $ok; 'N=1 us' = $n1; 'N=2 us' = $n2 }
    }
    $res | Format-Table -AutoSize | Out-String -Width 200 | Tee-Object -FilePath $log -Append | Write-Host
    Write-Host 'N=2 here is decode_core without the per-row transform of exact mode (the GEMV launches are the same). If a setting other than'
    Write-Host 'warps 16 / pf 16 is fastest for N=2: -Step one -Tag n2 -Extra ''R39_QTIP_N2_WARPS=<w>,R39_QTIP_N2_PF=<p>'''
  }
  'one' {
    if (-not $Tag) { Write-Host '-Step one needs -Tag <name>'; break }
    if ($Plain) { Run-Plain $Tag @{} | Out-Null } else { Run-Mtp $Tag @{} | Out-Null }
    Show-Table
  }
  'table' { Show-Table }
}
if ($Step -ne 'table' -and (Test-Path $log)) {
  Get-Content $log | Where-Object { $_ -notmatch '^\s+\[pool\]|^\[SPEED SUMMARY\]' } | Select-Object -Last 40 | ForEach-Object { $_.Substring(0, [math]::Min(420, $_.Length)) }
}
