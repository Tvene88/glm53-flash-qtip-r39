# GPU tests of the lossless expert-traffic work (host -> GPU copies and host overhead of plain decode), tag [R3.9T].
# One CUDA process at a time; every run waits for a free GPU (<= 4096 MiB used) and for 75 GiB of free RAM. Every run
# is compared with the reference run (--compare-ids): generated ids of every round AND the bits of the logits behind
# every generated token (R39_MTP_FP=1).
#   -Step reference : production flags + kernel v2, pool -PoolGib (18) -> out_r39s\traffic_reference.json. Run first.
#   -Step prof      : the same with R39T_PROF=1 (step-time split + no-miss floors) -> traffic_prof.json (-Tag other name,
#                     -Extra flags to profile a variant); -Step split prints its tables again (no GPU)
#   -Step trace     : R39T_PROF=1 + R39T_TRACE (the pool's call log, about 11 MB) -> trace_r39t_pool<g>.json, then the
#                     offline simulation table of traffic_sim_r39t.py (CPU, about 3 minutes)
#   -Step one       : one run: -Tag name -Extra 'NAME=VALUE,...' [-PoolGib g]
#   -Step changes   : every [R3.9T] change alone and the existing gates that were re-measured (list in the switch below)
#   -Step best      : the recommended combined setting only -> traffic_best.json
#   -Step ab        : the recommended setting (or -Extra / -PoolGib) against the reference flags, alternating, -Repeats (2)
#                     times each -> traffic_ab_<tag>_A<i>.json (reference flags) / _B<i>.json; prints both means. Use this
#                     for a speed claim: single runs differ by about 1 % on a quiet machine and by 10 % on a busy one.
#   -Step pair      : paired A/B inside ONE process (R39T_AB): -Pair '<fast_demand>,<preevict>' (default 1,0) against
#                     both off, 4 rounds x 4 prompts per arm, alternating -> traffic_pair_<fd>_<pe>.json; prints the
#                     ratio. The only comparison here that resolves 1 % while another job loads the machine.
#   -Step pairs     : print the paired results of every traffic_pair_*.json (no GPU)
#   -Step torture   : test_pool_r39t.py (the pool's GPU torture load with every new pool flag; about 4 minutes, no model)
#   -Step table     : print the table of every report written so far (no GPU)
#   -Step split     : print the step-time split of traffic_prof.json (or -Tag <name> of another R39T_PROF report)
# Logs: out_r39s\traffic_<step>.log. Reports: out_r39s\traffic_<tag>.json. A run is about 3.5 minutes.
param(
  [Parameter(Mandatory = $true)][ValidateSet('reference', 'prof', 'trace', 'one', 'changes', 'best', 'ab', 'pair', 'pairs', 'torture', 'table', 'split')][string]$Step,
  [string]$Fold = 'C:\models\GLM53-APEX-FOLD-R39-HYB-W16',
  [double]$PoolGib = 18,
  [string]$Extra = '',
  [string]$Tag = '',
  [int]$Repeats = 2,
  [string]$Pair = '1,0',
  [int]$WaitMinutes = 20
)
# [release] machine-specific paths replaced. R39_PYTHON = python.exe of the environment (default: python on PATH).
$PY = if ($env:R39_PYTHON) { $env:R39_PYTHON } else { 'python' }
$RT = $PSScriptRoot; $OUT = "$RT\out_r39s"
$BB = 'C:\models\GLM53-backbone-r34h'; $XBB = 'C:\models\GLM53-backbone'; $EVX = 'C:\models\GLM53-APEX-FOLD-TCQ33\_r33_ckpt\evx.pt'
$REF = "$OUT\traffic_reference.json"
New-Item -ItemType Directory -Force $OUT | Out-Null
Set-Location $RT
$log = "$OUT\traffic_$Step.log"
$noGpu = @('table', 'split', 'pairs') -contains $Step
if (-not $noGpu) { "START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') step $Step pool $PoolGib extra '$Extra' tag '$Tag'" | Add-Content $log }

# the recommended combined setting (2026-11-17): the registered demand copies issued without the stream context and
# with the hits' priority refresh after them, 4 slots freed ahead of every ensure(), the unused cold-path staging blocks
# not allocated and their 0.47 GiB of VRAM given to the pool as 80 more slots (18.47 GiB = 3,134 slots; VRAM free as with 18)
$BEST = @{ R39T_FAST_DEMAND = '2'; R39T_PREEVICT = '4'; R39T_LAZY_STAGING = '1' }
$BESTPOOL = 18.47

function Clear-R {
  Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]T?_' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }
  $env:PYTHONUTF8 = '1'; $env:R37_GPU_WINDOW = '1'
}
function Set-Prod {
  # the production flags of run_mtp_tests.ps1 (Set-Prod) + expert kernel v2 at its measured best setting
  Clear-R
  $env:R32_POOL = '6'; $env:R35_POOL_SIGNAL = '1'; $env:R35_POOL_DEMAND = '1'; $env:R35_POOL_DEMAND_STREAM = '1'
  $env:R35_POOL_GATHER = 'readfile'; $env:R35_POOL_EVICT = 'gdsf'; $env:R35_POOL_COLD_TOPK = '6'; $env:R35_POOL_COLD_MARGIN = '0.01'
  $env:R33_REGISTER_GIB = '68'; $env:R33_REGISTER_MODE = 'rows'
  $env:R36_FUSE_MHC = '1'; $env:R36_FP8_GEMV = 'v2'; $env:R37_POOL_REG_TOPK = '6'
  $env:R39_QTIP_KERNEL = 'v2'; $env:R39_QTIP_WARPS = '16'; $env:R39_QTIP_PF = '16'
  # bookkeeping only (no effect on any token): per-run pool statistics and logit fingerprints
  $env:R39_MTP_STATS = '1'; $env:R39_MTP_FP = '1'
}
function Set-Vars([hashtable]$vars) {
  if ($vars) { foreach ($k in $vars.Keys) { Set-Item "Env:$k" ([string]$vars[$k]) } }
  $sep = if ($Extra -match ';') { ';' } else { ',' }
  foreach ($kv in ($Extra -split $sep | Where-Object { $_ })) { $k, $v = $kv -split '=', 2; Set-Item "Env:$($k.Trim())" $v }
}
function Wait-Gpu {
  # GPU rule: not above 4096 MiB used by someone else; RAM rule: 75 GiB free (the runtime locks about 60 GiB)
  for ($i = 0; $i -lt ($WaitMinutes * 6); $i++) {
    $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    $freeGib = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($used -le 4096 -and $freeGib -ge 75) { return $true }
    Start-Sleep 10
  }
  return $false
}
function Run-One([string]$name, [string[]]$cmd) {
  # one run of this script at a time, also across several invocations of it (a second one waits here: two runs that
  # both see a free GPU between two loads would start two CUDA processes and lock 2 x 60 GiB of RAM)
  $mx = New-Object System.Threading.Mutex($false, 'Global\glm53_r39t_gpu_run')
  try { [void]$mx.WaitOne() } catch [System.Threading.AbandonedMutexException] { }
  try {
    Start-Sleep 5
    if (-not (Wait-Gpu)) { "GPU or RAM busy: $name not run" | Add-Content $log; Write-Host "GPU / RAM busy after $WaitMinutes min: $name not run"; return $false }
    $envs = (Get-ChildItem Env: | Where-Object { $_.Name -match '^R3[2-9]T?_' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' '
    "--- $name $(Get-Date -Format 'HH:mm:ss') | $envs" | Add-Content $log
    & $PY -u -W ignore @cmd *>> $log
    $code = $LASTEXITCODE
    "exit=$code $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
    return ($code -eq 0)
  } finally { $mx.ReleaseMutex(); $mx.Dispose() }
}
function Run-Plain([string]$tag, [hashtable]$vars, [double]$pool = $PoolGib, [bool]$compare = $true) {
  if ($compare -and -not (Test-Path $REF)) { Write-Host "no reference report ($REF): run -Step reference first"; return $false }
  $o = "$OUT\traffic_$tag.json"; $t0 = Get-Date
  Set-Prod; Set-Vars $vars
  $a = @("$RT\quality_eval_r39s.py", '--runner', 'qtip', '--fold', $Fold, '--backbone', $BB, '--exact-backbone', $XBB, '--evx', $EVX,
         '--qbench-only', '--speed-only', '--pool-gib', "$pool", '--out', $o)
  if ($compare) { $a += @('--compare-ids', $REF) }
  $ok = Run-One "plain decode, pool $pool ($tag)" $a
  return ($ok -and (Test-Path $o) -and ((Get-Item $o).LastWriteTime -ge $t0))
}
function Read-Report([string]$path) {
  if (-not (Test-Path $path)) { return $null }
  try { return (Get-Content $path -Raw | ConvertFrom-Json) } catch { return $null }
}
function Get-Lossless($j) {
  $ivr = $j.speed_gate.ids_vs_reference
  if (-not $ivr -or -not $ivr.r39m) { return @('-', '-') }
  $ids = if ($ivr.r39m.ids_all) { 'yes' } else { 'NO' }
  $lg = if ($null -eq $ivr.r39m.logits_all) { 'n/a' } elseif ($ivr.r39m.logits_all) { 'yes' } else { 'NO' }
  return @($ids, $lg)
}
function Get-Row([string]$setting, [string]$path) {
  $j = Read-Report $path
  if ($null -eq $j -or $null -eq $j.speed_gate) { return [pscustomobject]@{ setting = $setting; 'tok/s' = 'no report' } }
  $sg = $j.speed_gate; $pp = $sg.per_prompt_mean_tok_s; $s = $sg.r39m_summary; $r = Get-Lossless $j
  $row = [ordered]@{ setting = $setting; at = (Get-Item $path).LastWriteTime.ToString('HH:mm'); 'tok/s' = $sg.overall_mean_tok_s; code = $pp.code; capital = $pp.capital; math = $pp.math; systems = $pp.systems
                     'ms/tok' = ''; 'miss/tok' = ''; 'pf/tok' = ''; 'pf used' = ''; 'claimed' = ''; 'dropped' = ''; 'GiB/tok' = ''; slots = $j.runtime.pool_slots
                     'free GiB' = $sg.vram_free_gib_after; 'ids=' = $r[0]; 'logits=' = $r[1] }
  if ($s) {
    $row['ms/tok'] = $s.ms_per_token; $row['miss/tok'] = $s.misses_per_token; $row['pf/tok'] = $s.pf_issued_per_token
    $row['pf used'] = $s.pf_useful_per_token; $row['GiB/tok'] = $s.gib_h2d_per_token
    # queued prefetches ensure() copied itself (claimed) / that were dropped before any copy, per token (decode only)
    $cl = 0; $dr = 0; $nt = 0
    foreach ($r in $sg.r39m_runs) { $cl += $r.pool_decode.prefetch_claimed; $dr += $r.pool_decode.prefetch_dropped; $nt += $r.pool_decode.new_tokens }
    if ($nt -gt 0) { $row['claimed'] = [math]::Round($cl / $nt, 2); $row['dropped'] = [math]::Round($dr / $nt, 2) }
  }
  return [pscustomobject]$row
}
function Show-Table {
  $rows = @()
  $rows += Get-Row 'reference (production flags + kernel v2)' $REF
  Get-ChildItem $OUT -Filter 'traffic_*.json' | Where-Object { $_.Name -ne 'traffic_reference.json' -and $_.Name -notlike 'traffic_pair_*' } | Sort-Object LastWriteTime | ForEach-Object {
    $j = Read-Report $_.FullName
    $e = if ($j) { $j.speed_gate.env } else { $null }
    $desc = $_.BaseName -replace '^traffic_', ''
    if ($e) {
      $bits = @()
      foreach ($p in $e.PSObject.Properties) {
        if ($p.Name -match '^R39T_(?!TRACE)' -or ($p.Name -eq 'R37_POOL_REG_TOPK' -and $p.Value -ne '6') -or ($p.Name -eq 'R35_POOL_COLD_TOPK' -and $p.Value -ne '6') -or
            $p.Name -eq 'R35_POOL_DROP' -or $p.Name -eq 'R37_POOL_REG_MARGIN') {
          $bits += "$($p.Name -replace '^R3[0-9]T?_(POOL_)?', '')=$($p.Value)"
        }
      }
      # reports written before the R39T_* names were added to the report's env list: the flags of that tag
      $old = @{ fastdemand = 'FAST_DEMAND=1'; pfad_topk8 = 'PF_AFTER_DEMAND=1'; lazy_pool1847 = 'LAZY_STAGING=1'; prof = 'PROF=1'; proftrace = 'PROF=1' }
      $tagName = $_.BaseName -replace '^traffic_', ''
      if (-not ($bits | Where-Object { $_ -match '^(FAST_|PF_|LAZY_|PROF)' }) -and $old.ContainsKey($tagName)) { $bits = @($old[$tagName]) + $bits }
      if ($bits.Count) { $desc = "$desc [" + ($bits -join ' ') + ']' }
    }
    $rows += Get-Row $desc $_.FullName
  }
  $rows | Format-Table -Property * -AutoSize | Out-String -Width 400 | Write-Host
  Write-Host "ids= : generated ids of every round equal to the reference's.  logits= : the bits of the logits behind every generated token equal to the"
  Write-Host "       reference's ('-' = the reference itself or a run without --compare-ids).  miss/tok, pf/tok, pf used, GiB/tok: demand misses, prefetches"
  Write-Host "       issued, prefetches later used, and bytes copied host -> GPU per generated token (decode only).  free GiB: VRAM free after the run."
  Write-Host "       RAM locked by every run: the registered tier, 59.95 GiB (R33_REGISTER_GIB=68 covers all 10,174 rows)."
  Write-Host "       claimed / dropped: queued prefetches that ensure() copied itself / that were withdrawn before any copy.  at: when the run ended."
  Write-Host "       tok/s of single runs: compare only runs made under the same machine load (on 2026-11-17 a second session loaded the machine from"
  Write-Host "       14:18 on: the same reference flags gave 25.41 before and 22.7 to 24.6 after). For a speed claim use -Step pair or -Step ab."
  Write-Host "       The paired runs (traffic_pair_*) are listed by -Step pairs."
}
function Show-Pairs {
  $rows = @()
  Get-ChildItem $OUT -Filter 'traffic_pair_*.json' | Sort-Object LastWriteTime | ForEach-Object {
    $j = Read-Report $_.FullName
    if ($j -and $j.speed_gate.r39t_ab) {
      $ab = $j.speed_gate.r39t_ab; $r = Get-Lossless $j
      $rows += [pscustomobject][ordered]@{ report = $_.BaseName; 'B: FAST_DEMAND' = $ab.B.fast_demand; 'B: PREEVICT' = $ab.B.preevict
        'A tok/s' = ('{0:N2}' -f $ab.A_mean_tok_s); 'B tok/s' = ('{0:N2}' -f $ab.B_mean_tok_s)
        'B vs A, mean of pairs' = ('{0:+0.00;-0.00} %' -f (100 * ($ab.B_over_A_mean_of_pairs - 1)))
        'median of pairs' = ('{0:+0.00;-0.00} %' -f (100 * ($ab.B_over_A_median_of_pairs - 1))); 'pairs B faster' = "$($ab.pairs_B_faster) of $($ab.pairs)"
        'AB ids = round 1' = $ab.ids_equal_to_round_1; 'ids=' = $r[0]; 'logits=' = $r[1] }
    }
  }
  if ($rows.Count) { $rows | Format-Table -AutoSize | Out-String -Width 300 | Write-Host } else { Write-Host 'no traffic_pair_*.json yet' }
  Write-Host "A = R39T_FAST_DEMAND and R39T_PREEVICT off, B = the values shown; one process, every prompt generated alternately with A and B"
  Write-Host "(order swapped every round), ratio per pair of back-to-back runs. ids= / logits= : round 1 of that process against the reference."
}
function Show-Split([string]$path) {
  $j = Read-Report $path
  if ($null -eq $j -or $null -eq $j.speed_gate.r39t_prof) { Write-Host "no R39T_PROF data in $path"; return }
  $sg = $j.speed_gate
  $rows = @()
  $tags = @($sg.r39t_prof.PSObject.Properties.Name)
  $src = @{}
  foreach ($t in $tags) { $src[$t] = $sg.r39t_prof.$t }
  if ($sg.r39t_floor.loop_no_miss) { $tags += 'no-miss loop'; $src['no-miss loop'] = $sg.r39t_floor.loop_no_miss }
  foreach ($t in $tags) {
    $d = $src[$t]; $g = $d.gpu_ms; $h = $d.host_ms; $p = $d.pool_per_token
    $rows += [pscustomobject][ordered]@{ prompt = $t; steps = $d.steps; 'wall ms' = $d.wall_ms
      'GPU a' = $g.a; 'GPU p' = $g.p; 'GPU b' = $g.b; 'GPU dense' = $g.dense; 'GPU final' = $g.final; 'graphs' = $d.gpu_graphs_ms
      'stall(copy)' = $g.gap_stall; 'layers stalled' = $d.layers_stalled; 'idle(hit)' = $g.gap_hit
      'host replay' = $h.replay; 'host ensure' = $h.ensure; 'of it stalled' = $d.host_ensure_stall_ms; 'host prefetch' = $h.prefetch; 'host embed' = $h.embed
      'wait a' = $h.sync_a; 'wait p' = $h.sync_p; 'wait final' = $h.item
      'miss' = $p.misses; 'claimed' = $p.prefetch_claimed; 'pf' = $p.prefetch_issued; 'pf used' = $p.prefetch_useful; 'GiB' = $p.gib_h2d
      'pre-issue us/fetch' = $(if ($p.r39t_fetch_calls -and $null -ne $p.t_r39t_pre_issue_ms) { [math]::Round(1000 * $p.t_r39t_pre_issue_ms / $p.r39t_fetch_calls, 1) } else { '' })
      'pre-issue ms/tok' = $p.t_r39t_pre_issue_ms
      'wk start ms/item' = $(if ($p.r39t_wk_items) { [math]::Round($p.t_r39t_wk_start_ms / $p.r39t_wk_items, 3) } else { '' })
      'wk done ms/item' = $(if ($p.r39t_wk_items) { [math]::Round($p.t_r39t_wk_done_ms / $p.r39t_wk_items, 3) } else { '' }) }
  }
  $rows | Format-List | Out-String -Width 200 | Write-Host
  Write-Host "per generated token, ms.  GPU a/p/b/dense/final: compute-stream time between the graphs' end events (a = attention + router and"
  Write-Host "the idle time until the host queued it; p = prediction + shared expert; b = routed experts + merge).  stall(copy): compute stream"
  Write-Host "between p{l} and b{l} in layers whose ensure() copied or waited for a copy; idle(hit): the same gap in all-hit layers (GPU idle"
  Write-Host "behind the host).  host replay/ensure/prefetch/embed: host busy; wait a / p / final: host waiting for the GPU."
  Write-Host "pre-issue: host time from the start of an ensure() that has to copy to its first copy handed to CUDA (the compute stream is"
  Write-Host "already stalled then).  wk start / done: prefetch() hand-over to the worker starting the item / having issued its copies."
  Write-Host ('floors: ' + ($sg.r39t_floor | Select-Object n, graph_launches_per_token, gpu_queue_ms, submit_ms, gpu_per_token_sync_ms | ConvertTo-Json -Compress))
  Write-Host ('memory: ' + ($sg.r39t_mem | ConvertTo-Json -Compress))
}

switch ($Step) {
  'reference' {
    $t = if ($Tag) { $Tag } else { 'reference' }
    Run-Plain $t @{} $PoolGib $false | Out-Null
    Show-Table
  }
  'prof' {
    $t = if ($Tag) { $Tag } else { 'prof' }
    Run-Plain $t @{ R39T_PROF = '1' } | Out-Null
    Show-Split "$OUT\traffic_$t.json"
  }
  'one' {
    if (-not $Tag) { Write-Host '-Step one needs -Tag <name>'; break }
    Run-Plain $Tag @{} | Out-Null
    Show-Table
  }
  'changes' {
    # every [R3.9T] change alone (pool 18 GiB unless said), then the existing gates that were re-measured
    Run-Plain 'fastdemand' @{ R39T_FAST_DEMAND = '1' } | Out-Null
    Run-Plain 'fastdemand2' @{ R39T_FAST_DEMAND = '2' } | Out-Null
    Run-Plain 'preevict4' @{ R39T_PREEVICT = '4' } | Out-Null
    Run-Plain 'pfad' @{ R39T_PF_AFTER_DEMAND = '1' } | Out-Null
    Run-Plain 'paced1' @{ R39T_PF_PACED = '1'; R35_POOL_DROP = 'all' } | Out-Null
    Run-Plain 'paced2' @{ R39T_PF_PACED = '2'; R35_POOL_DROP = 'all' } | Out-Null
    Run-Plain 'lazy_pool1847' @{ R39T_LAZY_STAGING = '1' } 18.47 | Out-Null
    Run-Plain 'regtopk7' @{ R37_POOL_REG_TOPK = '7' } | Out-Null
    Run-Plain 'regtopk8' @{ R37_POOL_REG_TOPK = '8' } | Out-Null
    Run-Plain 'pfad_topk8' @{ R39T_PF_AFTER_DEMAND = '1'; R37_POOL_REG_TOPK = '8' } | Out-Null
    Run-Plain 'margin004' @{ R37_POOL_REG_TOPK = '0'; R37_POOL_REG_MARGIN = '0.04' } | Out-Null
    Run-Plain 'overlap' @{ R36_OVERLAP = '1' } | Out-Null          # the existing hit-first scheduling (Round 3.6 K3)
    Show-Table
  }
  'best' {
    Run-Plain 'best' $BEST $BESTPOOL | Out-Null
    Show-Table
  }
  'ab' {
    $t = if ($Tag) { $Tag } else { 'best' }
    $bVars = if ($Extra) { @{} } else { $BEST }
    $bPool = if ($Extra -or $PSBoundParameters.ContainsKey('PoolGib')) { $PoolGib } else { $BESTPOOL }
    $extraKeep = $Extra
    for ($i = 1; $i -le $Repeats; $i++) {
      $Extra = ''                                   # A = the reference flags, pool 18
      Run-Plain "ab_${t}_A$i" @{} 18 | Out-Null
      $Extra = $extraKeep                           # B = the variant
      Run-Plain "ab_${t}_B$i" $bVars $bPool | Out-Null
    }
    foreach ($side in @('A', 'B')) {
      $v = @(); Get-ChildItem $OUT -Filter "traffic_ab_${t}_$side*.json" | ForEach-Object { $j = Read-Report $_.FullName; if ($j) { $v += [double]$j.speed_gate.overall_mean_tok_s } }
      if ($v.Count) { Write-Host ("{0} ({1}): mean {2:N2} tok/s over {3} runs [{4}]" -f $side, $(if ($side -eq 'A') { 'reference flags' } else { 'variant' }), ($v | Measure-Object -Average).Average, $v.Count, (($v | ForEach-Object { '{0:N2}' -f $_ }) -join ' ')) }
    }
  }
  'pair' {
    $t = 'pair_' + ($Pair -replace ',', '_')
    Run-Plain $t @{ R39T_AB = $Pair } | Out-Null
    Show-Pairs
  }
  'pairs' { Show-Pairs }
  'trace' {
    $tr = "$OUT\trace_r39t_pool$PoolGib.json"
    Run-Plain 'proftrace' @{ R39T_PROF = '1'; R39T_TRACE = $tr } | Out-Null
    if (Test-Path $tr) { & $PY "$RT\traffic_sim_r39t.py" $tr | Tee-Object "$OUT\traffic_sim_pool$PoolGib.txt" }
  }
  'torture' {
    $mx = New-Object System.Threading.Mutex($false, 'Global\glm53_r39t_gpu_run')
    try { [void]$mx.WaitOne() } catch [System.Threading.AbandonedMutexException] { }
    try {
      if (Wait-Gpu) {
        Clear-R
        & $PY -u -W ignore "$RT\test_pool_r39t.py" --json "$OUT\pool_torture_r39t.json" *>> $log
        "exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
      } else { Write-Host "GPU / RAM busy after $WaitMinutes min: torture test not run" }
    } finally { $mx.ReleaseMutex(); $mx.Dispose() }
    Get-Content $log | Select-String '^\[r39t torture\]' | Select-Object -Last 24 | ForEach-Object { $_.Line }
  }
  'table' { Show-Table }
  'split' { $t = if ($Tag) { $Tag } else { 'prof' }; Show-Split "$OUT\traffic_$t.json" }
}
if (-not $noGpu -and (Test-Path $log)) {
  Get-Content $log | Where-Object { $_ -notmatch '^\s+\[pool\]|^\[SPEED SUMMARY\]' } | Select-Object -Last 25 | ForEach-Object { $_.Substring(0, [math]::Min(400, $_.Length)) }
}
