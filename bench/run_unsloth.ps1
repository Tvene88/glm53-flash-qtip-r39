# Serves unsloth/GLM-5.3-Flash-GGUF UD-IQ2_XXS with llama-server (README.md section 3.2), checks the rendered prompt
# (section 4), runs the smoke test and then HumanEval + the GSM8K sample, and stops the server.
# If the smoke test shows fewer than -SlowBelow tokens per second, the fixed subsets (-SubHe / -SubGsm) are used.
param([string]$Python = $env:R39_PYTHON, [string]$Root = (Split-Path -Parent $PSScriptRoot),
      # LlamaDir: folder of a prebuilt llama.cpp server build (llama-server.exe) that supports the glm5-next architecture.
      # LlamaVendor: optional folder of extra runtime DLLs that build needs (added to PATH). Gguf: first file of the model.
      [string]$LlamaDir = $env:LLAMA_SERVER_DIR,
      [string]$LlamaVendor = $env:LLAMA_VENDOR_DIR,
      [string]$Gguf = 'C:\models\competitors\unsloth-GLM-5.3-Flash-UD-IQ2_XXS\UD-IQ2_XXS\GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf',
      [int]$CpuMoe = 36, [int]$ReasoningBudget = -1, [double]$SlowBelow = 8.0, [int]$SubHe = 80, [int]$SubGsm = 100, [switch]$SmokeOnly, [string[]]$ExtraServerArgs = @())
$Bench = "$Root\bench"; $Runs = "$Bench\runs"; $Kind = 'unsloth'
$Port = 8188; $Url = "http://127.0.0.1:$Port/v1"
New-Item -ItemType Directory -Force $Runs | Out-Null
$slog = "$Runs\server_$Kind.log"
foreach ($p in $Gguf, "$LlamaDir\llama-server.exe") { if (-not (Test-Path $p)) { "missing $p"; exit 2 } }
$used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if ($used -gt 4096) { "GPU busy ($used MiB): not started"; exit 2 }
if (-not $LlamaDir) { "set -LlamaDir or LLAMA_SERVER_DIR"; exit 2 }
$env:PATH = if ($LlamaVendor) { "$LlamaDir;$LlamaVendor;$env:PATH" } else { "$LlamaDir;$env:PATH" }
$sargs = @('-m', $Gguf, '--host', '127.0.0.1', '--port', "$Port", '--alias', 'unsloth-ud-iq2_xxs',
  '-c', '8192', '-np', '1', '-t', '8', '-ngl', '99', '--n-cpu-moe', "$CpuMoe",
  '--jinja', '--reasoning-format', 'deepseek', '--reasoning-budget', "$ReasoningBudget", '--no-context-shift', '--cache-ram', '0') + $ExtraServerArgs
"START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') llama-server $($sargs -join ' ')" | Set-Content $slog
$srv = Start-Process -FilePath "$LlamaDir\llama-server.exe" -ArgumentList $sargs -WorkingDirectory $LlamaDir -RedirectStandardOutput "$slog.out" -RedirectStandardError "$slog.err" -PassThru -WindowStyle Hidden
$ready = $false
for ($i = 0; $i -lt 180; $i++) {
  Start-Sleep 10
  if ($srv.HasExited) { break }
  try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5; if ($h.status -eq 'ok') { $ready = $true; break } } catch {}
}
if (-not $ready) {
  "server not ready (exited=$($srv.HasExited))"
  Get-Content "$slog.err", "$slog.out" -ErrorAction SilentlyContinue | Select-Object -Last 45 | ForEach-Object { $_.Substring(0, [math]::Min(300, $_.Length)) }
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
  exit 3
}
$os = Get-CimInstance Win32_OperatingSystem
"server ready $(Get-Date -Format 'HH:mm:ss'); GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader); RAM free $([math]::Round($os.FreePhysicalMemory/1MB,1)) GiB"
try {
  try {
    $pr = (Invoke-RestMethod "http://127.0.0.1:$Port/apply-template" -Method Post -ContentType 'application/json' -Body (Get-Content "$Bench\preflight_body.json" -Raw)).prompt
    "rendered prompt head: $($pr.Substring(0, [math]::Min(70, $pr.Length)).Replace("`n", '\n'))"
    "rendered prompt tail: $($pr.Substring([math]::Max(0, $pr.Length - 40)).Replace("`n", '\n'))"
  } catch { "apply-template failed: $($_.Exception.Message)" }
  $Common = @('--base-url', $Url, '--max-tokens', '1024', '--temperature', '0', '--thinking', 'off', '--reasoning-effort', 'low',
              '--extra-body', "@$Bench\extra_body.json", '--concurrency', '1')
  Remove-Item "$Runs\smoke_$Kind" -Recurse -Force -ErrorAction SilentlyContinue
  & $Python "$Bench\bench_openai.py" --task humaneval --ids-file "$Bench\data\humaneval_order.json" --limit 3 --model $Kind --out "$Runs\smoke_$Kind" @Common
  "smoke exit=$LASTEXITCODE"
  if ($LASTEXITCODE -eq 0 -and -not $SmokeOnly) {
    $sm = Get-Content "$Runs\smoke_$Kind\items.jsonl" | ForEach-Object { $_ | ConvertFrom-Json }
    $tps = ($sm | Measure-Object tok_s_reported -Average).Average
    $slow = ($tps -lt $SlowBelow)
    "smoke decode speed $([math]::Round($tps, 2)) tok/s -> $(if ($slow) { "subset: HumanEval $SubHe, GSM8K $SubGsm" } else { 'full sets' })"
    $he = @('--task', 'humaneval', '--ids-file', "$Bench\data\humaneval_order.json", '--model', $Kind, '--out', "$Runs\$Kind\humaneval")
    if ($slow) { $he += @('--limit', "$SubHe") }
    if (Test-Path "$Runs\$Kind\humaneval\responses.jsonl") { $he += '--resume' }
    & $Python "$Bench\bench_openai.py" @he @Common; "humaneval exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
    $gs = @('--task', 'gsm8k', '--model', $Kind, '--out', "$Runs\$Kind\gsm8k")
    if ($slow) { $gs += @('--limit', "$SubGsm") }
    if (Test-Path "$Runs\$Kind\gsm8k\responses.jsonl") { $gs += '--resume' }
    & $Python "$Bench\bench_openai.py" @gs @Common; "gsm8k exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss')"
  }
}
finally {
  if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
  Start-Sleep 8
  "server stopped; GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
}
