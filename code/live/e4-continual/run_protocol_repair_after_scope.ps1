param(
    [int[]]$WaitForPids = @(17224, 9756)
)

$ErrorActionPreference = "Stop"
$root = "C:\Users\shubh\Downloads\s2path"
Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$status = Join-Path $root "e4-continual\results\protocol_repair_queue.log"

function Write-Status([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Add-Content -LiteralPath $status -Value $line
}

Write-Status "queued; waiting for interface-localization runner PIDs $($WaitForPids -join ',')"
while ($true) {
    $alive = @($WaitForPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    $trainers = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "train_e4\.py"
    })
    if ($alive.Count -eq 0 -and $trainers.Count -eq 0) {
        break
    }
    Start-Sleep -Seconds 30
}

Write-Status "starting Qwen protocol-repair phase A"
& $python e4-continual/src/train_e4.py `
    --phase A --arm baseline --model qwen2.5-1.5b `
    --steps 300 --lr 2e-6 --eval-every 25 `
    --train-file repair_train_math.jsonl `
    --eval-split-file e4-continual/data/repair_val_math.jsonl `
    --select-best-split --tag-suffix _qwen_repair `
    --ckpt-name ckpt_A_qwen_repair.pt `
    *>> e4-continual/results/protocol_repair_phaseA.console.log
if ($LASTEXITCODE -ne 0) {
    Write-Status "phase A failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

$phaseALog = Get-Content -LiteralPath e4-continual/results/log_A_qwen_repair.jsonl |
    ForEach-Object { $_ | ConvertFrom-Json }
$a0 = $phaseALog | Where-Object step -eq 0
$abest = $phaseALog | Sort-Object nll_split | Select-Object -First 1
if ($abest.step -eq 0 -or $abest.nll_split -ge $a0.nll_split) {
    Write-Status "phase A gate failed: validation math did not improve"
    exit 0
}
Write-Status "phase A gate passed at step $($abest.step); starting phase B baseline scout"

& $python e4-continual/src/train_e4.py `
    --phase B --arm baseline --model qwen2.5-1.5b `
    --steps 200 --lr 2e-6 --eval-every 25 `
    --train-file repair_train_code.jsonl `
    --eval-split-file e4-continual/data/repair_val_code.jsonl `
    --init e4-continual/results/ckpt_A_qwen_repair.pt `
    --tag-suffix _qwen_repair_scout `
    *>> e4-continual/results/protocol_repair_phaseB_scout.console.log
if ($LASTEXITCODE -ne 0) {
    Write-Status "phase B scout failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

$phaseBLog = Get-Content -LiteralPath e4-continual/results/log_B_baseline_qwen_repair_scout.jsonl |
    ForEach-Object { $_ | ConvertFrom-Json }
$b0 = $phaseBLog | Where-Object step -eq 0
$bbest = $phaseBLog | Sort-Object nll_split,step | Select-Object -First 1
$gate = [ordered]@{
    phase_a_best_step = [int]$abest.step
    phase_a_validation_nll_start = [double]$a0.nll_split
    phase_a_validation_nll_best = [double]$abest.nll_split
    phase_a_test_math_start = [double]$a0.ppl_math
    phase_a_test_math_at_selected = [double]$abest.ppl_math
    phase_b_best_step = [int]$bbest.step
    phase_b_validation_nll_start = [double]$b0.nll_split
    phase_b_validation_nll_best = [double]$bbest.nll_split
    phase_b_test_code_start = [double]$b0.ppl_code
    phase_b_test_code_at_selected = [double]$bbest.ppl_code
    validation_gate_passed = [bool]($bbest.step -gt 0 -and $bbest.nll_split -lt $b0.nll_split)
    untouched_test_acquisition = [bool]($bbest.step -gt 0 -and $bbest.ppl_code -lt $b0.ppl_code)
}
$gate | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath e4-continual/results/protocol_repair_gate.json
Write-Status "phase B scout complete; validation gate=$($gate.validation_gate_passed), test acquisition=$($gate.untouched_test_acquisition), selected step=$($gate.phase_b_best_step)"
