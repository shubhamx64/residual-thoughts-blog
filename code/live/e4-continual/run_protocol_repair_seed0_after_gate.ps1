$ErrorActionPreference = "Stop"
$root = "C:\Users\shubh\Downloads\s2path"
Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$gatePath = Join-Path $root "e4-continual\results\protocol_repair_gate.json"
$queueLog = Join-Path $root "e4-continual\results\protocol_repair_seed0_queue.log"

function Write-Status([string]$message) {
    Add-Content -LiteralPath $queueLog -Value "$(Get-Date -Format o) $message"
}

Write-Status "waiting for phase-A/phase-B validation gate"
while (-not (Test-Path -LiteralPath $gatePath)) {
    $scout = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "run_protocol_repair_after_scope\.ps1"
    })
    if ($scout.Count -eq 0) {
        Write-Status "gate runner exited without writing a gate; stopping"
        exit 0
    }
    Start-Sleep -Seconds 30
}

$gate = Get-Content -LiteralPath $gatePath | ConvertFrom-Json
if (-not $gate.validation_gate_passed) {
    Write-Status "phase-B validation gate failed; no arm comparison"
    exit 0
}
if (-not $gate.untouched_test_acquisition) {
    Write-Status "validation improved but untouched test did not; no arm comparison"
    exit 0
}

$steps = [int]$gate.phase_b_best_step
Write-Status "clean acquisition gate passed at step $steps; recomputing Fisher"
& $python e4-continual/src/prep_qwen_masks.py --fisher `
    --ckpt e4-continual/results/ckpt_A_qwen_repair.pt `
    --train-file e4-continual/data/repair_train_math.jsonl `
    --out e4-continual/data/qwen2.5-1.5b/mask_fisher_repair.npz `
    *>> e4-continual/results/protocol_repair_fisher.console.log
if ($LASTEXITCODE -ne 0) {
    Write-Status "Fisher preparation failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}

$maskDir = "e4-continual/data/qwen2.5-1.5b"
$masks = @{
    random = "$maskDir/mask_random.npz"
    weights = "$maskDir/mask_weights.npz"
    footprint = "$maskDir/mask_footprint.npz"
    join = "$maskDir/mask_join.npz"
    fisher = "$maskDir/mask_fisher_repair.npz"
}

foreach ($arm in @("baseline", "random", "weights", "footprint", "join", "fisher")) {
    Write-Status "starting seed-0 arm $arm for $steps steps"
    $args = @(
        "e4-continual/src/train_e4.py",
        "--phase", "B",
        "--arm", $arm,
        "--model", "qwen2.5-1.5b",
        "--steps", "$steps",
        "--lr", "2e-6",
        "--eval-every", "25",
        "--train-file", "repair_train_code.jsonl",
        "--eval-split-file", "e4-continual/data/repair_val_code.jsonl",
        "--init", "e4-continual/results/ckpt_A_qwen_repair.pt",
        "--seed", "0",
        "--tag-suffix", "_qwen_repair_s0"
    )
    if ($arm -ne "baseline") {
        $args += @("--mask-file", $masks[$arm])
    }
    & $python @args *>> "e4-continual/results/protocol_repair_$($arm)_s0.console.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Status "arm $arm failed with exit $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "finished seed-0 arm $arm"
}

Write-Status "all seed-0 arms complete"
