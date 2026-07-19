$ErrorActionPreference = "Stop"
$root = "C:\Users\shubh\Downloads\s2path"
Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$seed0Log = Join-Path $root "e4-continual\results\protocol_repair_seed0_queue.log"
$queueLog = Join-Path $root "e4-continual\results\protocol_repair_seeds1_4_queue.log"
$gatePath = Join-Path $root "e4-continual\results\protocol_repair_gate.json"

function Write-Status([string]$message) {
    Add-Content -LiteralPath $queueLog -Value "$(Get-Date -Format o) $message"
}

Write-Status "waiting for all seed-0 arms"
while ($true) {
    if ((Test-Path -LiteralPath $seed0Log) -and
        (Select-String -LiteralPath $seed0Log -SimpleMatch "all seed-0 arms complete" -Quiet)) {
        break
    }
    $seed0Runner = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "run_protocol_repair_seed0_after_gate\.ps1"
    })
    if ($seed0Runner.Count -eq 0) {
        Write-Status "seed-0 runner exited before completing; stopping"
        exit 0
    }
    Start-Sleep -Seconds 30
}

$gate = Get-Content -LiteralPath $gatePath | ConvertFrom-Json
$steps = [int]$gate.phase_b_best_step
$maskDir = "e4-continual/data/qwen2.5-1.5b"
$fixedMasks = @{
    weights = "$maskDir/mask_weights.npz"
    footprint = "$maskDir/mask_footprint.npz"
    join = "$maskDir/mask_join.npz"
    fisher = "$maskDir/mask_fisher_repair.npz"
}

foreach ($seed in 1..4) {
    foreach ($arm in @("baseline", "random", "weights", "footprint", "join", "fisher")) {
        Write-Status "starting seed $seed arm $arm for $steps steps"
        $args = @(
            "e4-continual/src/train_e4.py",
            "--phase", "B",
            "--arm", $arm,
            "--model", "qwen2.5-1.5b",
            "--steps", "$steps",
            "--lr", "2e-6",
            "--eval-every", "$steps",
            "--train-file", "repair_train_code.jsonl",
            "--eval-split-file", "e4-continual/data/repair_val_code.jsonl",
            "--init", "e4-continual/results/ckpt_A_qwen_repair.pt",
            "--seed", "$seed",
            "--tag-suffix", "_qwen_repair_s$seed"
        )
        if ($arm -eq "random") {
            $args += @("--mask-file", "$maskDir/mask_random_s$seed.npz")
        } elseif ($arm -ne "baseline") {
            $args += @("--mask-file", $fixedMasks[$arm])
        }
        & $python @args *>> "e4-continual/results/protocol_repair_$($arm)_s$seed.console.log"
        if ($LASTEXITCODE -ne 0) {
            Write-Status "seed $seed arm $arm failed with exit $LASTEXITCODE"
            exit $LASTEXITCODE
        }
        Write-Status "finished seed $seed arm $arm"
    }
}

& $python e4-continual/src/analyze_protocol_repair.py *>> e4-continual/results/protocol_repair_analysis.console.log
if ($LASTEXITCODE -ne 0) {
    Write-Status "analysis failed with exit $LASTEXITCODE"
    exit $LASTEXITCODE
}
Write-Status "all five seeds complete and analyzed"
