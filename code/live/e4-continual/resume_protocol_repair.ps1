$ErrorActionPreference = "Stop"
$root = "C:\Users\shubh\Downloads\s2path"
Set-Location -LiteralPath $root
$python = Join-Path $root ".venv\Scripts\python.exe"
$status = Join-Path $root "e4-continual\results\protocol_repair_seed0_queue.log"

function Write-Status([string]$message) {
    Add-Content -LiteralPath $status -Value "$(Get-Date -Format o) $message"
}

# Baseline, random, weights, and footprint seed-0 arms completed before the
# 2026-07-12 Codex-app-update pause. The partial join log is intentionally
# overwritten by this clean restart from the shared after-math checkpoint.
$masks = @{
    join = "e4-continual/data/qwen2.5-1.5b/mask_join.npz"
    fisher = "e4-continual/data/qwen2.5-1.5b/mask_fisher_repair.npz"
}

foreach ($arm in @("join", "fisher")) {
    Write-Status "resume: starting seed-0 arm $arm from step 0"
    & $python e4-continual/src/train_e4.py `
        --phase B --arm $arm --model qwen2.5-1.5b `
        --steps 200 --lr 2e-6 --eval-every 25 `
        --train-file repair_train_code.jsonl `
        --eval-split-file e4-continual/data/repair_val_code.jsonl `
        --init e4-continual/results/ckpt_A_qwen_repair.pt `
        --seed 0 --tag-suffix _qwen_repair_s0 `
        --mask-file $masks[$arm] `
        *>> "e4-continual/results/protocol_repair_$($arm)_s0.console.log"
    if ($LASTEXITCODE -ne 0) {
        Write-Status "resume: arm $arm failed with exit $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Status "resume: finished seed-0 arm $arm"
}

Write-Status "all seed-0 arms complete"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
    e4-continual/run_protocol_repair_seeds1_4.ps1
exit $LASTEXITCODE
