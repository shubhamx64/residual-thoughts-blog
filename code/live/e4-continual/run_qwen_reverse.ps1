$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
$m = "qwen2.5-1.5b"
$md = "e4-continual\data\qwen2.5-1.5b"

# reverse direction: code -> math, healthy-reference schedule (phase A 150 steps)
& $py e4-continual\src\train_e4.py --phase A --model $m --train-file train_B_code.jsonl `
    --steps 150 --tag-suffix _qwenR
Write-Output "=== phase A_qwenR (code, 150) exit $LASTEXITCODE ==="

& $py e4-continual\src\train_e4.py --phase B --arm baseline --model $m `
    --init e4-continual\results\ckpt_A_qwenR.pt --train-file train_A_math.jsonl `
    --probe-class code --tag-suffix _qwenR
Write-Output "=== phase B_baseline_qwenR exit $LASTEXITCODE ==="

& $py e4-continual\src\train_e4.py --phase B --arm join_code --model $m `
    --init e4-continual\results\ckpt_A_qwenR.pt --train-file train_A_math.jsonl `
    --probe-class code --mask-file "$md\mask_join_code.npz" --tag-suffix _qwenR
Write-Output "=== phase B_join_code_qwenR exit $LASTEXITCODE ==="
Write-Output "E4 QWEN REVERSE DONE"
