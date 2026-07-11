$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"

foreach ($arm in @("footprint","fisher")) {
  & $py e4-continual\src\train_e4.py --phase B --arm $arm --init e4-continual\results\ckpt_A.pt
  Write-Output "=== phase B $arm exit $LASTEXITCODE ==="
}

& $py e4-continual\src\train_e4.py --phase A --train-file train_B_code.jsonl --tag-suffix 2
Write-Output "=== phase A2 exit $LASTEXITCODE ==="
& $py e4-continual\src\train_e4.py --phase B --arm baseline --init e4-continual\results\ckpt_A2.pt --train-file train_A_math.jsonl --probe-class code --tag-suffix 2
Write-Output "=== phase B2 baseline exit $LASTEXITCODE ==="
& $py e4-continual\src\train_e4.py --phase B --arm join_code --init e4-continual\results\ckpt_A2.pt --train-file train_A_math.jsonl --probe-class code --tag-suffix 2
Write-Output "=== phase B2 join_code exit $LASTEXITCODE ==="
Write-Output "E4X ALL DONE"
