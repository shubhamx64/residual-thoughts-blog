$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"

& $py e4-continual\src\train_e4.py --phase A --train-file train_B_code.jsonl --steps 150 --tag-suffix 3
Write-Output "=== phase A3 exit $LASTEXITCODE ==="
foreach ($arm in @("baseline","join_code")) {
  & $py e4-continual\src\train_e4.py --phase B --arm $arm --init e4-continual\results\ckpt_A3.pt --train-file train_A_math.jsonl --probe-class code --tag-suffix 3
  Write-Output "=== phase B3 $arm exit $LASTEXITCODE ==="
}
Write-Output "E4R ALL DONE"
