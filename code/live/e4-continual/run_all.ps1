$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
Remove-Item e4-continual\results\*_smoke* -ErrorAction SilentlyContinue

& $py e4-continual\src\train_e4.py --phase A
Write-Output "=== phase A exit $LASTEXITCODE ==="
foreach ($arm in @("baseline","random","weights","join")) {
  & $py e4-continual\src\train_e4.py --phase B --arm $arm --init e4-continual\results\ckpt_A.pt
  Write-Output "=== phase B $arm exit $LASTEXITCODE ==="
}
Write-Output "E4 ALL DONE"
