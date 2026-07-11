$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
foreach ($m in @("qwen2.5-1.5b","gemma-2-2b","pythia-1.4b","tinyllama-1.1b")) {
  & $py e3-sufficiency\src\pairs_and_readers.py --model $m
  Write-Output "=== pairs $m exit $LASTEXITCODE ==="
  & $py e3-sufficiency\src\capture_coact.py --model $m
  Write-Output "=== coact $m exit $LASTEXITCODE ==="
}
Write-Output "E3 ALL DONE"
