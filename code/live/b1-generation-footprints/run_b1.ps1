$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
foreach ($m in @("qwen2.5-1.5b","gemma-2-2b")) {
  & $py b1-generation-footprints\src\generate_b1.py --model $m
  Write-Output "=== b1 gen $m exit $LASTEXITCODE ==="
  & $py b1-generation-footprints\src\analyze_b1.py --model $m
  Write-Output "=== b1 analyze $m exit $LASTEXITCODE ==="
}
Write-Output "B1 DONE"
