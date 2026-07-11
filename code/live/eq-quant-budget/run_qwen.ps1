$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
& $py eq-quant-budget\src\maps_eq.py --model qwen2.5-1.5b
Write-Output "=== maps exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_eq.py --model qwen2.5-1.5b
Write-Output "=== eq exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_eq2.py --model qwen2.5-1.5b
Write-Output "=== eq2 exit $LASTEXITCODE ==="
Write-Output "EQ QWEN DONE"
