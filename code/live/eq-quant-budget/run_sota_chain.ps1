$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"

& $py eq-quant-budget\src\run_sota.py --model qwen2.5-1.5b --arms gptq4_rand,gptq4_hdiag,gptq4_fp,gptq4_gain
Write-Output "=== sota 1.5b exit $LASTEXITCODE ==="

& $py e1-footprint-stability\src\run_capture.py --model qwen2.5-3b
Write-Output "=== capture 3b exit $LASTEXITCODE ==="

& $py eq-quant-budget\src\run_sota.py --model qwen2.5-3b
Write-Output "=== sota 3b exit $LASTEXITCODE ==="
Write-Output "SOTA CHAIN DONE"
