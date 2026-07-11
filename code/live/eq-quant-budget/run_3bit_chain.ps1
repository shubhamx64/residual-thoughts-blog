$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"

& $py eq-quant-budget\src\run_sota.py --model qwen2.5-1.5b --bits 3 --arms rtn,gptq,gptq_rand,gptq_hdiag,gptq_fp
Write-Output "=== 1.5b w3 exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_sota.py --model qwen2.5-1.5b --bits 3 --protect-frac 0.03 --arms gptq_hdiag,gptq_fp
Write-Output "=== 1.5b w3 p3 exit $LASTEXITCODE ==="

& $py eq-quant-budget\src\run_sota.py --model qwen2.5-3b --bits 3 --arms gptq,gptq_rand,gptq_hdiag,gptq_fp
Write-Output "=== 3b w3 exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_sota.py --model qwen2.5-3b --bits 3 --protect-frac 0.03 --arms gptq_hdiag,gptq_fp
Write-Output "=== 3b w3 p3 exit $LASTEXITCODE ==="

& $py e1-footprint-stability\src\run_capture.py --model qwen2.5-7b
Write-Output "=== capture 7b exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_sota.py --model qwen2.5-7b --bits 4 --arms gptq,gptq_hdiag,gptq_fp
Write-Output "=== 7b w4 exit $LASTEXITCODE ==="
& $py eq-quant-budget\src\run_sota.py --model qwen2.5-7b --bits 3 --arms gptq,gptq_hdiag,gptq_fp
Write-Output "=== 7b w3 exit $LASTEXITCODE ==="
Write-Output "3BIT CHAIN DONE"
