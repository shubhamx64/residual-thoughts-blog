$ErrorActionPreference = "Continue"
Set-Location "C:\Users\shubh\Downloads\s2path"
$py = ".venv\Scripts\python.exe"
$m = "qwen2.5-1.5b"
$md = "e4-continual\data\qwen2.5-1.5b"

# primary direction: math -> code
& $py e4-continual\src\train_e4.py --phase A --model $m --tag-suffix _qwen
Write-Output "=== phase A_qwen exit $LASTEXITCODE ==="

# fisher mask needs the after-A checkpoint
& $py e4-continual\src\prep_qwen_masks.py --fisher
Write-Output "=== fisher mask exit $LASTEXITCODE ==="

foreach ($arm in @("baseline","random","weights","footprint","join","fisher")) {
  $maskArg = @()
  if ($arm -ne "baseline") { $maskArg = @("--mask-file", "$md\mask_$arm.npz") }
  & $py e4-continual\src\train_e4.py --phase B --arm $arm --model $m `
      --init e4-continual\results\ckpt_A_qwen.pt --tag-suffix _qwen @maskArg
  Write-Output "=== phase B_$arm`_qwen exit $LASTEXITCODE ==="
}
Write-Output "E4 QWEN PRIMARY DONE"
