param(
    [string]$VenvPath = '.venv'
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $File $($Arguments -join ' ')"
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'This script requires Windows x86-64.'
}
$VenvRoot = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $VenvPath))
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$CheckPython = "import platform, struct, sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'; assert struct.calcsize('P') == 8 and platform.machine().lower() in ('amd64', 'x86_64'), 'x86-64 Python required'; print(sys.version); print(platform.machine())"

if (-not (Test-Path $VenvPython)) {
    if (Test-Path $VenvRoot) {
        throw 'The target folder already exists without a Windows venv. Choose another -VenvPath; nothing was deleted.'
    }
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Install 64-bit Python 3.12 with the Python launcher first. Existing Python 3.11 can remain installed.'
    }
    Invoke-Checked -File 'py' -Arguments @('-3.12', '-c', $CheckPython)
    Invoke-Checked -File 'py' -Arguments @('-3.12', '-m', 'venv', $VenvRoot)
}
Invoke-Checked -File $VenvPython -Arguments @('-c', $CheckPython)
$env:KERAS_BACKEND = 'tensorflow'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:KERAS_HOME = Join-Path $VenvRoot '.keras'
$env:MPLCONFIGDIR = Join-Path $VenvRoot '.matplotlib'
Invoke-Checked -File $VenvPython -Arguments @('-m', 'pip', 'install', '--index-url', 'https://pypi.org/simple', '-r', 'requirements.txt')
Invoke-Checked -File $VenvPython -Arguments @('-m', 'pip', 'check')
Invoke-Checked -File $VenvPython -Arguments @('scripts/check_environment.py', '--output', (Join-Path $VenvRoot 'environment.json'))
$Freeze = & $VenvPython -m pip freeze
if ($LASTEXITCODE -ne 0) { throw 'pip freeze failed.' }
$Freeze | Set-Content -Encoding UTF8 (Join-Path $VenvRoot 'requirements-windows.txt')
Write-Host "Environment check passed. Python: $VenvPython"
