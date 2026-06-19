<#
PowerShell setup script for Windows:
 - Creates a virtual environment named .venv
 - Activates it (for the current shell)
 - Upgrades pip and installs packages from requirements.txt
Usage: In PowerShell run: .\setup_env.ps1
#>

$venvPath = Join-Path $PSScriptRoot '.venv'
if (-Not (Test-Path $venvPath)) {
    python -m venv $venvPath
}

Write-Host "Activating virtual environment at $venvPath"
& "$venvPath\Scripts\Activate.ps1"

Write-Host 'Upgrading pip...'
python -m pip install --upgrade pip

Write-Host 'Installing requirements from requirements.txt...'
pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host 'Saving installed packages to requirements-installed.txt'
pip freeze > (Join-Path $PSScriptRoot 'requirements-installed.txt')

Write-Host 'Environment setup complete.'
