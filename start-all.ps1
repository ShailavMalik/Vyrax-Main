param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendCommand = "Set-Location '$repoRoot'; .\\.venv\\Scripts\\python.exe -m backend.web_app"
$frontendCommand = "Set-Location '$repoRoot\\frontend'; npm run dev"

$pythonExe = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$backendEntryFile = Join-Path $repoRoot "backend\\web_app.py"
$frontendPackage = Join-Path $repoRoot "frontend\\package.json"

if (-not (Test-Path $pythonExe)) {
    throw "Python virtual environment not found at '$pythonExe'."
}

if (-not (Test-Path $backendEntryFile)) {
    throw "Backend entry point not found at '$backendEntryFile'."
}

if (-not (Test-Path $frontendPackage)) {
    throw "Frontend package.json not found at '$frontendPackage'."
}

if ($DryRun) {
    Write-Host "Dry run: would run backend job with:" -ForegroundColor Cyan
    Write-Host "  $backendCommand"
    Write-Host "Dry run: would run frontend job with:" -ForegroundColor Cyan
    Write-Host "  $frontendCommand"
    exit 0
}

Get-Job -Name vyrax_backend, vyrax_frontend -ErrorAction SilentlyContinue |
    Stop-Job -PassThru |
    Remove-Job -Force

# Prevent stale backend instances from keeping port 8000 and serving old camera state.
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -eq 'python.exe' -and
        $_.CommandLine -like '*-m backend.web_app*'
    } |
    ForEach-Object {
        try {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
            Write-Host "Stopped stale backend process PID=$($_.ProcessId)" -ForegroundColor DarkYellow
        }
        catch {
            # Process may already have exited; ignore and continue startup.
        }
    }

$backendJob = Start-Job -Name vyrax_backend -ScriptBlock {
    Set-Location $using:repoRoot
    & $using:pythonExe -m backend.web_app 2>&1
}

$frontendJob = Start-Job -Name vyrax_frontend -ScriptBlock {
    Set-Location "$using:repoRoot\frontend"
    npm run dev 2>&1
}

Write-Host "Started backend and frontend in this terminal. Press Ctrl+C to stop both." -ForegroundColor Green

try {
    while ($true) {
        Receive-Job -Job $backendJob -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "[backend] $_" }

        Receive-Job -Job $frontendJob -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "[frontend] $_" }

        $states = @($backendJob.State, $frontendJob.State)
        if ($states -contains 'Failed' -or $states -contains 'Stopped' -or $states -contains 'Completed') {
            Write-Host "One process exited. Stopping remaining process." -ForegroundColor Yellow
            break
        }

        Wait-Job -Job $backendJob, $frontendJob -Any -Timeout 1 | Out-Null
    }
}
finally {
    Get-Job -Name vyrax_backend, vyrax_frontend -ErrorAction SilentlyContinue |
        Stop-Job -PassThru |
        Remove-Job -Force
}