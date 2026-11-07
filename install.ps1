# docagent installer for Windows.
#
# Paste this into PowerShell:
#   irm https://raw.githubusercontent.com/Camdoni/doc_agent/main/install.ps1 | iex
#
# It makes sure three things are installed, and only installs what is missing:
#   1. Python 3.10 or newer
#   2. Ollama (runs the AI model on your own computer)
#   3. docagent itself
#
# To see what it WOULD do without installing anything, download it and run:
#   .\install.ps1 -DryRun

param([switch]$DryRun)

$ErrorActionPreference = "Stop"
$ZipUrl = "https://github.com/Camdoni/doc_agent/archive/refs/heads/main.zip"

function Say($message)  { Write-Host "==> $message" -ForegroundColor Cyan }
function Note($message) { Write-Host "    $message" -ForegroundColor DarkGray }

# After installing something, Windows doesn't update the PATH of the window you are
# already in. Re-read it so we can find what we just installed.
function Update-SessionPath {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

# Returns the first working Python 3.10+ as @{Exe=..; Args=..}, or $null.
function Get-Python {
    $candidates = @(
        @{ Exe = "py";      Args = @("-3") },
        @{ Exe = "python";  Args = @() },
        @{ Exe = "python3"; Args = @() }
    )
    foreach ($c in $candidates) {
        if (-not (Get-Command $c.Exe -ErrorAction SilentlyContinue)) { continue }
        try {
            # The Microsoft Store "python" stub fails this check, which is what we want.
            $version = & $c.Exe @($c.Args + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")) 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match '^(\d+)\.(\d+)$') {
                if ([int]$Matches[1] -gt 3 -or ([int]$Matches[1] -eq 3 -and [int]$Matches[2] -ge 10)) { return $c }
            }
        } catch { }
    }
    return $null
}

function Install-WithWinget($id, $name) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "I need to install $name but this PC has no 'winget'. Please install $name by hand (see the README), then run this again."
    }
    winget install -e --id $id --accept-package-agreements --accept-source-agreements
    Update-SessionPath
}

if ($DryRun) { Write-Host "DRY RUN: nothing will be installed.`n" -ForegroundColor Yellow }

# 1. Python ---------------------------------------------------------------
Say "Checking for Python 3.10 or newer"
$python = Get-Python
if ($python) {
    Note "Found it."
} elseif ($DryRun) {
    Note "Not found. I would install Python 3.12 with winget."
} else {
    Note "Not found. Installing Python 3.12 (this takes a minute)..."
    Install-WithWinget "Python.Python.3.12" "Python"
    $python = Get-Python
    if (-not $python) {
        throw "Python installed, but this window can't see it yet. Close PowerShell, open a new one, and run the install line again."
    }
}

# 2. Ollama ---------------------------------------------------------------
Say "Checking for Ollama"
$ollamaFolder = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
if ((Get-Command ollama -ErrorAction SilentlyContinue) -or (Test-Path (Join-Path $ollamaFolder "ollama.exe"))) {
    Note "Found it."
} elseif ($DryRun) {
    Note "Not found. I would install Ollama with winget."
} else {
    Note "Not found. Installing Ollama..."
    Install-WithWinget "Ollama.Ollama" "Ollama"
}

# 3. docagent -------------------------------------------------------------
Say "Installing docagent"
if ($DryRun) {
    Note "I would install 'pipx' (a tool that installs Python programs cleanly), then docagent from:"
    Note $ZipUrl
} else {
    # pipx gives docagent its own private environment, so it can't clash with other Python things.
    # It is the only reason for the next three lines.
    & $python.Exe @($python.Args + @("-m", "pip", "install", "--user", "--quiet", "--upgrade", "pipx"))
    & $python.Exe @($python.Args + @("-m", "pipx", "ensurepath"))
    # --force makes running this again an upgrade instead of an error.
    # This step downloads PyTorch and friends (a few hundred MB), so it can take several minutes.
    & $python.Exe @($python.Args + @("-m", "pipx", "install", "--force", $ZipUrl))
    if ($LASTEXITCODE -ne 0) { throw "Installing docagent failed. The message above says why." }
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run finished. Nothing was changed." -ForegroundColor Yellow
} else {
    Write-Host "Done!" -ForegroundColor Green
    Write-Host "Close this window, open a NEW PowerShell, and type:   docagent"
    Write-Host "The first time, it asks where your notes are and downloads the AI model (about 2 GB)."
}
