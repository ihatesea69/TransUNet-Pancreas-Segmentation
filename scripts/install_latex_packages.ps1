# PowerShell script to install required LaTeX packages for the paper
# Usage: .\install_latex_packages.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LaTeX Package Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if MiKTeX is installed
Write-Host "Checking MiKTeX installation..." -ForegroundColor Yellow
try {
    $miktexVersion = pdflatex --version 2>&1 | Select-Object -First 1
    Write-Host "Found: $miktexVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: MiKTeX not found!" -ForegroundColor Red
    Write-Host "Please install MiKTeX from: https://miktex.org/download" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if mpm (MiKTeX Package Manager) is available
Write-Host "Checking MiKTeX Package Manager..." -ForegroundColor Yellow
try {
    mpm --version | Out-Null
    Write-Host "MiKTeX Package Manager found!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: mpm command not found!" -ForegroundColor Red
    Write-Host "Please ensure MiKTeX bin directory is in PATH." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# List of required packages
$packages = @(
    "IEEEtran",      # IEEE document class
    "cite",          # Citation package
    "amsmath",       # Math symbols
    "amssymb",       # Additional math symbols
    "graphicx",      # Graphics support
    "booktabs",      # Professional tables
    "hyperref",      # Hyperlinks
    "url",           # URL formatting
    "multirow",      # Table multirow
    "array",         # Enhanced arrays/tables
    "algorithm2e",   # Algorithm typesetting
    "caption",       # Caption customization
    "subcaption",    # Subfigures
    "xcolor",        # Colors
    "amsfonts"       # AMS fonts
)

Write-Host "Installing required LaTeX packages..." -ForegroundColor Cyan
Write-Host "This may take a few minutes..." -ForegroundColor Yellow
Write-Host ""

$successCount = 0
$failCount = 0

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor White -NoNewline
    
    try {
        # Install package silently
        $result = mpm --verbose --install=$package 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $successCount++
        } else {
            # Check if already installed
            if ($result -match "already installed") {
                Write-Host " Already installed" -ForegroundColor Gray
                $successCount++
            } else {
                Write-Host " FAILED" -ForegroundColor Red
                $failCount++
            }
        }
    } catch {
        Write-Host " FAILED" -ForegroundColor Red
        $failCount++
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "  Successful: $successCount packages" -ForegroundColor Green
Write-Host "  Failed: $failCount packages" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Gray" })
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($failCount -gt 0) {
    Write-Host "Some packages failed to install." -ForegroundColor Yellow
    Write-Host "You can install them manually using MiKTeX Console:" -ForegroundColor Yellow
    Write-Host "  1. Open MiKTeX Console" -ForegroundColor White
    Write-Host "  2. Go to 'Packages' tab" -ForegroundColor White
    Write-Host "  3. Search and install missing packages" -ForegroundColor White
    Write-Host ""
}

# Refresh file name database
Write-Host "Refreshing MiKTeX file name database..." -ForegroundColor Yellow
try {
    initexmf --update-fndb 2>&1 | Out-Null
    Write-Host "Database refreshed!" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not refresh database automatically" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "You can now compile the paper:" -ForegroundColor Cyan
Write-Host "  .\compile.ps1" -ForegroundColor White
Write-Host ""

# Ask if user wants to compile now
$compile = Read-Host "Compile paper now? (Y/n)"
if ($compile -ne "n" -and $compile -ne "N") {
    Write-Host ""
    Write-Host "Starting compilation..." -ForegroundColor Cyan
    & ".\compile.ps1"
}
