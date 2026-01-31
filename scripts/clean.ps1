# PowerShell script for cleaning LaTeX auxiliary files
# Usage: .\clean.ps1 [full]

param(
    [string]$Mode = "normal"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LaTeX Cleanup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Define file extensions to remove
$auxExtensions = @(
    "*.aux", "*.log", "*.out", "*.toc", "*.lof", "*.lot",
    "*.bbl", "*.blg", "*.synctex.gz", "*.fdb_latexmk", "*.fls",
    "*.nav", "*.snm", "*.vrb", "*.dvi", "*.ps"
)

# Count files before deletion
$fileCount = 0
foreach ($ext in $auxExtensions) {
    $files = Get-ChildItem -Filter $ext -ErrorAction SilentlyContinue
    $fileCount += $files.Count
}

if ($fileCount -eq 0) {
    Write-Host "No auxiliary files to clean." -ForegroundColor Green
    Write-Host ""
    exit 0
}

Write-Host "Found $fileCount auxiliary file(s) to remove:" -ForegroundColor Yellow
foreach ($ext in $auxExtensions) {
    $files = Get-ChildItem -Filter $ext -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        Write-Host "  - $($file.Name)" -ForegroundColor Gray
    }
}
Write-Host ""

# Remove auxiliary files
Write-Host "Removing auxiliary files..." -ForegroundColor Yellow
foreach ($ext in $auxExtensions) {
    Remove-Item -Path $ext -Force -ErrorAction SilentlyContinue
}
Write-Host "Auxiliary files removed!" -ForegroundColor Green
Write-Host ""

# Full clean mode (also remove PDF)
if ($Mode -eq "full" -or $Mode -eq "all") {
    if (Test-Path "paper.pdf") {
        Write-Host "Removing paper.pdf (full clean mode)..." -ForegroundColor Yellow
        Remove-Item "paper.pdf" -Force
        Write-Host "paper.pdf removed!" -ForegroundColor Green
    } else {
        Write-Host "paper.pdf not found." -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To recompile, run:" -ForegroundColor Cyan
Write-Host "  .\compile.ps1" -ForegroundColor White
