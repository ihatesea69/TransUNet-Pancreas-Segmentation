# Simple compilation script - bypasses package checks
# Just tries to compile directly

Write-Host "Attempting direct compilation..." -ForegroundColor Cyan
Write-Host ""

# Try to compile - MiKTeX will handle missing packages
pdflatex -interaction=nonstopmode paper.tex

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "First pass successful! Running BibTeX..." -ForegroundColor Green
    bibtex paper
    
    Write-Host "Second pass..." -ForegroundColor Green
    pdflatex -interaction=nonstopmode paper.tex
    
    Write-Host "Final pass..." -ForegroundColor Green
    pdflatex -interaction=nonstopmode paper.tex
    
    Write-Host ""
    Write-Host "SUCCESS! Paper compiled." -ForegroundColor Green
    
    if (Test-Path "paper.pdf") {
        Start-Process "paper.pdf"
    }
} else {
    Write-Host ""
    Write-Host "Compilation failed. Check paper.log for errors." -ForegroundColor Red
    Write-Host ""
    Write-Host "Common fix: Open MiKTeX Console and install missing packages" -ForegroundColor Yellow
}
