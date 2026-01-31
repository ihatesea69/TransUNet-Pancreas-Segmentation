# PowerShell script for compiling LaTeX paper on Windows
# Usage: .\compile.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  LaTeX Paper Compilation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if pdflatex is available
Write-Host "Checking for LaTeX installation..." -ForegroundColor Yellow
try {
    $latexVersion = pdflatex --version 2>&1 | Select-Object -First 1
    Write-Host "Found: $latexVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: pdflatex not found!" -ForegroundColor Red
    Write-Host "Please install MiKTeX or TeX Live:" -ForegroundColor Yellow
    Write-Host "  - MiKTeX: https://miktex.org/download" -ForegroundColor Yellow
    Write-Host "  - TeX Live: https://www.tug.org/texlive/" -ForegroundColor Yellow
    exit 1
}

# Check if IEEEtran.cls exists (common issue)
Write-Host "Checking for required LaTeX packages..." -ForegroundColor Yellow
$ieeetranCheck = kpsewhich IEEEtran.cls 2>&1
if (-Not $ieeetranCheck -or $LASTEXITCODE -ne 0) {
    Write-Host "ERROR: IEEEtran.cls not found!" -ForegroundColor Red
    Write-Host "Required LaTeX packages are missing." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please run the package installation script first:" -ForegroundColor Cyan
    Write-Host "  .\install_latex_packages.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Or install packages manually in MiKTeX Console." -ForegroundColor Yellow
    exit 1
}
Write-Host "All required packages found!" -ForegroundColor Green

Write-Host ""

# Check if paper.tex exists
if (-Not (Test-Path "paper.tex")) {
    Write-Host "ERROR: paper.tex not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory." -ForegroundColor Yellow
    exit 1
}

# Check if references.bib exists
if (-Not (Test-Path "references.bib")) {
    Write-Host "WARNING: references.bib not found!" -ForegroundColor Yellow
    Write-Host "Bibliography will be empty." -ForegroundColor Yellow
    Write-Host ""
}

# Compilation sequence
Write-Host "===== Step 1/4: First pdflatex pass =====" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode paper.tex
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: First pdflatex pass failed!" -ForegroundColor Red
    Write-Host "Check paper.log for details." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "===== Step 2/4: Running BibTeX =====" -ForegroundColor Cyan
if (Test-Path "references.bib") {
    bibtex paper
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: BibTeX had errors (continuing anyway)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping BibTeX (no references.bib)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===== Step 3/4: Second pdflatex pass =====" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode paper.tex
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Second pdflatex pass failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===== Step 4/4: Third pdflatex pass =====" -ForegroundColor Cyan
pdflatex -interaction=nonstopmode paper.tex
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Third pdflatex pass failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Compilation successful!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if PDF was generated
if (Test-Path "paper.pdf") {
    $pdfSize = (Get-Item "paper.pdf").Length / 1KB
    Write-Host "Output: paper.pdf ($([math]::Round($pdfSize, 2)) KB)" -ForegroundColor Green
    
    # Ask to open PDF
    Write-Host ""
    $open = Read-Host "Open paper.pdf? (Y/n)"
    if ($open -ne "n" -and $open -ne "N") {
        Start-Process "paper.pdf"
    }
} else {
    Write-Host "WARNING: paper.pdf not found!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Auxiliary files created:" -ForegroundColor Cyan
Get-ChildItem "paper.*" | Where-Object { $_.Extension -ne ".tex" -and $_.Extension -ne ".pdf" } | ForEach-Object {
    Write-Host "  - $($_.Name)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "To clean auxiliary files, run:" -ForegroundColor Yellow
Write-Host "  .\clean.ps1" -ForegroundColor White
