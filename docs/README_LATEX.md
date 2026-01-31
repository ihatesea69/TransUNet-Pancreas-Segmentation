# LaTeX Paper Compilation Guide

## Quick Start

### Prerequisites
- LaTeX distribution installed:
  - **Windows**: MiKTeX or TeX Live
  - **macOS**: MacTeX
  - **Linux**: TeX Live

### Compile to PDF

**Method 1: Command Line (Recommended)**
```bash
# Full compilation sequence
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex

# Output: paper.pdf
```

**Method 2: Using Makefile**
```bash
make          # Compile paper
make clean    # Remove auxiliary files
make distclean # Remove all generated files
```

**Method 3: LaTeX Editor**
- **Overleaf**: Upload all files and click "Recompile"
- **TeXworks/TeXShop**: Press the "Typeset" button (set to pdfLaTeX + BibTeX)
- **VS Code**: Use LaTeX Workshop extension

---

## File Structure

```
paper/
├── paper.tex          # Main LaTeX document
├── references.bib     # BibTeX bibliography
├── figures/           # Figure directory (create this)
│   ├── architecture.pdf
│   ├── results.pdf
│   └── qualitative.pdf
└── paper.pdf          # Output (after compilation)
```

---

## Compilation Steps Explained

1. **First pdflatex**: Processes document, creates `.aux` file with citation keys
2. **bibtex**: Generates bibliography from `references.bib` based on citations
3. **Second pdflatex**: Incorporates bibliography, updates references
4. **Third pdflatex**: Resolves all cross-references (figures, tables, equations)

---

## Common Issues & Solutions

### Issue: "Undefined control sequence"
**Solution**: Missing package. Add to preamble:
```latex
\usepackage{packagename}
```

### Issue: "Citation undefined"
**Solution**: Run BibTeX step:
```bash
bibtex paper
pdflatex paper.tex
```

### Issue: "Figure not found"
**Solution**: Check file path and extension:
```latex
\includegraphics[width=0.8\columnwidth]{figures/architecture.pdf}
```

### Issue: References not appearing
**Solution**: Ensure `references.bib` exists and run full sequence

---

## Customization Guide

### Change Conference Format

**ACM Format:**
```latex
\documentclass[sigconf]{acmart}
```

**Springer LNCS:**
```latex
\documentclass{llncs}
```

**CVPR (IEEE-style):**
```latex
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage{cvpr}
```

### Add Figures

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.9\columnwidth]{figures/architecture.pdf}
\caption{Your caption here}
\label{fig:architecture}
\end{figure}
```

Reference with: `Figure~\ref{fig:architecture} shows...`

### Add Tables

```latex
\begin{table}[!t]
\caption{Your table title}
\label{tab:results}
\centering
\begin{tabular}{lcc}
\toprule
Model & DSC & HD95 \\
\midrule
Baseline & 0.72 & 18.3 \\
Ours & 0.81 & 11.2 \\
\bottomrule
\end{tabular}
\end{table}
```

### Add Algorithms

```latex
\usepackage{algorithm}
\usepackage{algorithmic}

\begin{algorithm}
\caption{Training Procedure}
\label{alg:training}
\begin{algorithmic}[1]
\STATE Initialize model parameters $\theta$
\FOR{epoch = 1 to $N$}
    \FOR{batch in training set}
        \STATE Compute loss $\mathcal{L}$
        \STATE Update $\theta$ using gradient descent
    \ENDFOR
\ENDFOR
\end{algorithmic}
\end{algorithm}
```

---

## Advanced Options

### Enable Draft Mode (faster compilation)
```latex
\documentclass[conference,draft]{IEEEtran}
```

### Add Line Numbers (for review)
```latex
\usepackage{lineno}
\linenumbers
```

### Include Supplementary Material
```latex
\appendix
\section{Supplementary Material}
Additional experimental results...
```

---

## Submission Checklist

- [ ] Compile without errors or warnings
- [ ] All figures display correctly
- [ ] All references are cited and appear in bibliography
- [ ] Cross-references (figures, tables, equations) are correct
- [ ] Page limit respected (typically 8-10 pages for conferences)
- [ ] Author information complete (if not anonymous review)
- [ ] Abstract within word limit (typically 150-250 words)
- [ ] Keywords provided
- [ ] Acknowledgments section updated
- [ ] Copyright/license statement added (if required)

---

## Online LaTeX Editors (No Installation Required)

1. **Overleaf** (https://www.overleaf.com/)
   - Free tier available
   - Real-time collaboration
   - Built-in templates

2. **Papeeria** (https://papeeria.com/)
   - Fast compilation
   - Git integration

3. **CoCalc** (https://cocalc.com/)
   - Jupyter notebook integration
   - Real-time sync

---

## Converting to Other Formats

### LaTeX → Word (for some journals)
```bash
pandoc paper.tex -o paper.docx --bibliography=references.bib
```

### LaTeX → HTML
```bash
htlatex paper.tex
```

---

## Additional Resources

- **LaTeX Documentation**: https://www.latex-project.org/help/documentation/
- **IEEE Template**: https://www.ieee.org/conferences/publishing/templates.html
- **ACM Template**: https://www.acm.org/publications/proceedings-template
- **Overleaf Guides**: https://www.overleaf.com/learn
- **LaTeX Stack Exchange**: https://tex.stackexchange.com/

---

## Troubleshooting Commands

```bash
# Check LaTeX version
pdflatex --version

# Find missing packages
kpsewhich packagename.sty

# View compilation log
cat paper.log

# Clean and recompile from scratch
rm -f paper.aux paper.bbl paper.blg paper.log paper.out
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

---

## Version Control Tips

Add to `.gitignore`:
```
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls
```

Keep in repository:
```
paper.tex
references.bib
figures/
Makefile
README_LATEX.md
```
