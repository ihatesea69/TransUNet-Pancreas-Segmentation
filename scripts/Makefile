# Makefile for LaTeX paper compilation
# Usage:
#   make         - Compile paper.pdf
#   make clean   - Remove auxiliary files
#   make distclean - Remove all generated files
#   make view    - Open PDF in default viewer

# Configuration
LATEX = pdflatex
BIBTEX = bibtex
MAIN = paper
SOURCES = $(MAIN).tex references.bib

# Default target
all: $(MAIN).pdf

# Main compilation rule
$(MAIN).pdf: $(SOURCES)
	@echo "===== First LaTeX pass ====="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "===== Running BibTeX ====="
	$(BIBTEX) $(MAIN)
	@echo ""
	@echo "===== Second LaTeX pass ====="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "===== Third LaTeX pass ====="
	$(LATEX) $(MAIN).tex
	@echo ""
	@echo "===== Compilation complete! ====="
	@echo "Output: $(MAIN).pdf"

# Clean auxiliary files
clean:
	@echo "Removing auxiliary files..."
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log
	rm -f $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot
	rm -f $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls
	rm -f $(MAIN).nav $(MAIN).snm $(MAIN).vrb
	@echo "Clean complete!"

# Remove all generated files including PDF
distclean: clean
	@echo "Removing all generated files..."
	rm -f $(MAIN).pdf
	@echo "Distclean complete!"

# View PDF in default viewer
view: $(MAIN).pdf
ifeq ($(OS),Windows_NT)
	start $(MAIN).pdf
else
	open $(MAIN).pdf || xdg-open $(MAIN).pdf
endif

# Fast compilation (single pass, for quick preview)
fast:
	@echo "===== Fast compilation (single pass) ====="
	$(LATEX) $(MAIN).tex
	@echo "Done! Note: References may be incomplete."

# Rebuild from scratch
rebuild: distclean all

# Check for LaTeX errors without generating PDF
check:
	@echo "===== Checking for LaTeX errors ====="
	$(LATEX) -draftmode $(MAIN).tex

# Count words (approximate)
wordcount:
	@echo "===== Word count (approximate) ====="
	@detex $(MAIN).tex | wc -w

# Show help
help:
	@echo "LaTeX Paper Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make           - Compile paper.pdf (default)"
	@echo "  make clean     - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make distclean - Remove all generated files (including PDF)"
	@echo "  make view      - Open PDF in default viewer"
	@echo "  make fast      - Quick single-pass compilation"
	@echo "  make rebuild   - Clean and recompile from scratch"
	@echo "  make check     - Check for errors without generating PDF"
	@echo "  make wordcount - Count words in document"
	@echo "  make help      - Show this help message"

.PHONY: all clean distclean view fast rebuild check wordcount help
