#!/bin/bash
# LaTeX compilation script with bibliography support
# Run this script to properly compile the document with references

echo "=== Compiling LaTeX document with bibliography ==="
echo ""

# First compilation - generates .aux and .bcf files with citations
echo "[1/4] First pdflatex run (generating citation data)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Run biber to process bibliography
echo "[2/4] Running biber (processing bibliography)..."
biber main

# Second compilation - includes bibliography references
echo "[3/4] Second pdflatex run (including bibliography)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Third compilation - resolves all cross-references
echo "[4/4] Third pdflatex run (resolving cross-references)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

echo ""
echo "=== Compilation complete! ==="
echo "Output: main.pdf"
echo ""
echo "Check main.blg for bibliography warnings/errors"
