# build-paper.ps1
# Builds paper/main.pdf using latexmk (recommended).
# Prereq (Windows): install MiKTeX or TeX Live, and ensure latexmk is available.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location "$PSScriptRoot\paper"
try {
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
}
finally {
  Pop-Location
}
