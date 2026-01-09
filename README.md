Purpose: Document the concept-scarcity paper, its LaTeX source, and how to build the PDF.
Persists: None.
Security Risks: None.

# Language as a Bottleneck for Embodied Intelligence: An Embodied Concept Vocabulary for Squeezing and Beyond

**Repository:** concept-scarcity-paper

## TL;DR / Abstract
Text-based reasoning systems can describe physical situations but often struggle with embodied tasks because ordinary language lacks stable, fine-grained concepts for manipulation micro-phenomena. A compact concept scaffold for hand squeezing provides explicit handles for tactile cues, material response, and control laws, reducing ambiguity and improving mechanistic reasoning in a bounded domain. The paper proposes a controlled evaluation that holds physical information constant while varying representational form, plus a short verification protocol (Box 1) to reproduce the effect.

## Download PDF
No PDF is checked into this repository. Build the PDF locally using the steps below (or via the GitHub Actions workflow described in `build-paper.ps1`). If you want a downloadable artifact, consider attaching `paper/main.pdf` as a GitHub Release asset after a successful build.

## Keywords
concept scarcity; concept scaffolds; LLM reasoning; squeezing; embodied intelligence; tactile feedback; manipulation; representation engineering; attention dilution; mechanistic reasoning

## Build
The canonical build steps are captured in `build-paper.ps1` (GitHub Actions workflow definition). To build locally, use a LaTeX toolchain such as TeX Live with `latexmk` or `pdflatex`.

**Windows PowerShell**
```powershell
powershell -Command "latexmk -pdf -cd paper/main.tex"
```

**PowerShell (pwsh)**
```powershell
pwsh -Command "latexmk -pdf -cd paper/main.tex"
```

## Repository layout
- `paper/main.tex`: Main LaTeX entrypoint (title, abstract, sections, appendices).
- `paper/`: All paper source files (currently a single `main.tex`).
- `build-paper.ps1`: GitHub Actions workflow definition for building `paper/main.pdf`.
- `LICENSE`: License for this repository.
- `README.md`: This file.

## Citation
```bibtex
@article{yap2025language,
  title  = {Language as a Bottleneck for Embodied Intelligence: An Embodied Concept Vocabulary for Squeezing and Beyond},
  author = {Joe Yap},
  year   = {2025},
  url    = {https://github.com/jenka-askly/concept-scarcity-paper}
}
```
