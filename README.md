Purpose: Document the concept-scarcity paper, its LaTeX source, and how to build the PDF.
Persists: None.
Security Risks: None.

# Language as a Bottleneck for Embodied Intelligence: An Embodied Concept Vocabulary for Squeezing and Beyond

**Repository:** concept-scarcity-paper

## Overview
This repository contains the LaTeX source for the paper “Language as a Bottleneck for Embodied Intelligence: An Embodied Concept Vocabulary for Squeezing and Beyond,” which argues that concept scarcity in natural language is a bottleneck for embodied reasoning and introduces a compact concept scaffold for hand squeezing as a case study.

## Search Keywords
concept scarcity, concept scaffold, embodied intelligence, embodied vocabulary, hand squeezing, tactile feedback, manipulation, robotics, representation engineering, attention dilution, mechanistic reasoning

## Paper Metadata
- **Title:** Language as a Bottleneck for Embodied Intelligence: An Embodied Concept Vocabulary for Squeezing and Beyond
- **Author:** Joe Yap (Independent Research)
- **Date:** December 15, 2025
- **Abstract:** See `paper/main.tex` (Abstract section).

## Repository Layout
- `paper/main.tex`: Full LaTeX source for the paper, including abstract, vocabulary tables, and appendices.
- `build-paper.ps1`: GitHub Actions workflow (YAML content) that builds `paper/main.tex` and uploads `paper/main.pdf` as an artifact.
- `README.md`: This file.

## Build the PDF
The repository includes a GitHub Actions workflow definition in `build-paper.ps1` that uses `xu-cheng/latex-action@v3` to compile the paper and upload `paper/main.pdf`. To build locally, compile `paper/main.tex` with your preferred LaTeX toolchain.

## Key Concepts (From the Paper)
- **Concept scarcity:** Natural language lacks stable, fine-grained terms for embodied micro-phenomena.
- **Concept scaffold:** A compact, structured vocabulary that reduces ambiguity and anchors mechanistic reasoning.
- **Squeezing vocabulary:** Terms organized around motor control, tactile feedback, material response, and failure modes.
- **Evaluation protocol:** A lightweight, reader-run verification protocol (“Ten-Minute Replication Challenge”) to test representational shifts.

## Evaluation & Reproduction
See Box 1 in `paper/main.tex` for a quick verification protocol that compares baseline reasoning vs. scaffolded reasoning using the same physical information.

## Citation
If you reference this work, cite the paper title and author as listed in `paper/main.tex`.

## License
See `LICENSE`.
