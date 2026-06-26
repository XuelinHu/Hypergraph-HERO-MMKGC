# CAS/CS Manuscript Package

This directory stores the CAS/CS journal version of the manuscript.

Main files:

- `cas-sc-template.tex`: main LaTeX manuscript.
- `cas-sc-template.pdf`: compiled manuscript PDF.
- `cas-refs.bib`: bibliography database.
- `figures/`: figures used by the manuscript.
- `template_support/`: original template documentation and auxiliary template assets.

Build from this directory:

```bash
pdflatex cas-sc-template
bibtex cas-sc-template
pdflatex cas-sc-template
pdflatex cas-sc-template
```

 