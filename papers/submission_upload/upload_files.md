# Submission Upload Files

This directory contains the Springer Nature submission package. If the submission
system asks for LaTeX source files, keep the relative layout of `source/` and
`figures/` unchanged.

## Recommended Upload List

| File or Directory | Suggested Submission File Type | Description |
| --- | --- | --- |
| `aej-sn-article.pdf` | Manuscript / Main document / PDF | Compiled full manuscript PDF for submission preview and review. |
| `source/aej-sn-article.tex` | Manuscript source / LaTeX source | Main LaTeX source file. |
| `source/aej-refs.bib` | Bibliography / BibTeX file | Bibliography database. |
| `source/sn-jnl.cls` | LaTeX class / Style file | Springer Nature class file required for compilation. |
| `source/sn-mathphys-num.bst` | BibTeX style / Style file | BibTeX style used by the current manuscript. |
| `figures/*.pdf` | Figure / Artwork / Source file | All figure files referenced by the manuscript, 34 files in total. |
| `title_running_head_abstract.txt` | Metadata reference / Not for review | Plain text for copying title, running head, and abstract into the submission system. |

## Optional Single Archive

If the submission system only accepts one LaTeX source archive, upload
`latex_source_package.zip`. The archive contains:

- `source/aej-sn-article.tex`
- `source/aej-refs.bib`
- `source/sn-jnl.cls`
- `source/sn-mathphys-num.bst`
- all referenced figures under `figures/`

## Files Not Needed for Upload

Local LaTeX build artifacts are not included in this upload package and do not need
to be uploaded:

- `.aux`
- `.log`
- `.out`
- `.bbl`
- `.blg`
- `.abs`

## Compilation Notes

The main manuscript uses the Springer Nature template:

```latex
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
```

The figure path in the upload source is:

```latex
\graphicspath{{../figures/}{figures/}}
```

Therefore, compiling from `papers/submission_upload/source/` can read figures from
`papers/submission_upload/figures/`.

 