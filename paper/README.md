# PAMPAr-o1 v9 arXiv Paper

## Purpose

This directory contains the LaTeX source for the arXiv preprint documenting PAMPAr-o1 v9's architecture and preliminary experiments.

## Files

- `pampar_v9_arxiv.tex` - Main LaTeX document
- `figures/` - Diagrams and figures (optional, TikZ diagrams are inline)

## Compilation

```bash
pdflatex pampar_v9_arxiv.tex
pdflatex pampar_v9_arxiv.tex  # Run twice for references
```

Or use an online editor like [Overleaf](https://www.overleaf.com/).

## arXiv Submission

### Categories
- **Primary**: cs.CL (Computation and Language)
- **Secondary**: cs.LG (Machine Learning), cs.NE (Neural and Evolutionary Computing)

### Submission Steps

1. Go to https://arxiv.org/submit
2. Create account if needed
3. Select categories: cs.CL (primary), cs.LG, cs.NE (cross-list)
4. Upload `pampar_v9_arxiv.tex`
5. Fill metadata (title, abstract, authors)
6. Submit for moderation

### License for arXiv
The paper uses **CC-BY-4.0** license (standard for arXiv papers).
Note: This is separate from the AGPL-3.0 code license.

## Citation

Once published, update `CITATION.cff` with the arXiv ID:

```yaml
identifiers:
  - type: arxiv
    value: "XXXX.XXXXX"
```

## Contact

Lucas Ricardo Mella Chillemi
lucas.mella@outlook.com
