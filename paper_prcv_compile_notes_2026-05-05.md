# PRCV Compile Notes

Files prepared:

- `paper_prcv_submission_2026-05-05.tex`
- `paper_prcv_template_aligned_2026-05-05.tex`
- `custom.bib`
- `llncs.cls`
- `splncs04.bst`

Current assumptions:

- The manuscript is now written in Springer LNCS / PRCV style.
- The bibliography is moved to `custom.bib` and the manuscript uses `\bibliographystyle{splncs04}` and `\bibliography{custom}`.

What you still need from the official PRCV package:

- `llncs.cls`
- `splncs04.bst`

How to use:

1. Put `paper_prcv_submission_2026-05-05.tex` into the PRCV template directory.
2. For direct use with the downloaded official template, prefer `paper_prcv_template_aligned_2026-05-05.tex`.
3. Keep `custom.bib`, `llncs.cls`, and `splncs04.bst` in the same directory.
4. Rename the TeX file if needed to match your project structure.
5. Compile with the normal Springer/LNCS sequence:

```bash
cd /Users/yiming/project/TimeCMA
pdflatex paper_prcv_template_aligned_2026-05-05.tex
bibtex paper_prcv_template_aligned_2026-05-05
pdflatex paper_prcv_template_aligned_2026-05-05.tex
pdflatex paper_prcv_template_aligned_2026-05-05.tex
```

Recommended next cleanup before submission:

- Replace anonymous author and institute placeholders.
- Check whether PRCV requires anonymous review or named submission.
- Recheck every citation against the final `.bib` style output.
- Add your final figures if you plan to include architecture or rolling-window diagrams.
