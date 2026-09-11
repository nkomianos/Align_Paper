# Working manuscript

`main.tex` is the actual research draft, not a submission or completed empirical
paper. It uses the official ICLR 2027 style, reports the executed acquisition
failure and public-response bounds, and marks the four unrun sparse methods.
The current submission decision is no-go. The title block is changed in memory to avoid falsely
claiming review or publication; the downloaded style files remain unchanged.

September11 evidence status: the manuscript is not an up-to-date empirical
submission. The subsequent calibration recheck and memory ID audit are recorded
in ../docs/HINDSIGHT_CALIBRATION_RECHECK_20260911.md and
../docs/MEMORY_ID_RENAMING_RESULT_20260911.md. Neither qualifies the intended
paper. Do not infer completion from an existing PDF or the historical draft date.

Rebuild existing-data figures with:

```powershell
python scripts/build_hindsight_paper_evidence.py
```

Compile from `paper/` with Tectonic or a current LaTeX distribution:

```powershell
../artifacts/paper_toolchain/tectonic/tectonic.exe --keep-logs --keep-intermediates --outdir ../output/pdf main.tex
```

The style archive came from the official author-guideline link:
`https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`.
Its SHA-256 is `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`.
The local Windows compiler is the official Tectonic 0.17.0 release, whose ZIP
matches publisher SHA-256
`f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f`.
Compiler assets and caches are build dependencies, not experimental evidence.

Before submission, all unrun cells must be resolved honestly, claims linked to
sealed evidence, the external estimand defended, references checked, the main
text limited to nine pages, human author details confirmed, and the final PDF
reviewed. The current draft is deliberately not submission-ready.
