# Addressable-memory security paper

This directory is a fresh manuscript based only on the verified Memory Grafting
experiments. It is separate from `paper/`, which is the historical Hindsight
draft.

Regenerate the evidence bundle and central figure from local raw artifacts:

```powershell
python ..\scripts\build_memory_boundary_paper_evidence.py
```

Compile with the pinned local Tectonic binary:

```powershell
..\artifacts\paper_toolchain\tectonic\tectonic.exe --keep-logs --keep-intermediates --outdir ..\output\memory_boundary main.tex
```

The manuscript is an evidence-grounded working submission draft. Author names,
artifact-release location, broader specificity analysis, and final independent
paper review remain before submission.
