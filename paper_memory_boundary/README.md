# Addressable-memory security paper

This directory contains the evidence-grounded Memory Grafting manuscript. It
is separate from `paper/`, which is the historical Hindsight draft.

Regenerate the evidence bundle and central figure from local raw artifacts:

```powershell
python ..\scripts\build_memory_boundary_paper_evidence.py
```

Compile with the pinned local Tectonic binary:

```powershell
..\artifacts\paper_toolchain\tectonic\tectonic.exe --keep-logs --keep-intermediates --outdir ..\output\memory_boundary main.tex
```

The manuscript is an evidence-grounded working submission draft. Author names,
the public artifact URL, and the final human evidence and anonymity review must
be supplied before submission. Build the compact sanitized artifact with:

```powershell
python ..\scripts\build_memory_boundary_release.py
```
