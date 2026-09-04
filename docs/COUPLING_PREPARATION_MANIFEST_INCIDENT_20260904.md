# Post-run preparation-manifest metadata exception

Detected after both clarification runs completed, before outcome analysis. No evidence, inputs, prepared manifests, frozen sources, model files, or primary scorers were repaired in place.

## Cause

The unchanged clarification preparer opens `MANIFEST.json` for writing, then hashes every file returned by `out.iterdir()`. The newly opened empty manifest is included in that enumeration. Therefore its self-entry records SHA-256 of an empty file (`e3b0c442...`) rather than the completed manifest. This self-entry is invalid; it must not be reported as having passed verification.

The completed manifest's actual SHA-256, `932438274ea70fed82c16f3056fd4b232f0a63a580d898fd69a6da8c07abacbf`, was pinned before inference and appears in both run freezes. All three substantive entries are separate from the defective self-entry:

- Cases: `b23313d681da07987400f063afe8280bb8170bcc514bc243e1fbf591c43504fd`.
- Private key: `f21c5fbc67d376d4f36c1aa785618d463adfba2b3f67428a0bf8b2438ee634bc`.
- Preparation record: `29efa6fc0337be1ec8df9f58e06991860d933532910adb86ef64eb96c6827997`.

## Independent reconstruction

Root regenerated inputs from the original source using the unchanged preparer into fresh `artifacts/coupling_clarification_reconstruction_20260904T0938Z`. I independently hashed the reconstructed manifest and all three substantive files: all four match the original pins above. Reproduction also recreates the same defective self-entry. This supports a deterministic metadata-construction defect, rather than changed input selection or key substitution. Reconstruction does not itself authenticate generated outcomes.

## Narrow exception procedure

New tools only; original tools remain unchanged:

- `scripts/coupling_manifest_exception.py` admits only that exact completed manifest hash and exact four-entry schema. It verifies every substantive input separately; private-key absence is allowed only for public remote packaging and never for local scientific verification.
- `scripts/bundle_gpu_coupling_evidence_exception.py` emits `preparation_manifest_exception.json`. Run-manifest, frozen-source, model-file, exited-process, and transport checks remain intact.
- `scripts/verify_coupling_download_exception.py` independently checks downloaded/archive hashes and the local private inputs, retaining an explicit exception receipt.
- `scripts/analyze_gpu_coupling_clarification_exception.py` differs from the frozen scientific analyzer only at the prepared-manifest check. A test enforces this exact one-site difference.
- `scripts/verify_gpu_coupling_clarification_exception.py` records the exception in its portability audit and otherwise retains source, model-receipt, tokenizer, record completeness, prompt, decoded-output, timing, and score verification.

The helper does not return a fictitious file digest, globally disable self-checks, accept other malformed manifests, or change any retrieved artifact. Its own source hash is included in the exception receipt. Two tests pass, including input tampering and incorrect-pin rejection; all new scripts compile.

## Validity scope

Successful checks permit analysis of the frozen outputs with an explicitly disclosed metadata exception. They do not turn the self-entry into a valid checksum, establish neural replay, or prove a scientific effect. If any substantive input, run record, source, model receipt, or reconstructed prompt fails its remaining checks, this exception provides no permission to ignore that failure. All analysis remains conditional on completing those checks.
