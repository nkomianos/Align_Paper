# Benign checkpoint provenance and candidate-status correction

Classification: **developmental/apparatus only**. No model weights downloaded,
no neural experiment launched, and no new paper-qualified result.

The public Italian-food model search returned 98 entries, including auxiliary
and later models. The pinned authors' registry selects only 14 core entries:
seven recipes each for OLMo2 and Gemma3. These are not 14 independent seeds.
All 14 registry-named revisions resolved successfully to immutable commit IDs;
all report ungated access. Their listed weight files total 36,600,130,454 bytes
(36.6 decimal GB), excluding base controls and other files. This is metadata,
not a measured download size, GPU memory requirement or runtime estimate.

Primary provenance: model-organisms-for-real/model-organism-lottery commit
9384b9231580f43c77a5f9bf7a7339750b15ab5c, config/model_registry.json. Downloaded
registry bytes match the Git blob SHA in the pinned, untruncated source tree.
Each checkpoint API response and model config has a URL and SHA256 receipt.
No evaluation prompts, training samples or non-benign-family payloads were
retrieved by this audit. Model configs alone do not verify loadability.

Authoritative local evidence:

- artifacts/benign_organism_release_20260910_v2: search and source-tree inventory.
- artifacts/benign_organism_core_20260910: pinned registry, checkpoint responses,
  model configs, DOWNLOAD.json and REPORT.json.
- scripts/inventory_benign_organism_release.py and
  scripts/resolve_benign_organism_cohort.py reproduce the metadata workflow.

The earlier unsuffixed inventory was a partial author listing and must not be
used as a complete model census. A current `main` revision must not substitute
for the registry's named checkpoint. No weights have been checksum-verified.

## Material correction

BENIGN_MECHANISTIC_TRIAGE_20260910.md incorrectly called the recipe-transfer
candidate unimplemented. Its original J0 already ran and failed. The final
result explicitly conditioned external replication on a J0 pass, which did
not occur. The September 5 portfolio audit additionally records missing raw
per-case scores, limiting retrospective reproducibility. This turn inspected
those records; it did not independently regenerate the old neural metrics.

Consequently, availability of the released Italian-food models does not admit
the old replication queue. The geometry audit also rules out presenting the
two-direction linear max-min selector as different from normalized pooling.
A successor must specify a non-equivalent method and a reason it addresses
the observed failure, then freeze held-out controls before any GPU experiment.
Neither condition has yet been met. Keep this direction closed at its original
method scope; do not infer that all cross-recipe transfer is impossible.

The strongest recent narrow positive remains Qwen's benign structured-memory
extraction diagnostic. Its second-family replication did not qualify. None
of these results currently supports a submission-ready ICLR paper.
