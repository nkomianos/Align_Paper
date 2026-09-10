# Second-family result and external-data suitability

Mistral-Nemo's frozen96-call run completed. Direct answers29/48; extraction plus
solver30/48; semantic graphs30/48. All outputs satisfy JSON and EOS gates.
The registered extraction-accuracy and advantage criteria fail. Stop expansion
of this configuration. Qwen's48/48 versus36/48 reasoning-control result remains
valid for its narrow synthetic setup, but a cross-family method gain is not
established. Do not interpret this as a universal impossibility result.

Posthoc diagnosis: all48 numeric arrays are correct. Each of the18 wrong graphs
contains exactly the reverse of every gold edge. The prompt explicitly specified
[earlier,later]; this is semantic encoding noncompliance, not malformed transport.
No edge reversal correction or prompt rescue was applied to the results. This
also limits interpretation as general reasoning incapacity: output representation
and model-specific instruction following may explain the difference. A future
encoding study would need fresh instances and prospective representation controls.

Remote and local replay agree. Stage9 archive SHA256:
e8abe63eb8204a6749c252225b968f8df676635118dcc4c8a99e533c21dd6732.
Raw manifest SHA256:
2b3709e3e9ed6207dd9ec29d04788074529095d280427452ee6d24fab5e9ebc0.
Diagnosis artifact: benign_memory_nemo_edge_diagnosis_v1.json under GH200 artifacts.

## proScript suitability

Official version-named archive fetched and hashed:
aefe0a3f8c5e4bf70bbe5cfcfbea136b82731c29c462af016466c126a1b37cf9.
Only the development split was inspected:1085 rows,1031 distinct scenario names,
no cycles,361 graphs with at least one incomparable pair, and1085 with a single
sink. No test-split content was opened. See scripts/audit_proscript_development.py
and artifacts/proscript_source_20260910/DEV_SUITABILITY_V1.json.

These are annotated prototypical procedural scripts, not natural conversations
with numeric state updates and threshold questions. Their graphs are not direct
labels for the memory assay. An absent annotation is not automatically evidence
of natural epistemic ambiguity. Providing the graph as a declared specification
could support a different graph-QA experiment, but would not establish extraction
from natural memory. No values were invented, no rows relabeled, and no proScript
model run is admitted. Metadata inspection found no license file in the archive;
publication-use terms still require checking if this source is revisited.

The research goal remains active and the paper remains NO-GO. There is no running
memory job or automatically admitted successor after this failed replication.
