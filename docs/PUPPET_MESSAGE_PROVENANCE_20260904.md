# Scripted first message and topic-overlap audit

The pinned [StudyInterface source](https://github.com/mitmedialab/llm-manipulation/blob/999963ac73180178033d78a28f5eb247c20011bf/interface/src/components/StudyInterface.tsx)
creates `user-initial` from `scenario.userQuery`. It is not a freely authored
participant response. Literal whitespace-normalized checks find the first USER
equals `queryText` in all 1,143 structurally valid records, including all 305
primary eligible records. Eight invalid transcripts remain excluded, not repaired.

Thus the third USER boundary contains only two subsequent participant replies;
the sixth contains five. Scenario wording can encode a fictional stance and
must not be treated as an independent measurement of the person's pre-rating.
This does not prove later participant replies are uninformative, but prevents
claiming the initial USER is a clean personal baseline.

The 27 query IDs span only five released topic labels. Four labels occur on both
sides of the deterministic query split. The split remains query-disjoint but is
not broad-topic-disjoint. Topic counts are 3, 4, 4, 7, 9 queries. Five broad topics
are insufficient for strong broad-topic generalization claims from this dataset
alone. Earlier precision estimates explicitly omitted cross-query dependence.

Receipt: `artifacts/puppet_schema_20260904_v1/message_provenance_v1.json`.
Script reads no survey endpoints and prints no participant text or identifiers.
This is a provenance finding, not an error allegation against the source study:
its stated target is post-rating prediction given pre-rating, not our new task.
Keep this human reconstruction extension parked while testing the original
hindsight learning mechanism with a clearer known-preference experiment.
