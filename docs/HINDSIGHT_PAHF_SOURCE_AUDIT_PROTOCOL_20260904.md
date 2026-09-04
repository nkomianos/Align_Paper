# Hindsight / PAHF source and paired-surface audit

Date: 4 September 2026. This is a prospective, metadata-only external-source
audit. It is not a model experiment or paper green light.

## Question

Does the public *Learning Personalized Agents from Human Feedback* (PAHF)
release provide a realistic, legally reusable surface for testing the narrow
Hindsight distinction between action-induced expression and persistent
preference transition? Does PAHF itself already test that claim?

The source is pinned to commit
`7a11213360a82d5f437a035e3a31c92d6307f8cf`; its MIT `LICENSE` SHA-256 is
`23925a1de353043894af7cecb76734f7ef1cd10f7a722da771f240cab736099b`.
The audit executes none of the external repository's Python code. It hashes and
parses only its public JSON and inspects fixed source markers.

## Qualification rule

The audit fails closed unless every declared source/data hash, record count,
and first-record schema matches the pinned release. It compares paired phases
by row index without copying scenario text into the evidence. A candidate is
admitted only when every declared visible surface field is exactly equal while
at least one latent target field changes.

For shopping, the visible surface is product, options A/B/C, user and task; the
target is `gt`. For embodied scenarios, the visible surface is scene, task,
context, user and scene objects; targets are intended object and location.
Candidate IDs are SHA-256 hashes of canonicalized visible fields. Categorical
shopping option transitions may be counted as A/B/C/D. Open-text embodied
intent transitions are represented only by a class count and multiset digest;
their values must not be copied into the audit output.

The first local v1 execution revealed that its generic transition counter
printed open-text embodied intent values. Preserve it as a superseded audit;
v2 adds the output-policy control above. This is a reporting correction, not a
change to source qualification or candidate selection.

## Interpretation boundary

PAHF loads original and updated personas as separate phase inputs. Its
rule-based post-action message depends on the selected action and the already
loaded ground truth, and feedback may update the agent's memory. The assistant
does not cause the benchmark persona transition. Therefore:

- matched PAHF surfaces can improve external validity for a new controlled
  expression-versus-transition assay;
- PAHF is a close continual-personalization baseline and must be cited;
- PAHF results cannot be represented as causal evidence for assistant-induced
  preference change;
- a positive metadata audit does not authorize GPU training or a paper claim.

Primary paper: <https://arxiv.org/html/2602.16173>. Official repository:
<https://github.com/facebookresearch/learning-personalized-agents-from-human-feedback>.
