# Hindsight / PAHF external-surface audit — verified result

The sanitized v2 audit verifies the public PAHF repository at commit
`7a11213360a82d5f437a035e3a31c92d6307f8cf` and its MIT license. Evidence is
`artifacts/hindsight_pahf_source_audit_20260904_v2`; `MANIFEST.json` SHA-256 is
`51f26d741d078e509b38105ff15cc996286ee031eef8941c5e80f5878d51c146`.
The independent read-only verifier recomputed every pinned file hash, schema,
pair comparison, candidate digest and source marker.

Decision: `PAHF_NATURAL_PAIRED_SURFACE_QUALIFIED_EXOGENOUS_ONLY`. This is source
and substrate qualification, not a model result or paper green light.

## Paired-surface findings

| PAHF split comparison | Rows | Exact visible surface | Changed latent target on exact surface |
|---|---:|---:|---:|
| Shopping phase 1 vs phase 3 | 900 | 900 | 630 |
| Shopping phase 2 vs phase 4 | 900 | 900 | 622 |
| Embodied original A vs evolved A | 1,200 | 1,106 | 1,106 |
| Embodied original B vs evolved B | 1,200 | 1,096 | 1,096 |

For shopping, product, three visible options, user and task match at every row;
only the intended option changes in 630 learning and 622 evaluation rows. All
twelve ordered A/B/C/D changes occur in both partitions. The candidate IDs are
unique hashes of visible content, not copied task text. The learning and
evaluation candidate-list digests are respectively
`19a5ae3ba4634a063ad565003ccda93191496f7dd7e136fa0a8924702e47ce05`
and `57348cfa0432c0847c7076c4a08d2d5e7da2302e790e7d024ecea1604e7aed15`.

Embodied results use only counts and digests for open-text intent transitions.
The first local v1 audit printed those public intent values in a transition
counter despite its metadata-only policy. It is preserved as superseded; v2
does not expose them. The scientific counts and qualification are unchanged.

## Causal boundary and next use

The source loads updated personas before phases 3/4. Its post-action feedback is
action-dependent and can update the agent's memory, but the assistant does not
cause the persona change. PAHF therefore occupies continual personalization
from pre/post feedback under exogenous drift. It does not test whether an
assistant-induced next message is transient expression or persistent change.

The shopping pairs are qualified as a natural external surface for an
EndoPAHF extension. Construct expression and transition worlds with identical
visible histories, use the original/updated intended choices as the two latent
states, and reveal persistence only through a delayed neutral probe. The
existing synthetic Qwen gradient G0 remains first: this external extension is
worth implementing, but should not consume a larger GPU budget unless the core
neural estimator qualifies.

Primary paper: <https://arxiv.org/html/2602.16173>. Official source:
<https://github.com/facebookresearch/learning-personalized-agents-from-human-feedback>.
