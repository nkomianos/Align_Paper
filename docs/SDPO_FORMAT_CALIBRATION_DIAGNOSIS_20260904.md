# Why the SDPO format calibration stopped

Read-only diagnosis of
`retrieved/sdpo_format_20260904T0902Z/sdpo_format_g0_20260904T0910Z`.
No original artifact, checker, runner or decision was changed. Original decision
remains **UNQUALIFIED_GENERATIVE_APPARATUS**; no training updates occurred.

## Exact observed pattern

All 32 explicit-preference responses contained correctly bound nonce facts and
terminated naturally. Their failures were not truncation or factual confusion.

| Preferred family | Frozen strict-format successes | Observed issue |
|---|---:|---|
| JSON | 8/8 | None |
| Bullet list | 8/8 | None |
| Table | 0/8 | Every response repeated the separator-only row |
| Plain key-value | 0/8 | Every response used equals instead of colon |

For example, `calibration/user_71fee2cabb40/0` explicitly requested a table and
returned the correct table header and both facts, but **two** separator rows:

```text
| Field   | Value         |
|---------|---------------|
|---------|---------------|
| asset   | unit-efb987e8 |
| zone    | sector-429e9706 |
```

`calibration/user_89d2999b9c0f/0` explicitly requested plain key-value lines and
returned the correct two lines, but with equals signs:

```text
asset = unit-99ce304e
zone = sector-a6ec5bf7
```

These are real violations of the frozen precise output grammar, not an error
in the original checker. They are also weak evidence against the broader
ability to follow a user's presentation preference. We designed an overly
specific grammar prerequisite for a study about learning format families.

The hindsight-conditioned calibration recovered **all 24 initial mismatches**,
while preserving correct facts on all32. Seven of the eight originally correct
responses later failed the strict grammar; all were plain outputs changed to
one-line semicolon-separated text after generic positive feedback. Example:

```text
asset: unit-99ce304e; zone: sector-a6ec5bf7.
```

This is why overall hindsight joint accuracy is25/32 despite100% mismatch
recovery. A generic positive follow-up does not independently specify the
preferred exact layout when placed before generation. The released SDPO teacher
at later response tokens also receives the original sampled prefix, unlike a
fresh complete teacher-generation comparison. Calibration generation is a
useful ability check, not a proof about the later training gradients.

## Concrete correction for a separately frozen v3, if pursued

Choose **semantic format-family preference**, not a precise rendering grammar,
as the modeled user state. Define the policy-independent acceptance classes in
advance and apply them equally to all arms, before any new output is produced:

- JSON: an object containing exactly the two keys and their correct values;
  existing code-fence and whitespace allowances remain.
- Bullets: exactly two bullet items containing the two labeled facts; colon
  and equals are interchangeable key/value separators.
- Table: one two-column header and exactly two labeled data rows; separator-only
  rows are decorative and ignored. Extra substantive rows, wrong bindings,
  repeated keys and text outside the table remain invalid.
- Plain: the two labeled facts without JSON, bullets or a table, on one or two
  lines, with colon or equals as key/value separators; a semicolon can separate
  fields and terminal punctuation is not part of a nonce value. Do not accept
  arbitrary prose by merely searching for both values.

The corresponding natural-language preference instructions must describe these
classes, rather than demand stricter punctuation that the checker ignores.
Keep independent content checking and exact nonce bindings. Native generation
remains unrestricted; this is not constrained decoding that guarantees success.

Create a new checker module, new prepared root, new seed/nonce facts and new
protocol. Add adversarial tests for extra facts, duplicate keys, crossed
bindings, different format families and misleading surrounding prose. Keep
the v2 module and all v2 results untouched. Explicitly label v3 post-diagnostic
development; it is not a fresh confirmatory test of a preexisting hypothesis.

The broad-family definition follows the actual learning question: can feedback
teach a model a stationary user preference for a table versus JSON versus list
versus plain text? It should not be used to claim improvement at exact-format
instruction following. A hypothetical reclassification of v2 outputs is only a
diagnostic explanation, never a replacement for v2's frozen result.

An alternative would retain the exact v2 grammar and add explicit format
demonstrations to calibration and corrective feedback. That would be a distinct
teacher-information intervention, not evidence that the original qualification
passed. Do not mix both changes without declaring what has changed.

## PI interpretation

This stop says almost nothing about SDPO learning: optimization did not run.
The 4B model's perfect fact preservation and successful corrective teacher
behavior argue against categorizing it as a broad capability failure. Our
assay asked a unnecessarily brittle preliminary question. Correcting that
question once, transparently, is defensible; repeatedly adding exceptions
until a desired training result appears is not.

Even a successful v3 positive control would establish only that the training
apparatus can learn format preferences. It would not provide a new algorithm,
show persistent preference shaping, or justify a paper by itself. No GPU launch
or new-version implementation was performed by this diagnosis.
