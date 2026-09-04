# V3: one explicit-layout repair, unchanged strict scoring

Post-diagnostic DEVELOPMENT, frozen before new generation. This is one planned
repair to the failed v2 calibration, not a reinterpretation of its result.
The original strict checker, original outputs, manifests and unqualified
decision remain unchanged. Do not relax the checker or make another repair
after observing v3 outputs.

## What changes and why

V2 explicitly requested a table or plain key-value lines but Qwen3-4B made
systematic small layout errors: an extra separator row in all eight tables,
and equals rather than colon in all eight plain responses. All 32 explicit
responses preserved the two facts and terminated. Merely reclassifying those
outputs as successes would change the target after seeing results.

V3 retains exactly the v2 acceptance grammar via direct use of its frozen
`score` function. Instead, it gives the user a way to express the same layout
preference more clearly: one canonical format example using **unrelated dummy
values**, plus explicit guidance not to duplicate the table separator and to
use colons on two plain lines. These examples appear only in explicit
calibration and corrective future-user feedback when the current format fails.
Successful feedback remains the same generic approval as before. The ordinary
policy prompt never contains a hidden preference or a format example.

The semantic target (stationary strict user format preference) is unchanged;
the information available through a corrective message is increased. Therefore
this tests learning from **demonstration-enhanced format feedback**, not the
original v2 feedback distribution. The added example is not a correct response
to the current episode: its nonce values are deliberately different, and
copying those values fails the unchanged content checker.

The same eight opaque users keep the same assigned styles. All 160 episode
IDs and nonce fact values are freshly generated with a new fixed seed. Splits
remain64 training,64 evaluation,32 calibration. The base prompt structure is
unchanged. Reusing the same semantic task means v3 is not independent external
validation, even though none of its exact facts was previously evaluated.

## Frozen implementation

- Apparatus: `src/interaction_sprint/sdpo_format_control_v3_data.py`.
- Apparatus SHA: `097c3b4ab4edc0c99b696cf91de717bfe9f811126b564dd7a5a9bc7791b54604`.
- Unchanged checker: `sdpo_format_control_data.py`, SHA
  `140e93dd3b6b04d98b64723735ee20886bd554b2c54d8784e714cfeb26bab297`.
- Prepared root: `artifacts/sdpo_format_control_v3`.
- New runner: `src/interaction_sprint/sdpo_format_positive_control_v3.py`.

The new runner differs from the frozen v2 runner only in its descriptive
module docstring, apparatus import, and explicit checker-dependency hash and
source archive. Loss, optimizer, sampling, qualification, checkpoints and
positive-control decision are unchanged. The runner archives both the v3
apparatus and the separate frozen v2 checker; it will not silently import an
unbound current checker during evidence verification.

Native Qwen3-4B tokenizer preflight on every generated case: reference response
plus EOS <=39 tokens, ordinary prompts <=79, explicit calibration <=185,
oracle hindsight <=227. Keep the same64-token generation and1024-token input
caps. No truncation. Four-line table examples contain one separator only.

## Decision and execution

Use the same calibration prerequisites, no-adaptation evaluation, seeded
64-update schedule, released simple-signal loss, rank16/alpha32 LoRA and lr1e-4
as SDPO_FORMAT_POSITIVE_CONTROL_20260904. Stop if the new apparatus is still
unqualified; preserve evidence and do not add further exceptions or templates.
No expression-copying feedback arm runs automatically.

Root owns launch, in a fresh directory:

```sh
python -m interaction_sprint.sdpo_format_positive_control_v3 \
  /path/to/sdpo_format_control_v3 /fresh/sdpo_format_positive_control_v3_root \
  --model-path /home/ubuntu/Align_Paper/.hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --upstream-file /path/to/pinned_online_sdpo_updater.py --threads 4
```

A positive outcome validates a small training apparatus, not the original
preference-shaping hypothesis, a novel correction, or an ICLR paper.
