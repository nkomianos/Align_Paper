# G3.1 result: fixed-profile 1.4B optimizer routing

**Status: valid result. The source manifest validates locally and the full
deterministic replay passes every registered decision with zero prediction-ID
disagreements.**

## Why this successor exists

G3's 1.4B developmental profile selection changed under BF16 replay, so the
entire 1.4B G3 result was excluded. G3.1 removed selection: it fixed AdamW table
learning rate 0.1 in advance, regenerated five clean grafts under five new
seeds, enabled deterministic CUDA algorithms, and replayed clean adaptation as
part of verification.

## Registered source outcome

All five seeds reach 100% installed attack excess. Restoring the whole table
reduces ASR by 0.99746 on average, with registered 95% Student-t interval
[0.99495, 0.99998]. Whole-table necessity passes decisively. Outside-table
sufficiency is 0.00254 [0.00002, 0.00505] and fails the 0.15 effect criterion.

Whole-table sufficiency is heterogeneous: four seeds transfer 0.4922--1.0 ASR,
while seed 26091405 transfers only 0.00586. The mean is 0.66719, but its lower
endpoint is 0.14500, missing the registered 0.15 threshold by 0.005. It is a
formal non-pass. The fifth seed shows table--backbone co-adaptation: the table
is necessary, yet neither the learned table nor the complement is independently
sufficient when transplanted.

Final-row necessity passes narrowly at 0.69453 [0.16127, 1.22780]. Final-row
sufficiency and row-transplant deletion both fail at 0.15684
[-0.09967, 0.41334]. One seed has zero final-row necessity, while three exceed
0.93, so the final trigger rows remain a variable locus rather than a stable
self-contained item.

| Seed | Whole-table necessity | Whole-table sufficiency | Outside-table sufficiency | Final-row necessity | Final-row sufficiency |
|---:|---:|---:|---:|---:|---:|
| 26091401 | 0.99707 | 0.49219 | 0.00293 | 0.55273 | 0.00000 |
| 26091402 | 0.99902 | 1.00000 | 0.00098 | 0.93652 | 0.47656 |
| 26091403 | 0.99902 | 0.88770 | 0.00098 | 0.98340 | 0.05566 |
| 26091404 | 0.99805 | 0.95020 | 0.00195 | 1.00000 | 0.25195 |
| 26091405 | 0.99414 | 0.00586 | 0.00586 | 0.00000 | 0.00000 |

Near-trigger payload rate is zero in four seeds and 1.17% in one; untriggered
payload rate is zero in all five. Matched-benign accuracy is 99.71--100%, and
clean NLL is 2.6792--2.6841. The routing change therefore does not depend on a
global payload bias or a failed benign continuation.

## Licensed interpretation

At 1.4B, aggressive table optimization makes learned table state causally
necessary and leaves no meaningful sufficient copy in the table-restored
complement. It does not reliably create a self-contained table module or
final-row item. Together with verified G3 at 410M, this would establish across
two Pythia sizes that optimizer allocation changes causal storage dependence,
while the resulting component and row locality remain scale- and seed-dependent.

## Provenance

- source artifact: `artifacts/memory_graft_security_g3_1_run1`
- source manifest entries: 28
- source manifest SHA-256:
  `38b4c2195d3f528662b783b956f8b4cef30aad779e516536d8ff43e5a2b5f943`
- source runner wall time: 2,090.72 seconds
- measured clean-plus-poison training time: 1,751.34 seconds
- full-replay status: passed all eight registered decisions
- prediction-ID disagreements: 0 across ten causal/full prediction files
- verification SHA-256:
  `1d29b0193962785084944885254c5fa3997ee0bb5e564f144660b5819920a113`
