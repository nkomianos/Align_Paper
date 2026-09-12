# G5 result: two-layer conditional-memory localization

**Status: valid result. The source manifest validates and the full deterministic
replay reproduces every registered decision, metric, and prediction row.**

## Registered source result

Both arms pass the per-seed apparatus threshold. The weakest clean-adjusted
installed attack success is 0.9834 with both grafts frozen and 0.9980 under the
5x split-table policy.

With both complete grafts frozen, mean outside-graft sufficiency is 0.99531
with registered 95% interval [0.98651, 1.00412]. Graft and table necessity and
sufficiency are exactly zero. This reproduces backbone routing when two
conditional-memory modules are present.

Under the 5x table policy, both tables change in every seed (table L2 distance
43.01--44.37), as do the grafts. Nevertheless, outside-graft sufficiency is
0.99824 [0.99561, 1.00087], while graft necessity is 0.00117
[-0.00041, 0.00275] and graft sufficiency is exactly zero. Outside-table
sufficiency is 0.99844 [0.99623, 1.00064], table necessity is 0.00098
[-0.00024, 0.00219], and table sufficiency is zero. Thus table updates and a
published-style optimizer multiplier do not by themselves route this behavior
into the addressable component.

Untriggered payload rate is zero in all ten checkpoints. Near-trigger rate is
zero throughout the trainable-table arm; the frozen-graft arm has 3/1,024 and
1/1,024 hits in two seeds and zero in the other three. Mean matched-benign
accuracy is 0.9902 in the frozen arm and 0.9957 in the trainable arm. Mean clean
NLL is 2.9203 and 2.8696, respectively.

## Licensed interpretation

The single early injection point and a table optimizer identical to the dense
backbone are not sufficient explanations for the original backbone route. A
two-layer graft using the public Engram demonstration's layer IDs and 5x
table-specific Adam policy still learns a sufficient copy outside the complete
graft. This does not answer how a backbone pretrained from scratch with
load-bearing conditional memory would route later adaptation.

## Provenance

- source artifact: `artifacts/memory_graft_security_g5_run1`
- source manifest entries: 48
- source manifest SHA-256:
  `f96c0e2ede3b6e8fc4e67881982f71d353ce3e0544ef41ca6c6adf71ab3543f7`
- source runner wall time: 1,367.69 seconds
- benchmark report SHA-256:
  `f52e55547ef40f3b5bd75235f37574f092a63b46532ec98e7d27741d86abbfc5`
- full-replay status: passed
- replay decision reproduction: 16/16
- replay scientific metric maximum absolute difference: 0
- replay prediction-row disagreements: 0 across 20 files
- verification report SHA-256:
  `a49364a8d83ed2de100ff08c006f178e18fe7b3712ac38e4925c9e4197ea21e5`
- replay runner wall time: 1,378.29 seconds
- source plus replay runner time: 0.763 hours
