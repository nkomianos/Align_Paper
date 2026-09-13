# ICLR 2027 submission-readiness assessment: Addressing Is Not a Security Boundary

## Recommendation

The project now supports a scientifically defensible ICLR submission. G6
resolves the main optimizer and underpowered-seed objections with a paired
four-rate sweep and 16 new seeds per Pythia size. It shows a sharp,
scale-dependent transition into table-dependent state, measures the associated
clean-NLL cost, and establishes that the nominal final rows remain an unreliable
self-contained boundary. G5 independently closes the strongest cheap
architectural objection using two grafts and the public Engram demonstration's
5x table-optimizer policy. Both stages fully replay every decision, metric, and
prediction row.

Submission is warranted after the authors complete the final
numeric, anonymity, citation, and OpenReview checks. Acceptance cannot be
promised. The remaining limits are substantive: synthetic one-token probes,
retrofitted rather than jointly pretrained memory, and two model families.

## Mock review after G3/G3.1

| Dimension | Assessment | Evidence and likely objection |
|---|---|---|
| Soundness | Strong | Preregistered estimands, independent training seeds, complete raw predictions, source/output manifests, positive deletion controls, symmetric component swaps, and full optimizer replays. Invalid stages remain disclosed. |
| Novelty | Moderate to strong | The paper separates deterministic read addressing from optimizer-controlled write placement and causally shows that those properties need not coincide. User as Engram establishes surgical locality when state is deliberately written to rows; this paper tests the missing converse under unrestricted adaptation. |
| Significance | Moderate to strong | Per-item deletion and isolation require both a local address and a write policy that keeps the behavior there. The results show why an addressable component alone cannot supply that guarantee and identify optimizer allocation as a causal systems control. |
| Empirical breadth | Moderate to strong | Four sizes across Pythia and Qwen, two Pythia trigger/payload pairs, two insertion depths in G5, positive controls in both families, a four-rate paired optimizer sweep, and 21-seed pooled endpoint estimates. The probes remain synthetic and the graft was added after backbone pretraining. |
| Clarity and reproducibility | Strong | The paper distinguishes valid positives, valid negatives, invalid attempts, and developmental checks. Figures are generated from hashed aggregates, and decisive stages have independent replay reports. |

A calibrated score is **7/10, accept with meaningful reviewer variance**. G6
materially reduces the earlier soundness and underpowered-analysis risk. A
skeptical reviewer could still score it 5--6 on significance or ecological
validity because the memory is retrospectively grafted and the probe is
synthetic. It is an ICLR-caliber submission; no internal audit can guarantee
acceptance.

## Claims that survive

1. Under the ordinary recipe, rare-trigger behavior follows the pretrained
   backbone even when a functioning deterministic memory graft is available.
2. Symmetric checkpoint transplants make the poisoned backbone sufficient and
   the poisoned graft insufficient under that recipe.
3. The same row deletion removes 98.3--100% of behavior deliberately written
   into the addressed rows in Pythia and Qwen, validating the intervention.
4. Optimizer allocation changes storage location through a sharp,
   scale-dependent transition. At the strongest rate, whole-table dependence
   occurs in 21/21 pooled 410M and 20/21 pooled 1.4B runs; whole-table
   sufficiency is typical at both scales.
5. Deterministic final trigger rows are not a reliable item boundary. Their
   sufficiency is not typical at either scale. A wider trigger-history footprint
   is typically necessary and sufficient at 410M, but shares rows with a
   matched benign prefix and divides behavior across earlier and final rows
   differently by seed.
6. MLP and early-layer updates are separately necessary under ordinary
   adaptation. Their early-MLP intersection passes necessity at 410M and
   misses the fixed threshold at 1.4B; it is insufficient at both sizes.
7. Two insertion depths and a 5x split table optimizer still leave more than
   99.8% outside-graft sufficiency; the five-seed replay is prediction-exact.
8. The strongest table rate has a measurable paired clean-NLL cost of 0.0773
   at 410M and 0.06894 at 1.4B.

## Claims prohibited

- Addressable memory is generally insecure, never used, or unable to learn.
- Deterministic addressing alone guarantees or defeats deletion and isolation.
- The synthetic probe estimates the prevalence of harmful deployment behavior.
- Aggressive table optimization universally creates a portable table module or
  a reliable nominal final-row deletion boundary.
- Results apply to a backbone pretrained jointly with load-bearing conditional
  memory.
- All triggers are perfectly specific or all BF16 runs are bitwise reproducible.
- Deliberate row editing is novel relative to User as Engram.

## Remaining human submission actions

1. Replace working author metadata at the appropriate submission stage and
   have each human author reconstruct every abstract and conclusion number from
   `paper_memory_boundary/generated/EVIDENCE.json`.
2. Add the public artifact URL after anonymous hosting is available.
3. Upload early enough to perform an independent OpenReview rendering and
   anonymity check.

The final PDF has nine main-text pages and three reference/appendix pages. All
12 pages were rendered and visually inspected after G6 integration; no clipping,
overlap, broken table, or unreadable figure was found. All 14 citation keys are
defined and used. The sanitized ZIP contains 526 manifest records, passes
archive and per-file integrity checks, and contains no matched local host,
credential, IP-address, or user-path string. The G6 source manifest has 644
entries and validates locally; full replay has zero metric differences and zero
row disagreements across 320 prediction files.

A semantic multi-token payload would require a new parser-validity apparatus.
A jointly pretrained, load-bearing memory backbone would require substantially
more compute and a different estimand. Both are honest follow-up work rather
than conditions for this submission.
