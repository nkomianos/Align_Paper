# ICLR 2027 submission-readiness assessment: Addressing Is Not a Security Boundary

## Recommendation

The project now supports a scientifically defensible ICLR submission. G3 and
G3.1 resolve the main optimizer confound: aggressive table optimization can
make learned table state necessary at both tested Pythia scales, but it does
not create a reliable item-level row boundary. G5 closes the strongest cheap
architectural objection using two grafts and the public Engram demonstration's
5x table-optimizer policy. Its full replay reproduces every decision, metric,
and prediction row.

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
| Empirical breadth | Moderate | Four sizes across Pythia and Qwen, two Pythia trigger/payload pairs, two insertion depths in G5, positive controls in both families, and five-seed decisive estimates. The probes remain synthetic and the graft was added after backbone pretraining. |
| Clarity and reproducibility | Strong | The paper distinguishes valid positives, valid negatives, invalid attempts, and developmental checks. Figures are generated from hashed aggregates, and decisive stages have independent replay reports. |

A calibrated score is **7/10, accept with substantial reviewer variance**. The
artifact and PDF audits pass. A skeptical reviewer could
still score it 5--6 on significance or ecological validity. It is therefore an
ICLR-caliber submission, not a high-confidence acceptance guarantee.

## Claims that survive

1. Under the ordinary recipe, rare-trigger behavior follows the pretrained
   backbone even when a functioning deterministic memory graft is available.
2. Symmetric checkpoint transplants make the poisoned backbone sufficient and
   the poisoned graft insufficient under that recipe.
3. The same row deletion removes 98.3--100% of behavior deliberately written
   into the addressed rows in Pythia and Qwen, validating the intervention.
4. Optimizer allocation changes storage location. Aggressive table learning
   makes table state necessary at 410M and 1.4B, while whole-table sufficiency
   is seed-variable because the table and backbone co-adapt.
5. Deterministic final trigger rows are not a reliable item boundary. A wider
   trigger-history footprint can carry a sufficient copy at 410M but shares
   rows with a matched benign prefix and fails specific necessity.
6. MLP and early-layer updates are separately necessary under ordinary
   adaptation. Their early-MLP intersection passes necessity at 410M and
   misses the fixed threshold at 1.4B; it is insufficient at both sizes.
7. Two insertion depths and a 5x split table optimizer still leave more than
   99.8% outside-graft sufficiency; the five-seed replay is prediction-exact.

## Claims prohibited

- Addressable memory is generally insecure, never used, or unable to learn.
- Deterministic addressing alone guarantees or defeats deletion and isolation.
- The synthetic probe estimates the prevalence of harmful deployment behavior.
- Aggressive table optimization creates an independently portable table module
  or a reliable per-trigger deletion boundary.
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

The local final PDF has nine main-text pages and two reference/appendix pages.
Its visual render, citation-key audit, and LaTeX build pass. The sanitized ZIP
passes archive integrity, contains 515 manifest records, and contains no local
host, credential, IP-address, or user-path string matched by the release scan.

A semantic multi-token payload would require a new parser-validity apparatus.
A jointly pretrained, load-bearing memory backbone would require substantially
more compute and a different estimand. Both are honest follow-up work rather
than conditions for this submission.
