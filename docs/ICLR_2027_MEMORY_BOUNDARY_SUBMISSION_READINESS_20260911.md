# ICLR 2027 submission-readiness assessment: Addressing Is Not a Security Boundary

## Recommendation

Do not submit the current claim until the frozen G3 optimizer-routing test is
complete and independently replayed. The project supports a coherent ICLR
paper, but G3 decides its strongest defensible thesis. If aggressive table
optimization creates a deletable table copy while a sufficient backbone copy
persists, the security-boundary claim strengthens. If it moves the behavior
entirely into deletable rows, the headline must narrow to optimizer policy
determining storage location. After that decision, submission also requires
author metadata, artifact release, and a human line-by-line evidence review.

## Mock review

| Dimension | Assessment | Evidence and likely objection |
|---|---|---|
| Soundness | Strong | Preregistered estimands, independent training seeds, raw predictions, source/output manifests, positive deletion controls, causal checkpoint swaps, and full replay. The preserved failures strengthen credibility. |
| Novelty | Moderate to strong | The new question is whether unrestricted optimization uses an available addressable store. User as Engram owns surgical row-local editing; the manuscript explicitly cedes that result. A reviewer may still view the work as a narrow diagnostic around one new architecture. |
| Significance | Moderate | The distinction between read addressing and write control matters for deletion and tenant-isolation claims. Limited-memory LM work explicitly invokes deletion-based unlearning, and User as Engram establishes surgical per-user locality, but neither claims that unrestricted fine-tuning preserves locality. The paper tests a necessary missing condition rather than a broken published promise. |
| Empirical breadth | Moderate | Four sizes, two base-model families, two Pythia trigger/payload pairs, positive controls in both families, and five-seed decisive estimates. All probes are synthetic, one-token continuations and the graft is inserted at one layer. |
| Clarity/reproducibility | Strong | The paper separates valid positives, valid negatives, invalid attempts, and developmental checks; all main figures are regenerated from ten hashed source aggregates. |

A calibrated overall recommendation is **6/10, weak accept**, with meaningful
reviewer variance. This is ICLR-caliber work in question and method; it is not a
high-confidence acceptance case because no honest audit can convert a single
custom Memory Grafting implementation and synthetic probes into deployment
generality.

## Claims that survive

1. Under the tested recipe, ordinary fine-tuning installs the rare-trigger
   mapping in the backbone despite an available deterministic memory table.
2. Target-row and whole-graft restoration do not remove the ordinarily learned
   mapping in the two Pythia sizes.
3. Symmetric hybrid checkpoints make the poisoned backbone sufficient and the
   poisoned graft insufficient.
4. The identical target-row deletion removes deliberately row-confined writes
   in Pythia and Qwen, validating the intervention.
5. Frozen-graft backbone routing reproduces across four model sizes and two
   families at the fixed reliable dose, with the N=16 Qwen-0.5B negative and
   one N=64 near-trigger specificity failure preserved.
6. MLP updates and layers 0--5 are separately necessary in both Pythia sizes.
   Their early-MLP intersection passes necessity at 410M and misses the fixed
   lower-bound threshold at 1.4B; it is insufficient alone at both sizes.

## Claims prohibited

- Addressable memory is generally insecure, never used, or incapable of
  storing new behavior.
- The probe represents a realistic harmful action or estimates deployment
  prevalence.
- Early MLP parameters are a cross-scale necessary locus.
- All trigger learning is perfectly specific or all BF16 runs are bitwise
  reproducible.
- Direct row editing is novel relative to User as Engram.

## Thesis-deciding work and final upload tasks

1. Complete and fully replay G3; revise every optimizer-independent sentence
   according to the frozen outcomes.
2. If G3 preserves a sufficient outside-table copy, run the prospectively
   justified multi-depth extension only if its additional clean adaptation and
   replay fit the remaining budget. Do not call it a cheap insertion-only test.
3. Replace anonymous working-draft metadata as required by the submission
   phase and remove the local draft-header patch only when uploading.
4. Publish a sanitized artifact bundle containing code, configs, receipts,
   aggregate evidence, and verification reports; confirm that every manuscript
   path resolves in the release.
5. Have the human authors verify every numeric claim against
   `generated/EVIDENCE.json`, approve the AI-use statement, and check author
   and citation metadata.
6. Enter the abstract by 18 September 2026 AOE and upload the final paper by 25
   September 2026 AOE, leaving time for an OpenReview PDF and anonymity check.

A semantic multi-token payload would require a new scoring and parser-validity
apparatus and remains lower priority for this submission.
