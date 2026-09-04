# C2C update pilot: novelty and utility review

4 September 2026 UTC. **PI decision: park the current same-input update/repair
study as a paper candidate before paying for training.** This is a portfolio
decision from prior work and existing baseline evidence, not a negative result
from an update experiment. No update experiment has run. Preserve all code,
frozen protocols, packages, data and earlier scientific decisions unchanged.

## Closest primary work changes the next action

| Primary source | Established result / relevant scope | Consequence for this proposal |
| --- | --- | --- |
| [DroidSpeak, NSDI 2026 final paper](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf), introduction and sections 3–4 | Studies cache reuse across fine-tuned variants with the same architecture; explicitly motivates models updated over time. Selective layer recomputation addresses quality loss from direct reuse. | Neither update-induced cache incompatibility nor selective compatibility repair is a new problem statement. |
| [PrefillShare, February 2026 v1](https://arxiv.org/html/2602.12029v1), section 3.2, figure 2 and tables 1–2 | Freezes a shared prefill model and trains specialized decoders conditioned on its cache. Reports collapse from naive high-ratio cache sharing and evaluates compatible tuning on math, coding and tool calling. | Even training to preserve cross-model cache compatibility is occupied; generic compatibility regularization is not a sufficient new method. |
| [Dual-Cache / XKV, August 2026 v1](https://arxiv.org/html/2608.20617v1), problem setup and experimental sections | Two frozen models see complementary private contexts. A trained receiver-aware translator supports different cache geometries and is compared with text communication and an extended LCF-X. | Heterogeneous communication, receiver-aware fusion and private-context collaboration are existing research targets, not new directions merely by changing our benchmark. |

Important distinctions: DroidSpeak/PrefillShare concern prefix reuse rather
than our fixed C2C fuser consuming an independently updated sender. XKV's
reported base models stay frozen during translator training; the inspected
experiments do not establish independent sender-update robustness. Thus these
sources are **substantial overlap, not proof that our exact experiment already
exists**. Conversely, a narrower untested combination is not enough to make a
strong contribution. No absence-of-prior-work claim is warranted.

Prior checks also identified [cross-model KV transfer](https://arxiv.org/abs/2608.03893)
and [CacheBridge](https://arxiv.org/abs/2609.00891). The implemented ridge,
orthogonal, diagonal and projector-retuning repairs are comparison baselines,
not a new algorithm. The study currently lacks a compelling result or method
beyond those ingredients.

## New offline analysis of the completed baseline

Executable: `scripts/audit_c2c_sender_utility.py`. It first runs the committed
immutable baseline verifier, then compares paired answers using both the
conservative explicit-prefix diagnostic and the pinned published parser. It
checks the exact published parser source SHA before extracting that function.
No model output was changed, no generation was rerun, and the original strict
format gate remains `BASELINE_INCONCLUSIVE_NO_UPDATE_TRAINING`.

Evidence: `retrieved/c2c_baseline_dev_20260904_v1/c2c_baseline_dev_20260904_v1`.
New report: `artifacts/c2c_sender_utility_audit_v1.json`.

| Paired outcome, 128 questions | Conservative parser | Published parser |
| --- | ---: | ---: |
| Both sender and C2C correct | 61 | 68 |
| Sender correct, C2C wrong/unparsed | 52 | 45 |
| C2C correct, sender wrong | 4 | 4 |
| Neither correct | 11 | 11 |
| Sender correct in total | 113 | 113 |
| C2C correct in total | 65 | 72 |
| Answer-key oracle choosing the correct answer from either arm | 117 | 117 |

The four C2C-only successes comprise two in each dataset. On these recorded
answers, even perfect answer-key selection improves the sender by only
4/128 = **3.125 percentage points**, from 88.28125% to 91.40625%. That is a
sample-specific upper bound for selecting between these two recorded answers,
not a realizable routing algorithm or a bound on all future communication.
It must not be reported as an achieved ensemble accuracy.

Recorded generation time sums are **35.324 seconds for sender alone** and
**60.797 seconds for C2C**. Sender alone has both higher aggregate accuracy and
lower recorded generation time on each dataset. These are sequential
single-run generation timings, not randomized serving benchmarks; loading,
training, memory and concurrent throughput are outside this comparison.

Both models received the same question and options. Consequently, this slice
does not establish that communication is needed to combine private information.
It also cannot rule out latent communication's usefulness on long contexts,
private-evidence tasks, or amortized multi-recipient workloads. Those would be
different experiments with their own competing baselines, not interpretations
we may retroactively attach to this run.

Five focused CPU tests passed: contingency/oracle arithmetic, duplicate and
missing-pair rejection, and invalid timing rejection. The real-evidence audit
completed successfully. Engineering tests are not scientific replication.

## Execution consequence

The current five-stage C2C plan is **not an executing queue**. Its top-level
status now records a novelty/utility hold. The prepared v1 bundle at commit
`bea4ee51c0d9083668cb44736b370ca82286db50` remains an archival implementation
package, not permission to launch; do not overwrite it to hide this decision.
No remote process, dependency, checkpoint, or credential was modified.

Do not run the full two-seed update/repair chain simply because implementation
is finished. To reopen it as a paper candidate, require:

1. A deployment setting where the communication path has measured value against
   sender alone and cost-matched text/recomputation alternatives, or a clearly
   justified system constraint making sender-only infeasible.
2. A specific contribution beyond the overlaps above, with the strongest
   applicable baseline identified and its interface differences accounted for.
3. A predeclared test that could discriminate that contribution, rather than
   merely demonstrate an expected stale-adapter failure.

This review does not kill all latent-interface research and does not affect the
separate Hindsight or dialogue-retraction results. In particular, their actual
learning hypotheses must not be called disproven by prompt-only assays. The
next portfolio task is to evaluate those missing learning-level tests before
building another large apparatus around a weakly motivated failure.
