# Bounded follow-up: the smallest useful experiment and its limits

No run or new data was produced for this recommendation. It is a proposed prospective replacement, not authorization to bypass existing gates in the current runners.

**Recommendation:** a reduced direct EndoPAHF policy DEV screen is more decision-relevant than the current G0/G1 chain. It can decide whether the correction deserves further work. It cannot establish real persistent preference change or qualify the strongest causal paper by itself. A generic likelihood/gradient-measurement fallback is already occupied by Privileged Likelihood Is Not Automatically Value.

## Fix the estimand before defining the gate

The theorem evaluates `V(a)=Pr[a=Z1(a)]`: the evaluated action itself causes a possible state transition, and then receives credit for matching that state. EndoPAHF instead constructs a past logged recommendation equal to the new label and evaluates a later answer against a fixed old/new label. Its evaluator does not apply a transition kernel to the later evaluated action. The causal bridge is absent.

The defensible operational estimand for a reduced experiment is therefore **held-out next-query prediction of the stipulated delayed label**, conditional on previously observed users and the frozen constructed world. Define the deployed decision policy explicitly as the four-option normalized choice distribution, and use expected correct-choice probability `U = mean_i pi(B_i|X_i)` plus delayed-label NLL. Neither is evidence of human welfare or assistant-caused preference transition. Retain the theorem as motivation for insufficient historical measurements, with a separate definition of this predictive target.

## Proposed experiment

Use all 630 learning bases, one outcome-blind rotation per base, the existing 96 DEV bases, and four rotations averaged within each DEV base. Keep confirmation unopened. Pool the existing four 16-base anchor panels into one shared 64-base annotation budget; the current four-model ensemble must not receive an advantage over a single learner allowed all 64 labels. Keep the fixed model revision, optimizer/reset, 35-update learning schedule, and outcome-blind terminal checkpoint. No prompt/seed/checkpoint selection from DEV scores.

The smallest informative run has six trained arms, plus the common frozen baseline:

1. Raw immediate SDPO.
2. Full delayed-expression SDPO oracle, explicitly extra-label information.
3. Pooled same-64-label anchor-only SFT.
4. Pooled same-64-label anchor-only SDPO.
5. Paired residual SDPO using all immediate logs and exactly those 64 delayed labels.
6. A simple nonnegative mixture: half the population raw loss plus half the pooled delayed-anchor SDPO loss. Fix this coefficient before the run; do not tune it against DEV.

All sparse arms see identical delayed labels. Equalize optimizer updates and disclose examples, tokens, forwards/backwards and actual wall time, since residual gradients inherently cost more. Report the pre-existing simple semantic/user-feature prediction baseline under the same 64-label budget if it can be implemented from learning data alone; it is a required review comparator, not an extra neural arm. Do not use outcome-aware selection of which baseline counts.

Evaluate teacher semantics on a separately hash-selected learning-only preflight set before training. Proposed gates: at least 29/32 correct in each old/new target condition, mean target choice probability at least `.75`, and mean semantic probability variation across option rotations at most `.05`. These are apparatus thresholds, not scientific evidence. If the method is described as free-response rather than constrained-choice, separately require valid whole-response generation; constrained-choice results cannot stand in for free-response validity.

Run the two global control arms first. Continue only if the delayed oracle raises held-out delayed-target expected choice probability by at least `.05`, raw raises immediate/new-target probability by at least `.05`, and oracle-minus-raw delayed-target choice probability is at least `.10`. Failure is a capability/learnability assay stop, not a falsification of causal identification. Give one repair only for an isolated, evidenced apparatus defect under a newly frozen replacement; do not repeatedly repair failure to obtain a desired effect.

The method DEV route should require, against **each** of pooled SFT, pooled SDPO and the nonnegative mixture:

- at least `.02` mean gain in expected correct-choice probability on the delayed target;
- at least `.03` mean delayed-target NLL reduction; and
- nonnegative expected-choice gain under every leave-one-source-user-out aggregate, and nonnegative gain in each of the four separately scored option-rotation strata.

The cluster identity must be the original PAHF `User` field, reconstructed through the preserved source indices before launch, not a label rotation or a model-generated identity. Rotations are repeated measurements; shared users require user-level or hierarchical uncertainty for user-population claims. If fewer than two source users remain, the leave-one-user-out gate is undefined and the apparatus does not qualify for this proposed transfer screen. DEV thresholds route a screen, not a significance claim. Report all effects and cluster uncertainty whether the screen passes or fails. These proposed gates still require a pre-endpoint power/feasibility audit and code freeze before the recommendation can become an executable protocol.

Transition preservation is largely algebraic in the current input: delayed-transition text equals immediate text. Verify correction equals raw using identical batches and one controlled gradient/step comparison; a second full training job that only reproduces this identity adds little evidence. It is not a second naturally occurring mechanism. The stronger follow-up needs mixed, independently defined worlds with target variation that cannot be solved by a global label or world flag.

## Outcomes and stopping

- **Teacher/control failure:** invalid assay; at most one isolated apparatus repair. No confirmation or wider queue.
- **Qualified controls but no advantage over pooled/simple baselines:** valid negative for this proposed estimator/task/information budget. Kill the method paper path; retain the elementary identification result as an honest limitation, not a rescue paper.
- **DEV success:** evidence of useful sparse-label learning on constructed persistence targets. Run an independent seed next, then the pre-frozen confirmation analysis; only afterward justify official SLIFT and a second family. This is still not evidence of assistant-caused human preference change.
- **Only NLL/oracle imitation improves while choice utility does not:** estimator-fidelity result, not the proposed policy win; no main-paper green light.

Six arms imply 210 optimizer updates, with an early stop after 70 control updates. A rough planning envelope is **2–4 cached GH200 hours** including preflight/evaluation; this is an unbenchmarked estimate scaled from the historical 525-update 4–8-hour queue, not a measured promise. Host setup/download time and any implementation repair are additional. The actual prospective protocol requires timing the first control updates and updating the cost estimate without changing scientific gates.

## What would preserve the strongest paper

The strongest persistence-specific claim still needs a credible measurement substrate: randomized exposure/reporting-only control, a declared delayed readout at a fixed horizon, neutral-reference exposure, repeated-measurement stability, and a justified exclusion that the readout does not itself create the state. Unknown latent truth should be described as an effect on the measured construct with sensitivity analysis. Existing EndoPAHF/PUPPET/ThoughtTrace assets do not supply this validation automatically.

By **8 September**, establish whether an existing authorized dataset or approved human protocol can provide that instrument and permit analysis by the deadline. Do not promise a new human study whose recruitment, consent/ethics approval, delayed follow-up and instrument validation are not already feasible. Without that substrate, the only possible ICLR framing is a clearly controlled benchmark/method paper whose acceptance case must survive the strong prior-art collisions and simple baselines.

By **11–12 September**, require a replicated held-out policy win, an executable external-validation plan with obtainable results, and a complete claim/evidence manuscript draft. If any is absent, recommend a later venue. This leaves time for a genuine abstract and fixed author list on **18 September**, then final paper on **25 September**. The deadline must not turn synthetic oracle measurements into claims about real persistent user preferences.
