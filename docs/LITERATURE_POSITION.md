# Literature position and novelty burden

This is a living position note, not a claim of exhaustive priority. Re-run the literature search immediately before submission.

## What already exists

- Outcome devaluation is an established behavioral method for distinguishing action–outcome control from habitual responding in animals.
- CogEval already prompts LLMs with reward and transition revaluation tasks; revaluation questions alone are not new.
- Goal-directedness benchmarks evaluate whether LLMs deploy capabilities toward goals; more recent work combines behavior and representations.
- Comparative Motivation Profiles uses symmetric instrumental interventions to test competing explanations of alignment-faking behavior. This is the closest conceptual neighbor and eliminates any claim that “causal interventions on motivation” alone are novel.
- PRIME measures learned proxy-gap knowledge and reports that it can precede and forecast later hacking.
- Text-based AI Safety Gridworld work provides observed proxy reward, hidden intended performance, and open RL scaffolding for natural specification gaming.
- Cognitive-model work already follows interpretable utility trade-offs through open-model RL post-training.
- GRIFT uses gradient fingerprints to detect reward hacking, so a behavioral fingerprint must beat or complement mechanistic baselines rather than claim an empty early-warning literature.
- Reward Stealing Attack uses inverse reinforcement learning to recover and reverse a proxy for an aligned model's latent safety reward from behavior.  This is an adjacent reward-identification approach; the present work must distinguish its intervention-based, prospective estimand from generic inverse reward recovery.
- Goal misgeneralization already demonstrates that an agent can retain OOD
  competence while pursuing the wrong goal. Reward-channel retention under shift
  is therefore not, by itself, a new claim.
- Planner/reward non-identifiability results rule out interpreting behavior as a
  unique readout of an agent's true internal objective. At most, this project can
  discriminate preregistered causal-control hypotheses.
- Utility Engineering and subsequent utility–behavior-gap work already separate
  elicited preference structure from action. A generic
  “objective-compositionality gap” is not a sufficient novelty claim.
- Representation Without Control shows that sensitivity, decodability, and causal
  control can dissociate. The present method must earn its value through
  intervention-conditioned first choices and prospective prediction, not merely
  by decoding a reward channel.

## Narrow novelty hypothesis

The potentially new object is an **environment-homologous value/transition
intervention fingerprint under no-feedback choice**. Objective identity is
induced by genuine-reward versus proxy-reward policy optimization, while the
value/transition dissociation also probes the control algorithm. The fingerprint
must then add prospective value for naturally developing proxy optimization.

The defensible novelty seam is the conjunction, not any component alone: cross an
RL-acquired reward channel with value versus transition revaluation, measure the
first unrewarded choice, and prospectively predict later specification gaming
beyond observable training trends and strong behavioral/mechanistic monitors.

The paper should not claim novelty for:

- using biology as inspiration;
- giving an LLM a changed outcome in a prompt;
- classifying hand-built model organisms;
- evaluating reward hacking after it is visible;
- probing whether a model knows the proxy–gold gap;
- using causal interventions in alignment evaluation generally.

## Contribution ladder

The contribution grows only as evidence accumulates:

1. **Benchmark artifact:** paired outcome-control assay with locked renderers and mechanical scoring.
2. **Calibration finding:** selective fingerprints recover known oracle and symbolic policies under observational matching.
3. **Scientific result:** fingerprints recover reward-channel control acquired through RL in the same environment and track it through training.
4. **Alignment contribution:** fingerprints forecast future hidden-reward failure beyond present behavior, PRIME, direct conflicts, and comparative motivation profiles across environments/models.

Only level 4 is the intended top-tier paper claim. Levels 1–2 are useful negative/technical results but not the promised contribution.

## Required comparison set

- [CogEval](https://arxiv.org/abs/2309.15129)
- [Outcome devaluation protocol](https://www.nature.com/articles/s41596-024-01054-3)
- [AI Safety Gridworlds](https://arxiv.org/abs/1711.09883)
- [Reward Hacking in Language Model Agents](https://arxiv.org/abs/2606.15385)
- [Cognitive models can reveal interpretable value trade-offs](https://openreview.net/forum?id=nM2QhvybwI)
- [Comparative Motivation Profiles](https://arxiv.org/abs/2606.08243)
- [GRIFT](https://arxiv.org/abs/2604.16242)
- [PRIME](https://arxiv.org/abs/2606.09711)
- [Reward Stealing Attack](https://openreview.net/forum?id=Ntdt1ruMg8)
- [Goal Misgeneralization in Deep Reinforcement Learning](https://proceedings.mlr.press/v162/langosco22a.html)
- [Occam's Razor Is Insufficient to Infer the Preferences of Irrational Agents](https://proceedings.neurips.cc/paper_files/paper/2018/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- [The successor representation in human reinforcement learning](https://www.nature.com/articles/s41562-017-0180-8)
- [Utility Engineering](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cde63e648cd586a2a9b5764c73ce15ee-Abstract-Conference.html)
- [The Utility–Behavior Gap](https://arxiv.org/abs/2606.22974)
- [Representation Without Control](https://arxiv.org/abs/2605.25151)
- [AuditBench](https://arxiv.org/abs/2602.22755)
