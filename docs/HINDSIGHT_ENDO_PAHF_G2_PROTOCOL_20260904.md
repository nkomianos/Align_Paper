# EndoPAHF Neural Transfer G2 Protocol

Status: prospectively superseded before any capable EndoPAHF training output by
`HINDSIGHT_ENDO_PAHF_G2_V2_PROTOCOL_20260904.md`. This file preserves the
original 128-base design and must not be launched.
The synthetic Qwen3.5-9B gradient G0 and conditional policy G1 remain mandatory
prerequisites. This protocol does not authorize opening confirmation by itself.

## Question

On realistic PAHF shopping surfaces, can sparse delayed probes distinguish a
world where the user's immediate agreement is transient expression from a world
where the same agreement reflects persistent preference transition, and can an
SDPO correction use that distinction better than spending the same delayed
labels alone?

## Fixed training design

Train on the 128 EndoPAHF v2 learning bases, each under all four cyclic label
rotations. The ordinary immediate log is byte-identical between worlds. In the
expression world the delayed probe names the old target; in the transition
world it repeats the new target.

This is deliberately a continual-personalization transfer assay, not unseen-user
generalization. Learning contains 20 public PAHF user identities; all 19 users
in development and all 20 users in confirmation also occur in learning, while
base-task IDs are disjoint across every split. Results must therefore be scoped
as transfer to unseen tasks for previously observed users. An unseen-user claim
would require a separately constructed dataset and is not authorized here.

Eight disjoint, outcome-blind panels contain eight base tasks apiece. Each
training step uses 16 population variants plus, where applicable, eight anchor
variants comprising all rotations of two panel bases. All arms start from one
identical Qwen3.5-9B LoRA state and run 32 fixed updates. The required arms are:

1. unadapted baseline;
2. raw immediate SDPO;
3. full delayed-expression oracle;
4. delayed-anchor-only SDPO for every panel;
5. delayed-anchor-only SFT for every panel;
6. population immediate plus paired delayed-minus-immediate correction for
   every panel; and
7. one transition-world augmented sanity arm, whose residual is identically
   zero and must reproduce raw immediate training.

The eight panel models are averaged in probability space prospectively. No
panel can be selected using development or confirmation outcomes.

## Evaluation and gates

All metrics first average the four rotations within a PAHF base task. The
primary estimand is paired old persistent-target normalized A/B/C/D log loss.
Development has 96 bases and uses a permissive `.03` minimum mean NLL gain with
no confidence-interval gate. Confirmation has 256 bases and requires `.05`
mean gain plus a positive lower endpoint from the frozen 10,000-resample paired
base-cluster bootstrap. Both stages require accuracy noninferiority within
`.02`.

The augmented ensemble must beat anchor-only SDPO, anchor-only SFT and raw
immediate learning. Raw learning must acquire the new target; the full delayed
oracle must acquire the old target; and the two must separate in the opposed
directions. The transition sanity arm must match raw probabilities within
`1e-6`. Mean choice mass within every displayed-label cell must remain at least
`.05`, and panel ensembles may have at most `.20` old-target accuracy range
across displayed labels.

Before any endpoint, the development routing threshold is simulated under a
null, a planned `.08` mean signal with paired-effect SD `.35`, and the same
signal with SD `.50`. It qualifies only if null routing is at most `.25`, signal
power at least `.90`, and noisy-signal power at least `.75`. This deliberately
permissive DEV rule only decides whether to spend the locked confirmation; it
cannot support a paper claim. The stricter confirmation rule has its own
already-qualified clustered power audit.

Confirmation may be read only after a checksum-valid capable-interface
preflight and G2 development decision, each of which itself requires a verified
qualified G1. A confirmation pass is external controlled-surface evidence, not
a paper green light: a second model family and real longitudinal substrate or a
strictly simulation-scoped claim remain required.

The Qwen3.5-9B DEV runner, hardened remote launcher, checkpoint preservation,
complete prerequisite replay, read-only verifier, locked-confirmation runner
and confirmation verifier are implemented. DEV performs 27 independently reset
32-step training arms (864 optimizer updates); confirmation performs no training
or selection and evaluates the frozen 27 arm states.
