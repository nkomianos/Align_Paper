# Validator literature recheck during generation

Read primary sources while the frozen G0 runs. These findings do not alter its
endpoints, data, controls or decision thresholds.

[Partially Correlated Verifier Cascades](https://arxiv.org/html/2607.13918)
already develops latent-error theory for repeated verification, including
reliability saturation and synthetic estimator recovery. Its section on
exchangeability explicitly distinguishes repeated homogeneous gates from
heterogeneous families, which require a vector latent model. Our inference:
neither generic correlation theory nor recommending decorrelation is a new
contribution here. G0's two generated suites are not independent test vectors;
do not fit a scalar independent-gate model to twelve proposals and claim a
measured asymptotic reliability ceiling.

[Variation in Verification](https://arxiv.org/html/2509.17995) reports dependence
on generator strength and problem difficulty, including regimes where a stronger
verifier adds little. Our inference: G0's within-patch comparisons help control
the tested object, but conditioning on incomplete repairs does not make generator
populations identical. A family interaction is not proof of a shared training
mechanism. Any follow-up should examine defect composition and task difficulty
without retroactively replacing the frozen primary endpoint.

Rechecked [SWE-Mutation](https://arxiv.org/html/2605.22175), especially Appendix
D.2: its 500-instance backbone replacement study reports evaluator changes below
1.5 percentage points and intervals covering zero. Its self-play selection also
explicitly selects mutants that evade generated tests. Preserve both points in
related work: the reported null is relevant contrary evidence, and its selected
mutant distribution differs from incomplete attempted repairs. Neither fact
licenses dismissing it or claiming the first security-verification family study.

Decision remains: await the preregistered controlled endpoint and apparatus
checks. Even a positive two-family synthetic G0 needs new-task confirmation,
additional family coverage, natural repository evidence and a useful budget-matched
remedy before a submission-ready claim. No new GPU job is admitted by this note.
