# Posterior-ratio reweighting: correct expectation, wrong welfare guarantee

Executed three CPU checks in `tests/test_hindsight_feedback_reweighting.py`.
With an exact Bayesian teacher q(a|o), frozen weights q(a|o)/pi(a) turn the
product-distribution expectation over actions and observations into the joint
expectation. Applied to the full reverse-KL gradient, this recovers the previous
own-action log-ratio direction. One hundred random positive binary channels agree
numerically; a stopped-weight finite-difference loss check independently agrees.

This is ordinary importance reweighting, not a claimed new algorithm. Crucially,
it is NOT a preference correction: a symmetric channel restores a balanced policy
even when the initial-preference objective favors one action. The explicit test
uses preference probability .9 and policy .8; the weighted update reduces that
utility. Thus do not propose this ratio alone as Anchor-SDPO or call it safer.

New close primary prior: Nguyen et al., August 2026,
https://arxiv.org/html/2608.09263v1 . The paper separately analyzes score meaning,
feedback self-dependence, and loss behavior, including forward-KL projection and
reverse-KL centroids. It reports that removing own-rollout feedback dependence
does not ensure useful credit. Our generic likelihood-versus-value argument is
therefore not novel; a paper would need a distinct result about preference
transitions and an identified target, not another instance of that distinction.

PI decision: retain the executable identities as apparatus checks. Do not spend
GPU time testing this unqualified reweighting as a claimed welfare correction.
The causal-preference candidate still lacks its central constructive contribution.
