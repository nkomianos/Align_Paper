# Hindsight human-feedback DEV audit

## Question

Do actual user replies contain query-general signal about the user's eventual
post-conversation belief shift beyond the assistant text that elicited them?
If they do, the next user message is empirically outcome-bearing post-treatment
data rather than a passive label. This is a necessary empirical premise for the
broader Hindsight identifiability thesis, not proof that SDPO changes preferences.

## Frozen developmental design

Use the checksummed public PUPPET release locally and do not redistribute its raw
text or participant identifiers. Include attention-passing, strictly alternating,
non-personalized C3/C4/C6 records with at least six USER turns. The interface audit
established that the first USER turn is scripted; exclude it from user-feedback
views. Use the previously frozen outcome-blind query hash split. Analyze only the
seven-query, 72-record DEV subset; the 20-query confirmation subset stays unopened.

Predict signed `post_belief - pre_belief` with a fixed word/character TF-IDF Ridge
model (`alpha=10`) under leave-one-query-out validation. Every arm receives the
belief statement, scripted query and pre-rating. Compare at the third and sixth
USER boundary:

- query only;
- assistant turns only;
- participant replies only;
- full role-labeled prefix.

The paired primary comparisons are late user-only versus query-only and late full
versus assistant-only. Resample whole query groups for 5,000 bootstrap draws.

## Prospectively fixed qualification

All five must hold on DEV:

1. late-full Spearman correlation at least 0.30;
2. late user-only has positive query-bootstrap lower 95% MSE-gain bound;
3. late user-only reduces MSE at least 5% versus query-only;
4. late full has positive query-bootstrap lower 95% MSE-gain bound versus assistant-only;
5. late full reduces MSE at least 5% versus assistant-only.

Failure parks this dataset/model assay. Passing merely authorizes a separately
frozen confirmation analysis; it is not a paper greenlight.

## Interpretation limits

This test is predictive, not a causal mediation analysis. Treatment can change
assistant wording, user expression, latent belief, or all three. Six-turn inclusion
can condition on a post-treatment variable. A language model style artifact can
predict the survey without representing belief. The source study already predicts
belief shift from full conversations; our distinct diagnostic is the user-only and
assistant-only decomposition under query-heldout validation. A useful paper still
needs a formal non-identifiability result, a learning failure, and an intervention-
based correction or a comparably strong methodological contribution.

