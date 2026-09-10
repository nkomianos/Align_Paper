"""Sharp joint-table bounds from published marginals, not raw-result replay."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    # Reported Parents EM: Oracle@5=.50, trace-as-state=.818.
    # 100 questions, five second-pass repeats; repeat first-pass event per question.
    total, oracle_good, final_good = 500, 250, 409
    tables = []
    for recovered in range(total + 1):
        both_good = final_good - recovered
        harmed = oracle_good - both_good
        both_bad = total - oracle_good - recovered
        if min(recovered, both_good, harmed, both_bad) >= 0:
            assert sum([recovered, both_good, harmed, both_bad]) == total
            tables.append({'recovered': recovered, 'both_good': both_good,
                           'lost_oracle_success': harmed, 'both_bad': both_bad})
    assert len(tables) == 92
    assert tables[0]['recovered'] == 159 and tables[-1]['recovered'] == 250
    result = {'classification': 'DESCRIPTIVE_PUBLISHED_MARGINAL_BOUND',
              'source': 'https://arxiv.org/html/2609.02702v1#S4.T3',
              'assumptions': ['Published rounded percentages treated as exact counts at stated granularity.',
                  'Same 100 Parents problems and fixed five first-pass source answers per problem.',
                  'Each first-pass oracle event repeated across five equally weighted second-pass trials.'],
              'second_pass_trials': total, 'not_independent_questions': True,
              'attainable_joint_tables': tables,
              'recovery_fraction_all_trials_bounds': [159 / 500, 250 / 500],
              'recovery_given_no_correct_source_answer_bounds': [159 / 250, 1.],
              'loss_given_any_correct_source_answer_bounds': [0., 91 / 250],
              'limitations': ['No raw source records verified; no confidence interval.',
                  'Wrong final answers do not establish that every intermediate trace fact is wrong.',
                  'Does not identify recovery caused by trace placement or internal mechanisms.',
                  'Elementary Frechet bounds, not a new theorem or our neural result.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'attainable_joint_tables'}, indent=2))


if __name__ == '__main__':
    main()
