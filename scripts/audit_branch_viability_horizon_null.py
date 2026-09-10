"""Exact developmental null: absolute branch viability is not teacher advantage.

Three interchangeable child symbols have identical teacher/student probability.
After any child, h independent steps each survive with probability 19/20.
An error is absorbing. Conditioning on an earlier correct spine does not change
the independently resampled continuation. No neural model or training is run.
"""
from fractions import Fraction as F
from math import comb
import json
from pathlib import Path


def binomial_probability(n, k, p):
    return comb(n, k) * p**k * (1 - p)**(n-k)


def exact_rows():
    rows = []
    for horizon in (1, 5, 10, 20, 40):
        success = F(19, 20)**horizon
        teacher = student = (F(1, 3),) * 3
        branch_values = (success,) * 3
        teacher_value = sum(q*v for q, v in zip(teacher, branch_values))
        student_value = sum(p*v for p, v in zip(student, branch_values))
        assert teacher_value == student_value == success
        # Six independent rollouts per child. Low <.40 means <=2 successes;
        # high >=.75 means >=5 successes. Every-low implies mean-low too.
        low = sum(binomial_probability(6, k, success) for k in range(3))
        high = sum(binomial_probability(6, k, success) for k in (5, 6))
        uncertain = low**3
        diversity = 3*high**2*(1-high) + high**3
        gray = 1-uncertain-diversity
        assert 0 <= gray <= 1
        # An independent route to these event probabilities: enumerate counts.
        events = {'uncertain': F(0), 'diversity': F(0), 'gray': F(0)}
        for a in range(7):
            for b in range(7):
                for c in range(7):
                    counts = (a, b, c)
                    probability = F(1)
                    for k in counts:
                        probability *= binomial_probability(6, k, success)
                    label = ('uncertain' if max(counts) <= 2 else
                             'diversity' if sum(k >= 5 for k in counts) >= 2
                             else 'gray')
                    events[label] += probability
        assert events == dict(uncertain=uncertain, diversity=diversity, gray=gray)
        rows.append(dict(remaining_steps=horizon,
                         branch_success_exact=str(success),
                         branch_success=float(success),
                         resampled_student_success=float(student_value),
                         teacher_advantage_exact=str(teacher_value-student_value),
                         teacher_student_kl=0,
                         probability_real_uncertain=float(uncertain),
                         probability_diversity=float(diversity),
                         probability_gray=float(gray)))
    assert rows[0]['probability_diversity'] > .99
    assert rows[-1]['probability_real_uncertain'] > .89
    return rows


if __name__ == '__main__':
    result = dict(classification='developmental exact counterexample',
                  claim='absolute viability does not identify incremental teacher benefit',
                  neural_experiment=False, rows=exact_rows())
    out = Path('artifacts/position_reliability_audit_20260910/HORIZON_NULL.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
