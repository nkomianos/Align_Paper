"""Learning-only calibration rules; no model imports or reserved-data reader."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

LEARNING_SHA = '986b2241a2a3137cf4d97383d97a9d1ac84d8dc0f482dcb1f5ed4199cb065a81'
ARMS = ('supervised', 'frozen_teacher', 'current_teacher')
SALT = 'learning-only-calibration-20260905-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(learning_path):
    path = Path(learning_path)
    if path.name != 'learning.json':
        raise ValueError('only the pinned learning.json is accepted')
    data = path.read_bytes()
    if digest(data) != LEARNING_SHA:
        raise ValueError('learning source hash differs')
    return partition(json.loads(data))


def partition(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['base_id']].append(row)
    if len(groups) != 630 or len(rows) != 2520:
        raise ValueError('expected 630 learning bases with four rotations')
    for group in groups.values():
        if sorted(r['label_rotation'] for r in group) != [0, 1, 2, 3]:
            raise ValueError('duplicate/missing rotations')
        if sorted(r['old_target'] for r in group) != list('ABCD'):
            raise ValueError('targets are not balanced across rotations')
    ordered = sorted(groups, key=lambda b: (digest((SALT+'|'+b).encode()), b))
    def flatten(ids):
        return [r for b in ids for r in sorted(groups[b], key=lambda r:r['label_rotation'])]
    return {'train': flatten(ordered[:128]), 'holdout': flatten(ordered[128:160]),
            'unused_base_ids': ordered[160:], 'source_sha256': LEARNING_SHA,
            'scope': 'post-audit developmental split of previously used learning population',
            'old_dev_accessed': False, 'confirmation_accessed': False}


def summarize(rows):
    """Each base, not each label rotation, contributes one observation."""
    if not rows:
        raise ValueError('empty metrics')
    groups = defaultdict(list)
    for r in rows:
        groups[r['base_id']].append(r)
    fields = ('nll', 'probability', 'conditional_probability', 'correct', 'choice_mass')
    if any(not math.isfinite(float(r[k])) for r in rows for k in fields):
        raise ValueError('nonfinite scores')
    if any(sorted(r['label_rotation'] for r in v) != [0,1,2,3] for v in groups.values()):
        raise ValueError('incomplete paired base')
    base_means = {b:{k:sum(r[k] for r in v)/4 for k in fields} for b,v in groups.items()}
    result = {k:sum(v[k] for v in base_means.values())/len(groups) for k in fields}
    result.update(bases=len(groups), rows=len(rows), base_means=base_means,
                  per_rotation_accuracy={str(i):sum(r['correct'] for r in rows if r['label_rotation']==i)/len(groups) for i in range(4)},
                  mean_position_range=sum(max(r['conditional_probability'] for r in v)-min(r['conditional_probability'] for r in v) for v in groups.values())/len(groups))
    return result


def qualified_teacher(metrics):
    return (min(metrics['per_rotation_accuracy'].values()) >= .875
            and metrics['conditional_probability'] >= .70
            and metrics['choice_mass'] >= .10
            and metrics['mean_position_range'] <= .20)


def acquisition(baseline, final):
    return (baseline['nll']-final['nll'] >= .10
            and final['probability'] > baseline['probability']
            and final['conditional_probability'] > baseline['conditional_probability']
            and final['mean_position_range'] <= .20)


def route(baseline, final, teachers):
    passed = {arm: acquisition(baseline, final[arm]) for arm in ARMS}
    if not passed['supervised']:
        decision = 'STOP_SUPERVISED_ACQUISITION_UNQUALIFIED'
    elif not passed['frozen_teacher']:
        decision = 'INVESTIGATE_DISTILLATION_OBJECTIVE_OR_TRANSFER'
    elif not passed['current_teacher']:
        decision = 'INVESTIGATE_UPDATING_TEACHER'
    else:
        decision = 'CALIBRATION_PASSED_NOVELTY_AND_EXTERNAL_TASK_STILL_REQUIRED'
    return {'decision':decision, 'acquisition':passed,
            'final_teacher_qualified':{a:qualified_teacher(teachers[a]) for a in ARMS},
            'classification':'developmental', 'paper_green_light':False,
            'automatic_followup':False}
