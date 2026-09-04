"""Read-only terminal-artifact audit, never a model/backward/optimizer replay.

Refuses to read outcomes until the final manifest exists. Optional --out must be
outside the artifact directory and is created exclusively (never overwritten).
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from interaction_sprint import hindsight_full_feedback as runner
from interaction_sprint.hindsight_acquisition_calibration import dataset, SEED
from interaction_sprint.hindsight_matched_learning import data_and_schedule


STOP = 'STOP_TRUTHFUL_HINDSIGHT_LEARNING_UNQUALIFIED'
COMPLETE = 'COMPLETED_SIX_ARM_DEVELOPMENTAL_COMPARISON'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, message, atol=1e-10):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == e.shape and np.isfinite(a).all()
            and np.allclose(a, e, atol=atol, rtol=1e-10), message)


def marginal(prob, target, rho):
    p = np.asarray(prob, dtype=float)
    require(p.shape == (2,) and np.isfinite(p).all() and (p >= 0).all()
            and p.sum() <= 1 + 1e-10, 'Invalid unconditional A/B probabilities')
    result = rho * np.array([p[0], p[1], max(0., 1 - p.sum())])
    result[target] += 1 - rho
    return result


def metrics(rows):
    return dict(n=len(rows),
        correct=sum(int(r['full_argmax'] == r['target_token']) for r in rows),
        nll=float(np.mean([r['nll'] for r in rows])),
        mean_target_probability=float(np.mean([r['target_probability'] for r in rows])),
        min_AB_mass=min(r['AB_mass'] for r in rows),
        domains={d: dict(n=sum(r['domain'] == d for r in rows),
            correct=sum(int(r['full_argmax'] == r['target_token']) for r in rows if r['domain'] == d))
            for d in sorted({r['domain'] for r in rows})})


def qualifies(metric):
    return (metric['correct'] / metric['n'] >= .9 and metric['min_AB_mass'] >= .95
            and all(d['correct'] / d['n'] >= .75 for d in metric['domains'].values()))


def validate_evaluation(rows, cases, base, vocab, answer_ids=None):
    lookup = {r['id']: r for r in cases}
    require(len(rows) == len(cases) and {r['id'] for r in rows} == set(lookup), 'Evaluation coverage')
    labels = [{r['target_token'] for r in rows if r['target'] == t} for t in (0, 1)]
    require(all(len(s) == 1 for s in labels) and labels[0] != labels[1], 'Answer-token mapping')
    ab = [next(iter(s)) for s in labels]
    require(all(type(t) is int and 0 <= t < vocab for t in ab), 'Answer-token range')
    require(answer_ids is None or ab == answer_ids, 'Answer-token mapping changed')
    for r in rows:
        c = lookup[r['id']]
        require(all(r[k] == c[k] for k in ('domain', 'split', 'target'))
                and r['anchor'] == base[c['base_id']]['anchor'], 'Evaluation metadata')
        p = np.asarray(r['AB_probabilities'], dtype=float)
        marginal(p, c['target'], .9)
        require(0 < r['target_probability'] <= 1 and 0 < r['AB_mass'] <= 1 + 1e-10,
                'Evaluation mass range')
        close(r['AB_mass'], p.sum(), 'Evaluation A/B mass')
        close(r['target_probability'], p[c['target']], 'Evaluation target probability')
        close(r['nll'], -math.log(r['target_probability']), 'Evaluation NLL')
        winner = r['full_argmax']
        require(type(winner) is int and 0 <= winner < vocab, 'Argmax token range')
        # Only necessary consistency conditions: full logits are not saved.
        if winner in ab:
            win_p = p[ab.index(winner)]
            require(win_p >= p.max() - 1e-12
                    and win_p >= (1 - p.sum()) / (vocab - 2) - 1e-12, 'Argmax contradicts saved masses')
        else:
            require(1 - p.sum() >= p.max() - 1e-12, 'Invalid-token argmax contradicts saved masses')
    return ab, {s: metrics([r for r in rows if r['split'] == s]) for s in ('train', 'eval')}


def validate_adapter(state, reference=None):
    require(isinstance(state, dict) and bool(state), 'Empty adapter')
    require(all(isinstance(v, torch.Tensor) and v.dtype == torch.float32
                and v.ndim == 2 and torch.isfinite(v).all() for v in state.values()), 'Invalid adapter tensor')
    prefixes = {k[:-2] for k in state}
    require(set(state) == {p + suffix for p in prefixes for suffix in ('.a', '.b')}, 'Adapter pair coverage')
    parents = {p.rsplit('.', 1)[0] for p in prefixes}
    require(prefixes == {p + '.' + q for p in parents for q in ('q_proj', 'k_proj', 'v_proj', 'o_proj')},
            'Attention projection coverage')
    for p in prefixes:
        require(state[p + '.a'].shape[0] == 8 and state[p + '.b'].shape[1] == 8,
                'Adapter rank differs from specification')
    if reference is not None:
        require(list(state) == list(reference)
                and all(state[k].shape == reference[k].shape for k in state), 'Adapter state shape/key mismatch')
    else:
        require(all(torch.count_nonzero(v) == 0 for k, v in state.items() if k.endswith('.b')),
                'Initial adapter is not zero-update')


def validate_optimizer(opt, initial):
    groups = opt['param_groups']
    require(len(groups) == 1, 'Optimizer parameter groups')
    group = groups[0]
    require(group['lr'] == .0003 and group['weight_decay'] == 0
            and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8,
            'Optimizer hyperparameters')
    ids = group['params']
    require(len(ids) == len(initial) and len(set(ids)) == len(ids)
            and set(opt['state']) == set(ids), 'Optimizer parameter coverage')
    for pid, tensor in zip(ids, initial.values()):
        state = opt['state'][pid]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Optimizer state fields')
        require(float(state['step']) == 96, 'Optimizer step count')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = state[key]
            require(isinstance(value, torch.Tensor) and value.shape == tensor.shape
                    and value.dtype == tensor.dtype and torch.isfinite(value).all(), 'Optimizer moment shape/finiteness')
        require((state['exp_avg_sq'] >= 0).all(), 'Negative optimizer second moment')


def verify(root):
    root = Path(root).resolve()
    # Do not inspect RESULT, baseline, or partial steps from an active run.
    require((root / 'MANIFEST.json').is_file(), 'Run is not finalized: MANIFEST.json absent')
    manifest = read_json(root / 'MANIFEST.json')
    require(isinstance(manifest, dict) and bool(manifest), 'Empty manifest')
    actual = {p.name for p in root.iterdir() if p.is_file()} - {'MANIFEST.json'}
    require(set(manifest) == actual, 'Manifest does not cover exactly the artifact files')
    for name, expected in manifest.items():
        path = (root / name).resolve()
        require(Path(name).name == name and path.parent == root and path.is_file(), 'Unsafe manifest path')
        require(digest(path) == expected, 'Manifest hash mismatch: ' + name)
    require('FAILED.json' not in manifest, 'Failed runs are not terminal scientific outcomes')
    result = read_json(root / 'RESULT.json')
    require(result['status'] in (STOP, COMPLETE), 'Unknown terminal status')
    arms = runner.ARMS[:1] if result['status'] == STOP else runner.ARMS
    require(list(result['results']) == list(arms), 'Completed-arm coverage/order')
    required = {'spec.json', 'cases.json', 'schedule.json', 'anchor_ids.json', 'runner_source.py',
                'dependencies.json', 'token_ids.json', 'teacher_logprobabilities.pt', 'teacher_prompts.json',
                'truthful_teacher_diagnostic.json', 'initial_adapter.pt', 'baseline.json',
                'fixed_marginals.json', 'runtime.json', 'RESULT.json'}
    required |= {arm + suffix for arm in arms for suffix in ('_adapter.pt', '_optimizer.pt', '_steps.json', '_eval.json')}
    require(set(manifest) == required, 'Terminal artifact inventory mismatch')
    source = Path(runner.__file__)
    require(digest(root / 'runner_source.py') == digest(source), 'Runner source differs from audited implementation')
    deps = [source.with_name(n) for n in ('hindsight_acquisition_calibration.py', 'hindsight_matched_learning.py',
                                       'hindsight_revelation_probe.py')]
    deps.append(source.parents[1] / 'latent_contract' / 'sender_update.py')
    require(read_json(root / 'dependencies.json') == {p.name: digest(p) for p in deps}, 'Dependency source mismatch')
    spec = read_json(root / 'spec.json')
    for key, value in dict(model=runner.MODEL, revision=runner.REVISION, seed=SEED, steps=96, batch=8,
                           lr=.0003, rank=8, alpha=16, arms=list(runner.ARMS),
                           stop_after_unqualified_truthful=True, paper_green_light=False).items():
        require(spec[key] == value, 'Specification mismatch: ' + key)
    for key in ('calibration_manifest_sha256', 'initial_adapter_sha256'):
        require(isinstance(spec[key], str) and len(spec[key]) == 64
                and all(c in '0123456789abcdef' for c in spec[key]), 'Invalid provenance digest')
    cases, schedule = dataset()
    require(read_json(root / 'cases.json') == cases and read_json(root / 'schedule.json') == schedule, 'Dataset/schedule mismatch')
    lookup = {r['id']: r for r in cases}
    original, _ = data_and_schedule()
    base = {r['id']: r for r in original}
    train = [r for r in cases if r['split'] == 'train']
    train_ids = {r['id'] for r in train}
    anchors = [r['id'] for r in train if base[r['base_id']]['anchor']]
    require(read_json(root / 'anchor_ids.json') == anchors, 'Anchor assignment mismatch')
    exposure = Counter(cid for batch in schedule for cid in batch)
    require(len(schedule) == 96 and all(len(b) == 8 for b in schedule)
            and exposure == Counter({cid: 12 for cid in train_ids}), 'Training exposure mismatch')
    require(len(anchors) == 16 and sum(exposure[cid] for cid in anchors) == 192, 'Anchor budget mismatch')
    runtime = read_json(root / 'runtime.json')
    vocab = runtime['vocabulary_size']
    require(type(vocab) is int and vocab > 2, 'Invalid vocabulary size')
    encoded = read_json(root / 'token_ids.json')
    require(set(encoded) == set(lookup), 'Student token coverage')
    def token_check(ids):
        require(isinstance(ids, list) and 0 < len(ids) <= 512
                and all(type(t) is int and 0 <= t < vocab for t in ids), 'Invalid token IDs')
    for ids in encoded.values():
        token_check(ids)
    prompts = read_json(root / 'teacher_prompts.json')
    require(len(prompts) == 192 and {(r['id'], r['report']) for r in prompts}
            == {(cid, report) for cid in train_ids for report in range(3)}, 'Teacher prompt coverage')
    for r in prompts:
        c = lookup[r['id']]
        options = base[c['base_id']]['options']
        feedback = [f'My actual preference is {o}. Please use that preference.' for o in options]
        feedback.append('Please answer using only A or B.')
        require(r['text'] == c['prompt'] + spec['template'].format(follow_up=feedback[r['report']]), 'Teacher prompt construction')
        token_check(r['token_ids'])
    baseline = read_json(root / 'baseline.json')
    ab, baseline_metrics = validate_evaluation(baseline, cases, base, vocab)
    baseline_byid = {r['id']: r for r in baseline}
    teacher = torch.load(root / 'teacher_logprobabilities.pt', weights_only=True, map_location='cpu')
    require(set(teacher) == train_ids, 'Teacher tensor coverage')
    diagnostic = read_json(root / 'truthful_teacher_diagnostic.json')
    require(set(diagnostic) == train_ids, 'Teacher diagnostic coverage')
    norm_error = 0.
    for c in train:
        q = teacher[c['id']]
        require(isinstance(q, torch.Tensor) and q.dtype == torch.float32 and q.shape == (3, vocab)
                and torch.isfinite(q).all(), 'Teacher tensor shape/finiteness')
        norm_error = max(norm_error, float(q.double().logsumexp(-1).abs().max()))
        require(norm_error <= 2e-6, 'Teacher log probabilities are not normalized')
        p = q[c['target']].double().exp()
        d = diagnostic[c['id']]
        close(d['target_probability'], float(p[ab[c['target']]]), 'Teacher target diagnostic')
        close(d['AB_mass'], float(p[ab].sum()), 'Teacher mass diagnostic')
        require(d['argmax'] == int(p.argmax()), 'Teacher argmax diagnostic')
    del teacher
    fixed = read_json(root / 'fixed_marginals.json')
    require(set(fixed) == train_ids, 'Fixed-marginal coverage')
    for c in train:
        close(fixed[c['id']], marginal(baseline_byid[c['id']]['AB_probabilities'], c['target'], .9), 'Fixed marginal arithmetic')
    initial = torch.load(root / 'initial_adapter.pt', weights_only=True, map_location='cpu')
    validate_adapter(initial)
    report = dict(status='TERMINAL_ARTIFACTS_AND_SAVED_ARITHMETIC_VERIFIED_NOT_NEURAL_REPLAY',
        decision=result['status'], manifest_files=len(manifest), teacher_max_log_normalization_error=norm_error,
        scope='No neural forward/full-logit, backward, optimizer-update, tokenizer, or external calibration replay; argmax decisions use saved token IDs.',
        metrics={'no_adaptation': baseline_metrics}, adapter_L2_change={}, anchor_exposures={}, counts=result['counts'])
    report['truthful_teacher_train'] = dict(n=len(train),
        correct=sum(diagnostic[c['id']]['argmax'] == ab[c['target']] for c in train),
        mean_target_probability=float(np.mean([diagnostic[c['id']]['target_probability'] for c in train])),
        mean_AB_mass=float(np.mean([diagnostic[c['id']]['AB_mass'] for c in train])))
    report['output_format'] = {}
    first = {}
    for arm in arms:
        steps = read_json(root / (arm + '_steps.json'))
        require(len(steps) == 96 and [s['step'] for s in steps] == list(range(1, 97)), 'Incomplete arm steps')
        anchor_count = 0
        for log, batch in zip(steps, schedule):
            require(math.isfinite(log['loss']) and math.isfinite(log['gradient_norm']) and log['gradient_norm'] >= 0,
                    'Nonfinite step or invalid gradient norm')
            require(len(log['AB_probabilities']) == 8, 'Step probability coverage')
            n_anchor = sum(cid in anchors for cid in batch)
            require(log['anchor_count'] == n_anchor, 'Step anchor count')
            anchor_count += n_anchor
            if arm in ('anchors_only', 'truthful_direct'):
                require(log['marginals'] == [], 'Unexpected supervised report weights')
            else:
                require(len(log['marginals']) == 8, 'Step marginal coverage')
            for i, cid in enumerate(batch):
                p = log['AB_probabilities'][i]
                target = lookup[cid]['target']
                expected = marginal(p, target, 0 if arm == 'truthful_kl' else .9)
                if arm == 'fixed_marginal_kl':
                    expected = fixed[cid]
                if arm not in ('anchors_only', 'truthful_direct'):
                    close(log['marginals'][i], expected, 'Step report-marginal arithmetic')
            if arm in ('anchors_only', 'truthful_direct'):
                positions = [i for i, cid in enumerate(batch) if arm == 'truthful_direct' or cid in anchors]
                expected_loss = float(np.mean([-math.log(log['AB_probabilities'][i][lookup[batch[i]]['target']])
                                              for i in positions])) if positions else 0.
                close(log['loss'], expected_loss, 'Supervised loss arithmetic')
                if not positions:
                    require(log['gradient_norm'] == 0, 'Nonzero gradient without anchors')
        first[arm] = steps[0]
        # Calibration permits .001 max full-logit discrepancy between padded
        # batching and singles. That implies <=.0005 absolute probability drift.
        close(steps[0]['AB_probabilities'], [baseline_byid[cid]['AB_probabilities'] for cid in schedule[0]],
              'Arm does not start at baseline probabilities', atol=5e-4)
        if arm in ('copying_plus_anchors', 'anchors_only'):
            require(anchor_count == 192, 'Anchor exposure mismatch')
            report['anchor_exposures'][arm] = anchor_count
        state = torch.load(root / (arm + '_adapter.pt'), weights_only=True, map_location='cpu')
        validate_adapter(state, initial)
        report['adapter_L2_change'][arm] = float(sum((state[k].double() - initial[k].double()).square().sum()
                                                    for k in initial).sqrt())
        opt = torch.load(root / (arm + '_optimizer.pt'), weights_only=True, map_location='cpu')
        validate_optimizer(opt, initial)
        final = read_json(root / (arm + '_eval.json'))
        _, replay = validate_evaluation(final, cases, base, vocab, ab)
        report['output_format'][arm] = {s: dict(
            invalid_argmax=sum(r['full_argmax'] not in ab for r in final if r['split'] == s),
            below_95_AB_mass=sum(r['AB_mass'] < .95 for r in final if r['split'] == s))
            for s in ('train', 'eval')}
        report['metrics'][arm] = replay
        require(set(result['results'][arm]) == {'train', 'eval'}, 'Summary split coverage')
        for split, m in replay.items():
            saved = result['results'][arm][split]
            require(set(saved) == set(m), 'Summary metric fields')
            for key in ('n', 'correct', 'domains'):
                require(saved[key] == m[key], 'Summary categorical arithmetic')
            for key in ('nll', 'mean_target_probability', 'min_AB_mass'):
                close(saved[key], m[key], 'Summary numeric arithmetic')
    clean_pass = qualifies(report['metrics']['truthful_kl']['eval'])
    require(clean_pass == (result['status'] == COMPLETE), 'Clean-stage stopping decision mismatch')
    if len(arms) == 6:
        # Batched current probabilities and unbatched baseline fixed weights may differ at roundoff scale.
        close(first['copying_kl']['AB_probabilities'], first['fixed_marginal_kl']['AB_probabilities'],
              'Initial matched-control policy mismatch')
        close(first['copying_kl']['marginals'], first['fixed_marginal_kl']['marginals'],
              'Initial matched-control reports mismatch', atol=5e-4)
    expected_counts = dict(forward_batches=160 + 192 * len(arms), forward_examples=288 + 864 * len(arms),
                           backwards=96 * len(arms), updates=96 * len(arms))
    require(result['counts'] == expected_counts, 'Execution count mismatch')
    require(result['paper_green_light'] is False and math.isfinite(result['elapsed']) and result['elapsed'] >= 0,
            'Invalid final metadata')
    report['elapsed_seconds'] = result['elapsed']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out is not None:
        require(args.root.resolve() not in args.out.resolve().parents, 'Audit output must be outside artifacts')
    report = verify(args.root)
    if args.out is not None:
        with args.out.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
