"""Synthetic artifact tests: no model, tokenizer, or live run is accessed."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'verify_hindsight_full_feedback.py'
SPEC = importlib.util.spec_from_file_location('verify_full_feedback', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def reseal(root):
    save_json(root / 'MANIFEST.json', {p.name: audit.digest(p) for p in root.iterdir()
                                     if p.name != 'MANIFEST.json'})


def make_bundle(root, complete=False):
    root.mkdir()
    cases, schedule = audit.dataset()
    original, _ = audit.data_and_schedule()
    base = {r['id']: r for r in original}
    byid = {r['id']: r for r in cases}
    train = [r for r in cases if r['split'] == 'train']
    anchors = [r['id'] for r in train if base[r['base_id']]['anchor']]
    source = Path(audit.runner.__file__)
    template = '\nHindsight: {follow_up}'
    spec = dict(model=audit.runner.MODEL, revision=audit.runner.REVISION, seed=audit.SEED,
                steps=96, batch=8, lr=.0003, rank=8, alpha=16, arms=list(audit.runner.ARMS),
                stop_after_unqualified_truthful=True, paper_green_light=False,
                calibration_manifest_sha256='0' * 64, initial_adapter_sha256='1' * 64, template=template)
    save_json(root / 'spec.json', spec)
    save_json(root / 'cases.json', cases)
    save_json(root / 'schedule.json', schedule)
    save_json(root / 'anchor_ids.json', anchors)
    (root / 'runner_source.py').write_bytes(source.read_bytes())
    deps = [source.with_name(n) for n in ('hindsight_acquisition_calibration.py', 'hindsight_matched_learning.py',
                                        'hindsight_revelation_probe.py')]
    deps.append(source.parents[1] / 'latent_contract' / 'sender_update.py')
    save_json(root / 'dependencies.json', {p.name: audit.digest(p) for p in deps})
    save_json(root / 'runtime.json', dict(vocabulary_size=5))
    save_json(root / 'token_ids.json', {r['id']: [2, 3] for r in cases})
    prompts, teacher, diagnostic = [], {}, {}
    for r in train:
        options = base[r['base_id']]['options']
        feedback = [f'My actual preference is {o}. Please use that preference.' for o in options]
        feedback.append('Please answer using only A or B.')
        for i, message in enumerate(feedback):
            prompts.append(dict(id=r['id'], report=i, text=r['prompt'] + template.format(follow_up=message), token_ids=[2, 3]))
        q = torch.tensor([[.97, .02, .004, .003, .003], [.02, .97, .004, .003, .003],
                          [.49, .49, .01, .005, .005]], dtype=torch.float64).log().float()
        teacher[r['id']] = q
        p = q[r['target']].double().exp()
        diagnostic[r['id']] = dict(target_probability=float(p[r['target']]), AB_mass=float(p[:2].sum()), argmax=int(p.argmax()))
    torch.save(teacher, root / 'teacher_logprobabilities.pt')
    save_json(root / 'teacher_prompts.json', prompts)
    save_json(root / 'truthful_teacher_diagnostic.json', diagnostic)

    def evaluations(qualified):
        out = []
        for r in cases:
            p = [.49, .49]
            if qualified:
                p[r['target']], p[1 - r['target']] = .97, .02
            out.append(dict(id=r['id'], domain=r['domain'], split=r['split'], target=r['target'],
                anchor=base[r['base_id']]['anchor'], target_token=r['target'],
                full_argmax=r['target'] if qualified else 0, target_probability=p[r['target']],
                nll=float(-np.log(p[r['target']])), AB_mass=sum(p), AB_probabilities=p))
        return out

    baseline = evaluations(False)
    save_json(root / 'baseline.json', baseline)
    fixed = {r['id']: audit.marginal([.49, .49], r['target'], .9).tolist() for r in train}
    save_json(root / 'fixed_marginals.json', fixed)
    initial = {}
    for projection in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        initial['model.layers.0.self_attn.' + projection + '.a'] = torch.ones((8, 2))
        initial['model.layers.0.self_attn.' + projection + '.b'] = torch.zeros((2, 8))
    torch.save(initial, root / 'initial_adapter.pt')
    opt = dict(param_groups=[dict(params=list(range(8)), lr=.0003, weight_decay=0,
                                  betas=(.9, .999), eps=1e-8)],
               state={i: dict(step=torch.tensor(96.), exp_avg=torch.zeros_like(t), exp_avg_sq=torch.ones_like(t))
                      for i, t in enumerate(initial.values())})
    arms = audit.runner.ARMS if complete else audit.runner.ARMS[:1]
    results = {}
    for arm in arms:
        logs = []
        for step, batch in enumerate(schedule, 1):
            n_anchor = sum(cid in anchors for cid in batch)
            loss = 0. if arm == 'anchors_only' and not n_anchor else -float(np.log(.49))
            logs.append(dict(step=step, loss=loss, gradient_norm=0. if not loss else .1,
                AB_probabilities=[[.49, .49] for _ in batch], anchor_count=n_anchor,
                marginals=[] if arm in ('truthful_direct', 'anchors_only') else
                    [audit.marginal([.49, .49], byid[cid]['target'], 0. if arm == 'truthful_kl' else .9).tolist()
                     for cid in batch]))
        save_json(root / (arm + '_steps.json'), logs)
        torch.save(initial, root / (arm + '_adapter.pt'))
        torch.save(opt, root / (arm + '_optimizer.pt'))
        final = evaluations(complete)
        save_json(root / (arm + '_eval.json'), final)
        results[arm] = {s: audit.metrics([r for r in final if r['split'] == s]) for s in ('train', 'eval')}
    save_json(root / 'RESULT.json', dict(status=audit.COMPLETE if complete else audit.STOP, results=results,
        counts=dict(forward_batches=160 + 192 * len(arms), forward_examples=288 + 864 * len(arms),
                    backwards=96 * len(arms), updates=96 * len(arms)), elapsed=1., paper_green_light=False))
    reseal(root)
    return root


@pytest.mark.parametrize('complete', [False, True])
def test_terminal_bundle_and_read_only(tmp_path, complete):
    root = make_bundle(tmp_path / 'run', complete)
    before = {p.name: audit.digest(p) for p in root.iterdir()}
    report = audit.verify(root)
    assert report['counts']['updates'] == (576 if complete else 96)
    assert report['decision'] == (audit.COMPLETE if complete else audit.STOP)
    assert {p.name: audit.digest(p) for p in root.iterdir()} == before
    if complete:
        assert report['anchor_exposures'] == {'copying_plus_anchors': 192, 'anchors_only': 192}


def test_unfinished_run_is_not_read(tmp_path):
    (tmp_path / 'RESULT.json').write_text('invalid incomplete JSON', encoding='utf-8')
    with pytest.raises(ValueError, match='not finalized'):
        audit.verify(tmp_path)


@pytest.mark.parametrize('corruption,match', [
    ('manifest', 'Manifest hash'), ('marginal', 'report-marginal'), ('teacher', 'not normalized'),
    ('optimizer', 'moment shape'), ('counts', 'Execution count'), ('decision', 'stopping decision'),
    ('anchor', 'anchor count'), ('metric', 'Summary numeric'), ('mass', 'A/B mass'),
])
def test_corruption_detected(tmp_path, corruption, match):
    root = make_bundle(tmp_path / 'run')
    if corruption == 'teacher':
        path = root / 'teacher_logprobabilities.pt'
        state = torch.load(path, weights_only=True)
        state[next(iter(state))] += .1
        torch.save(state, path)
    elif corruption == 'optimizer':
        path = root / 'truthful_kl_optimizer.pt'
        state = torch.load(path, weights_only=True)
        state['state'][0]['exp_avg'] = torch.zeros(1)
        torch.save(state, path)
    elif corruption in ('marginal', 'anchor'):
        path = root / 'truthful_kl_steps.json'
        logs = audit.read_json(path)
        if corruption == 'marginal':
            logs[3]['marginals'][0] = [.4, .5, .1]
        else:
            logs[3]['anchor_count'] += 1
        save_json(path, logs)
    elif corruption == 'mass':
        path = root / 'truthful_kl_eval.json'
        rows = audit.read_json(path)
        rows[0]['AB_mass'] = .8
        save_json(path, rows)
    else:
        path = root / 'RESULT.json'
        result = audit.read_json(path)
        if corruption in ('manifest', 'counts'):
            result['counts']['updates'] += 1
        elif corruption == 'metric':
            result['results']['truthful_kl']['eval']['min_AB_mass'] = .8
        elif corruption == 'decision':
            # Preserve the early-stop inventory but make its saved final evidence qualify.
            rows = audit.read_json(root / 'truthful_kl_eval.json')
            for row in rows:
                p = [.02, .02]
                p[row['target']] = .97
                row.update(AB_probabilities=p, AB_mass=.99, target_probability=.97,
                           nll=-float(np.log(.97)), full_argmax=row['target_token'])
            save_json(root / 'truthful_kl_eval.json', rows)
            result['results']['truthful_kl'] = {s: audit.metrics([r for r in rows if r['split'] == s]) for s in ('train', 'eval')}
        save_json(path, result)
    if corruption != 'manifest':
        reseal(root)
    with pytest.raises(ValueError, match=match):
        audit.verify(root)


def test_qualification_boundary():
    metric = dict(n=32, correct=29, min_AB_mass=.95, domains={'d': dict(n=4, correct=3)})
    assert audit.qualifies(metric)
    assert not audit.qualifies(dict(metric, correct=28))
    assert not audit.qualifies(dict(metric, min_AB_mass=.949999))
    assert not audit.qualifies(dict(metric, domains={'d': dict(n=4, correct=2)}))
