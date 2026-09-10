"""Replay scene-level error of saved decisions under the known Bayesian model."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from research_pilots.clara import worlds, probabilities, truth


def main():
    root = Path('artifacts/research_pilots_20260906/clara_cpu_final')
    for name, expected in json.loads((root/'MANIFEST.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    data = (root/'ROWS.json').read_bytes()
    groups = collections.defaultdict(list)
    for r in json.loads(data):
        groups[r['error'], r['correlation'], tuple(r['observation'])].append(r)
    ws = worlds(6)
    scenes = []
    for (e, c, obs), rows in sorted(groups.items()):
        probs = probabilities(ws, obs, e, c)
        assert abs(math.fsum(probs)-1) < 1e-12
        for r in rows:
            rebuilt = math.fsum(p for w, p in zip(ws, probs) if truth(r['query'], w))
            assert abs(rebuilt-r['posterior_true']) < 1e-12
        record = dict(error=e, correlation=c, observation=obs)
        for arm in ('joint_decision', 'oracle_decision'):
            accepted = [r for r in rows if r[arm] is not None]
            risk = math.fsum(p for w, p in zip(ws, probs)
                            if any(truth(r['query'], w) != r[arm] for r in accepted))
            record[arm] = dict(any_error=risk, coverage=len(accepted)/len(rows))
        assert record['joint_decision']['any_error'] <= 1-rows[0]['joint_mass']+1e-12
        scenes.append(record)
    cells = []
    for e, c in sorted({(r['error'], r['correlation']) for r in scenes}):
        subset = [r for r in scenes if (r['error'], r['correlation']) == (e, c)]
        # Uniform prior and translation-invariant flip channel imply uniform observations.
        cells.append(dict(error=e, correlation=c, **{
            arm: {metric: math.fsum(r[arm][metric] for r in subset)/len(subset)
                  for metric in ('any_error', 'coverage')}
            for arm in ('joint_decision', 'oracle_decision')}))
    out = Path('artifacts/clara_simultaneous_20260910')
    out.mkdir(exist_ok=True)
    for name, value in [('SCENES.json', scenes), ('RESULT.json', dict(
            classification='posthoc Bayesian apparatus; unmatched coverage',
            source_rows_sha256=hashlib.sha256(data).hexdigest(), cells=cells,
            posterior_replays=len(json.loads(data)), scene_bound_checks=len(scenes)))]:
        (out/name).write_text(json.dumps(value, indent=2)+'\n')
    (out/'MANIFEST.json').write_text(json.dumps({name: hashlib.sha256((out/name).read_bytes()).hexdigest()
                                               for name in ('SCENES.json', 'RESULT.json')}, indent=2)+'\n')
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
