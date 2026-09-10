import json
from pathlib import Path
import pytest
import explicit_numeric_answer
from audit_math_policy_value_bank import audit, resolve_parser
from run_unexplored_screens import sha


def test_parser_identity_is_explicit_and_checked():
    version, legacy = resolve_parser({})
    assert version == 'legacy-v1' and legacy('#### 4/3') == '4'
    _, strict = resolve_parser({'parser': explicit_numeric_answer.VERSION,
                               'parser_sha256': sha(Path(explicit_numeric_answer.__file__))})
    assert strict('#### 4/3') == '4/3'
    for protocol in ({'parser': 'unknown'}, {'parser': explicit_numeric_answer.VERSION},
                     {'parser': explicit_numeric_answer.VERSION, 'parser_sha256': 'wrong'}):
        with pytest.raises(ValueError):
            resolve_parser(protocol)


def test_full_bank_replays_rational_answers_with_recorded_parser(tmp_path):
    inputs = [dict(id=str(i), target='4/3', split='calibration' if i < 8 else 'dev') for i in range(24)]
    prefixes = [dict(base=r['id'], index=j, text='', ended=False, has_answer=False)
                for r in inputs for j in range(2)]
    rows = [dict(base=r['id'], split=r['split'], target=r['target'], prefix_index=j,
                 sample=k, completion='#### 4/3' if k%2 else '#### 2',
                 parsed_answer='4/3' if k%2 else '2', reward=k%2,
                 ids=[9], length=1, eos=True, prefix_ended=False, prefix_has_answer=False)
            for r in inputs for j in range(2) for k in range(8)]
    contents = {'INPUTS.json': inputs, 'PREFIXES.json': prefixes,
                'MODEL.json': {'eos_token_id':9},
                'PROTOCOL.json': {'parser': explicit_numeric_answer.VERSION,
                                  'parser_sha256': sha(Path(explicit_numeric_answer.__file__))}}
    for name, value in contents.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'ROLLOUTS.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    (tmp_path/'MANIFEST.json').write_text(json.dumps({p.name:sha(p) for p in tmp_path.iterdir()}))
    result = audit(tmp_path)
    assert result['qualified'] and result['eligible_dev_accuracy']==.5
    assert result['parser']==explicit_numeric_answer.VERSION
