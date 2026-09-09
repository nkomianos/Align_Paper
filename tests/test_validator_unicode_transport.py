import json

from under_extinction.io import canonical_json, read_jsonl, write_json, write_jsonl
from validator_monoculture.runner import _append_jsonl_record, _load_partial_jsonl
from validator_monoculture.schema import canonical_json_bytes, stable_hash
from validator_monoculture.verify import _canonical_hash


def test_lone_surrogate_survives_durable_append_and_resume(tmp_path):
    row = {'raw_completion': r'{"expected":"\ud83d"}', 'parsed_tests': [{'expected': '\ud83d'}]}
    path = tmp_path / 'phase.jsonl.partial'
    _append_jsonl_record(path, row)
    assert b'\\ud83d' in path.read_bytes()
    assert _load_partial_jsonl(path, recover_torn=False) == [row]
    _append_jsonl_record(path, {'next': True})
    assert list(read_jsonl(path)) == [row, {'next': True}]


def test_all_json_outputs_preserve_surrogate_values(tmp_path):
    value = {'text': '\ud83d', '\udfff': ['é', '😀', '\\ud83d']}
    write_json(tmp_path / 'report.json', value)
    write_jsonl(tmp_path / 'records.jsonl', [value])
    assert json.loads((tmp_path / 'report.json').read_text(encoding='utf8')) == value
    assert list(read_jsonl(tmp_path / 'records.jsonl')) == [value]
    assert json.loads(canonical_json_bytes(value)) == value
    assert stable_hash(value) == stable_hash(json.loads(canonical_json(value)))
    assert _canonical_hash(value) == _canonical_hash(json.loads(canonical_json(value)))


def test_normal_unicode_commitments_do_not_change():
    value = {'text': 'é😀', 'literal_escape': '\\ud83d'}
    previous = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    assert canonical_json(value) == previous
    assert canonical_json_bytes(value) == (previous + '\n').encode('utf8')
