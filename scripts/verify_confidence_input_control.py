"""Independent token, terminal-format, coverage and frozen-gate verification."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def parse(text, marker, choices):
    # Separate implementation: inspect only the final nonempty line.
    line = text.strip().splitlines()[-1] if text.strip() else ''
    pieces = line.strip().split(':')
    return pieces[1].strip() if (len(pieces) == 2 and pieces[0] == marker
                                and pieces[1].strip() in choices) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'data', 'source', 'tokenizer', 'tokenizer-provenance', 'out']:
        ap.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    manifest = read(a.run/'MANIFEST.json')
    assert {'RESULT.json', 'ROWS.jsonl'} <= set(manifest)
    for name, expected in manifest.items():
        assert Path(name).name == name and sha(a.run/name) == expected
    result = read(a.run/'RESULT.json')
    assert sha(a.source) == result['source_sha256']
    assert sha(a.data) == result['data_sha256']
    provenance = read(a.tokenizer_provenance)
    for name in ['tokenizer.json', 'generation_config.json']:
        assert sha(a.tokenizer/name) == provenance['base_files'][name]
    decoder = Tokenizer.from_file(str(a.tokenizer/'tokenizer.json'))
    eos = read(a.tokenizer/'generation_config.json')['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    data = read(a.data)
    expected = data[:20] if result['phase'] == 'smoke' else data[20:]
    rows = [json.loads(s) for s in (a.run/'ROWS.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows] == [r['id'] for r in expected]
    nvalid = ncorrect = nabstain = ntokens = capped = 0
    for r, item in zip(rows, expected):
        parsed = {}
        for key, marker, letters in [('answer_generation','FINAL_ANSWER','ABCD'),
                                      ('decision_generation','FINAL_DECISION','AB')]:
            if key not in r:
                continue
            g = r[key]
            assert decoder.decode(g['ids'], skip_special_tokens=True).strip() == g['text']
            assert decoder.decode(g['input_ids'], skip_special_tokens=False) == g['prompt']
            assert item['question'] in g['prompt']
            for choice in item['choices']:
                assert choice in g['prompt']
            ended = bool(g['ids'] and g['ids'][-1] in eos)
            assert ended == g['ended']
            parsed[key] = parse(g['text'], marker, letters) if ended else None
            ntokens += len(g['ids'])
            capped += not ended
        answer = parsed.get('answer_generation')
        decision = parsed.get('decision_generation')
        av, dv = answer is not None, decision is not None
        assert av == r['answer_valid'] and dv == r['decision_valid']
        assert item['answer'] == r['gold']
        correct = av and ord(answer)-65 == item['answer']
        abstain = dv and decision == 'B'
        assert correct == r['correct'] and abstain == r['abstain']
        if av:
            assert '\nYour previous answer: '+answer in r['decision_generation']['prompt']
        nvalid += av and dv
        ncorrect += correct
        nabstain += abstain
    admitted = result['phase'] == 'smoke' and nvalid >= 19 and 5 <= ncorrect <= 18 and 3 <= nabstain <= 17
    assert (nvalid,ncorrect,nabstain,admitted) == (result['jointly_valid'],result['correct'],result['abstain'],result['admitted'])
    report = dict(manifest_files_verified=len(manifest), rows=len(rows),jointly_valid=nvalid,
                  correct=ncorrect,abstain=nabstain,admitted=admitted,generated_tokens=ntokens,
                  capped_outputs=capped,scope='Token/text and scoring replay; not a neural rerun or independent dataset gold audit.')
    with a.out.open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
