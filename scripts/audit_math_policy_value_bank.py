"""Saved-row qualification and paired-prefix DEV ranking audit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from run_reasoning_bank import answer
import explicit_numeric_answer
from run_unexplored_screens import sha,dump


def resolve_parser(protocol):
    version=protocol.get('parser','legacy-v1')
    if version=='legacy-v1':
        parser=answer;path=Path(__file__).with_name('run_reasoning_bank.py')
    elif version==explicit_numeric_answer.VERSION:
        parser=explicit_numeric_answer.answer;path=Path(explicit_numeric_answer.__file__)
    else:
        raise ValueError('Unsupported recorded parser: '+str(version))
    # Unversioned historical banks did not record a separate parser hash.
    if 'parser' in protocol:
        if protocol.get('parser_sha256') != sha(path):
            raise ValueError('Recorded parser source hash does not match replay code')
    return version,parser


def audit(root):
    for name,h in json.loads((root/'MANIFEST.json').read_text()).items():assert sha(root/name)==h
    version,parse=resolve_parser(json.loads((root/'PROTOCOL.json').read_text()))
    data={r['id']:r for r in json.loads((root/'INPUTS.json').read_text())}
    prefixes={(r['base'],r['index']):r for r in json.loads((root/'PREFIXES.json').read_text())}
    rows=[json.loads(x) for x in (root/'ROLLOUTS.jsonl').read_text().splitlines()]
    assert len(rows)==384 and {(r['base'],r['prefix_index'],r['sample']) for r in rows}=={
        (base,prefix,sample) for base in data for prefix in (0,1) for sample in range(8)}
    eos_id=json.loads((root/'MODEL.json').read_text())['eos_token_id']
    groups=defaultdict(list);eligible=[]
    for base,row in data.items():
        if row['split']=='dev' and not any(prefixes[base,j]['ended'] or prefixes[base,j]['has_answer'] for j in (0,1)):eligible.append(base)
    for r in rows:
        source=data[r['base']];prefix=prefixes[r['base'],r['prefix_index']]
        assert source['target']==r['target'] and source['split']==r['split']
        parsed=parse(prefix['text']+r['completion']);assert parsed==r['parsed_answer']
        assert r['reward']==int(parsed==source['target']) and r['length']==len(r['ids'])
        assert r['eos']==bool(r['ids'] and r['ids'][-1]==eos_id)
        assert eos_id not in r['ids'][:-1] and 0<len(r['ids'])<=2048
        assert r['prefix_ended']==prefix['ended'] and r['prefix_has_answer']==prefix['has_answer']
        groups[r['base'],r['prefix_index']].append(r['reward'])
    assert all(len(v)==8 for v in groups.values()) and len(groups)==48
    dev=[r for r in rows if r['base'] in eligible]
    coverage=sum(r['parsed_answer'] is not None for r in rows)/384
    eos=sum(r['eos'] for r in rows)/384
    accuracy=sum(r['reward'] for r in dev)/len(dev) if dev else None
    qualified=coverage>=.95 and eos>=.90 and len(eligible)>=12 and .10<=accuracy<=.90
    return {'scope':'DEV bank qualification; no learning or full parameter gradients','parser':version,'parse_coverage':coverage,'eos_rate':eos,
        'eligible_dev_questions':eligible,'eligible_dev_accuracy':accuracy,'qualified':qualified,
        'prefix_success':{base:[sum(groups[base,j])/8 for j in (0,1)] for base in eligible},'manifest_sha256':sha(root/'MANIFEST.json')}


def compare(first,second):
    a=audit(first);b=audit(second)
    if a['parser'] != b['parser']:raise ValueError('Cannot compare banks with different answer parsers')
    assert json.loads((first/'INPUTS.json').read_text())==json.loads((second/'INPUTS.json').read_text())
    assert json.loads((first/'PREFIXES.json').read_text())==json.loads((second/'PREFIXES.json').read_text())
    assert a['eligible_dev_questions']==b['eligible_dev_questions']
    rows=[]
    for base in a['eligible_dev_questions']:
        x=a['prefix_success'][base];y=b['prefix_success'][base];dx=x[0]-x[1];dy=y[0]-y[1]
        rows.append({'base':base,'first':x,'second':y,'reversed':dx*dy<0,'large_reversal':dx*dy<0 and abs(dx)>=.5 and abs(dy)>=.5})
    large=sum(r['large_reversal'] for r in rows)
    return {'first':a,'second':b,'rows':rows,'large_reversals':large,
        'route':'INVALID_BANK_QUALIFICATION' if not(a['qualified'] and b['qualified']) else ('INDEPENDENT_SEED_REPLICATION_NEEDED' if large>=4 else 'STOP_NO_DECISIVE_RANK_REVERSAL'),
        'scope':'single model family, public DEV integer-answer MATH; no novel algorithm or paper green light'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('--second',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(a.first,a.second) if a.second else audit(a.first);dump(a.out,result);print(json.dumps(result,indent=2))
