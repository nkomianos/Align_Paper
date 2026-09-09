"""Measure full rendered records; no truncation, prompt scoring, or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('fresh report required')
    tokenizer=Tokenizer.from_file(str(a.tokenizer))
    records=json.loads((a.records/'MANIFEST.json').read_text(encoding='utf8'))['records'];result=[]
    for run,r in sorted(records.items(),key=lambda x:x[1]['characters']):
        raw=(a.records/r['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('input changed')
        n=len(tokenizer.encode(raw.decode('utf8'),add_special_tokens=False).ids)
        result.append({'run_id':run,'raw_record_tokens':n,'input_sha256':r['sha256']})
        print(json.dumps(result[-1]),flush=True)
    report={'tokenizer_sha256':hashlib.sha256(a.tokenizer.read_bytes()).hexdigest(),
        'records':result,'qualification_limit':32768,
        'within_limit':sum(r['raw_record_tokens']<=32768 for r in result),
        'scope':'Raw record tokens only, without monitor prompt/chat-template overhead; no truncation or inference.'}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
