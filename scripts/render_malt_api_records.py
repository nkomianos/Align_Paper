"""Reversible API-record representation; never infers executed candidates."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def pack(samples):
    # Intern full values, including nulls and message metadata. The source run
    # metadata containing labels/model/run ID never enters this function.
    messages=[];index={};requests=[]
    def intern(message):
        key=canonical(message)
        if key not in index:
            index[key]=len(messages);messages.append(message)
        return index[key]
    for sample in samples:
        if set(sample)!={'input','output','metadata'}:
            raise ValueError('unrecognized sample fields')
        requests.append({'input':[intern(m) for m in sample['input']],
            'output_candidates':[[intern(m) for m in group] for group in sample['output']],
            'metadata':sample['metadata']})
    return {'messages':messages,'requests':requests}


def unpack(record):
    pool=record['messages']
    return [{'input':[pool[i] for i in r['input']],
        'output':[[pool[i] for i in group] for group in r['output_candidates']],
        'metadata':r['metadata']} for r in record['requests']]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    manifest_raw=(a.raw/'MANIFEST.json').read_bytes();source=json.loads(manifest_raw)
    a.out.mkdir(parents=True,exist_ok=False);records={}
    for run,entry in source['records'].items():
        compressed=(a.raw/entry['file']).read_bytes()
        if hashlib.sha256(compressed).hexdigest()!=entry['compressed_sha256']:
            raise ValueError('source record changed')
        raw=gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest()!=entry['canonical_json_sha256']:
            raise ValueError('source JSON changed')
        row=json.loads(raw);packed=pack(row['samples'])
        if unpack(packed)!=row['samples']:raise ValueError('non-reversible representation')
        # Metadata labels remain outside the model input. Literal source messages
        # are untouched and can still mention identity or evaluation details.
        encoded=canonical(packed).encode('utf8');target=a.out/f'run_{run}.json'
        target.write_bytes(encoded)
        records[run]={'file':target.name,'sha256':hashlib.sha256(encoded).hexdigest(),
            'characters':len(encoded.decode('utf8')),'messages':len(packed['messages']),
            'requests':len(packed['requests']),'roundtrip_passed':True}
    (a.out/'MANIFEST.json').write_text(json.dumps({'records':records,
        'source_manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'executed_path_certified':False,'monitor_ready':False,
        'scope':'Complete API samples with interned messages; candidate groups do not assert execution.'},indent=2),encoding='utf8')
    print(json.dumps({'runs':len(records),'max_characters':max(r['characters'] for r in records.values())}))


if __name__=='__main__':main()
