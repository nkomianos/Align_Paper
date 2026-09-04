"""Portable, read-only verification with explicit source/model path mappings.

All non-weight bytes must be local and hash-identical. Large weights may remain
remote only with a separate checksum receipt, compared to the run's frozen
hashes; the report explicitly distinguishes that from local weight verification.
Neither receipt nor hashes certify scientific validity or neural replay.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def run(root, output, mapping_path, weight_receipt_path=None):
    from interaction_sprint.byte_clock_coupling import digest
    module_path=Path(__file__).with_name('analyze_gpu_coupling_clarification.py')
    spec=importlib.util.spec_from_file_location('frozen_squad_analysis',module_path)
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    mappings=json.loads(mapping_path.read_text())
    assert isinstance(mappings,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in mappings.items())
    entries=sorted(mappings.items(),key=lambda item:len(item[0]),reverse=True)
    def resolve(value):
        text=str(value).replace('\\','/')
        for original,destination in entries:
            original=original.replace('\\','/').rstrip('/')
            if text==original or text.startswith(original+'/'):
                return Path(destination)/text[len(original):].lstrip('/')
        return Path(value)
    frozen=json.loads((root/'FROZEN.json').read_text())
    receipt=json.loads(weight_receipt_path.read_text()) if weight_receipt_path else {}
    remote_weights={};used=[]
    for model in frozen['config']['models']:
        for name,expected in frozen['model_files'][model['name']].items():
            if name.endswith('.safetensors'):
                original=model['path'].rstrip('/')+'/'+name
                local=resolve(original)
                if not local.exists():
                    assert receipt.get(original)==expected, 'Missing or mismatched remote weight checksum receipt'
                    remote_weights[str(local.resolve())]=(original,expected)
    def portable_digest(value):
        local=resolve(value)
        if local.exists():return digest(local)
        original,expected=remote_weights[str(local.resolve())]
        used.append(original)
        return expected
    analyzer.Path=resolve
    analyzer.digest=portable_digest
    # The unchanged scientific scorer and verification logic are shared; only
    # location resolution and explicitly attested missing weights are adapted.
    audit_path=output.with_name(output.name+'.portability.json')
    if output.exists() or audit_path.exists():raise FileExistsError(output)
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Analysis must be outside the immutable evidence root')
    analyzer.analyze(root,output)
    with audit_path.open('x') as f:
        json.dump({'evidence_manifest_sha':digest(root/'MANIFEST.json'),
                   'analysis_sha':digest(output),'portable_verifier_sha':digest(__file__),
                   'original_analysis_source_sha':digest(module_path),
                   'mapping_sha':digest(mapping_path),
                   'weight_receipt_sha':digest(weight_receipt_path) if weight_receipt_path else None,
                   'weights_remote_checksum_attested_not_locally_rehashed':sorted(set(used)),
                   'scope':'All evidence/source/tokenizer/input bytes locally hashed; listed model weights use independent remote checksum receipt. No neural or full sampler replay.'},f,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('output',type=Path);parser.add_argument('mapping',type=Path)
    parser.add_argument('--weight-receipt',type=Path)
    args=parser.parse_args();run(args.root,args.output,args.mapping,args.weight_receipt)
