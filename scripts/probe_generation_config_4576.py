"""Read-only CPU probe; emit config-only receipts, never inspect outcomes.

Run using the existing 4.57.6 environment. No model weights are loaded and no
input/evidence files are written. Actual generation kwargs are inferred from
the pinned runner/upstream calls, not claimed as runtime-captured overrides.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

METHOD_SHA='fbd277c3bc70bcc944dee222b653e0702234bdeebb8826d19e9613997db9f6ef'


def digest(data): return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('policy');p.add_argument('simulator');args=p.parse_args()
    import transformers
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin
    import transformers.generation.utils as gu
    import transformers.generation.configuration_utils as gc
    if transformers.__version__!='4.57.6':raise ValueError('wrong environment version')
    source=inspect.getsource(GenerationMixin._prepare_generation_config)
    if digest(source.encode())!=METHOD_SHA:raise ValueError('unexpected installed merge implementation')
    configs={};models={};paths={'policy':Path(args.policy),'simulator':Path(args.simulator)}
    for role,path in paths.items():
        raw=(path/'generation_config.json').read_bytes()
        configs[role]=GenerationConfig.from_pretrained(path,local_files_only=True)
        models[role]=dict(generation_config_sha256=digest(raw),generation_config_file=json.loads(raw),
                          resolved_config=configs[role].to_dict())
    data=Path(args.root,'calibration_records.jsonl').read_text()
    lines=data.splitlines(); observations=[];unique={};incomplete_last_line=False
    for index,line in enumerate(lines):
        try:record=json.loads(line)
        except json.JSONDecodeError:
            if index!=len(lines)-1:raise
            incomplete_last_line=True;break
        for condition,role in [('ordinary','policy'),('explicit','policy'),('teacher','policy'),('feedback','simulator')]:
            native=record['simulator_native'] if role=='simulator' else record[condition]['native']
            passed=native['generation_config']
            before=json.dumps(passed,sort_keys=True).encode();key=role+'/'+digest(before)
            if key not in unique:
                config=GenerationConfig.from_dict(passed)
                effective,remaining=GenerationMixin._prepare_generation_config(
                    SimpleNamespace(generation_config=configs[role]),config,use_model_defaults=None,
                    input_ids=None,attention_mask=None)
                actual=effective.to_dict()
                changes={k:dict(passed=passed.get(k),effective=v) for k,v in actual.items()
                         if not k.startswith('_') and k!='transformers_version' and passed.get(k)!=v}
                unique[key]=dict(role=role,passed=passed,effective=actual,changes=changes,
                                 passed_object_unmutated=json.dumps(config.to_dict(),sort_keys=True).encode()==before,
                                 generation_mode=str(effective.get_generation_mode()),
                                 method_kwargs_generation_overrides={},method_use_model_defaults=None,
                                 non_generation_kwargs=['input_ids','attention_mask'])
            observations.append(dict(id=record['id'],condition=condition,config_receipt=key))
    print(json.dumps(dict(transformers_version=transformers.__version__,merge_method_sha256=METHOD_SHA,
        merge_method_source=source,installed_generation_utils_sha256=digest(Path(gu.__file__).read_bytes()),
        installed_configuration_utils_sha256=digest(Path(gc.__file__).read_bytes()),
        global_defaults=GenerationConfig().to_dict(),models=models,unique_config_receipts=unique,
        observed_calls=observations,incomplete_last_line=incomplete_last_line,
        snapshot_scope='Config-only read of currently complete records, not experiment completion or outcome analysis. Installed helper replay; no model forward, no RNG replay, no claim runtime effective config was logged. Generation kwargs inferred from pinned source calls.'),indent=2,allow_nan=False))


if __name__=='__main__':main()
