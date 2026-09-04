"""Read-only calibration receipt/token/score audit. No model inference or approval."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from scripts.verify_sdpo_frozen_forward import sha,read,rows,require,manifest_check

# Executed LF bytes from root's git archive 168cf41, not CRLF working-copy bytes.
RUNNER_SHA='5c23a0ca58b56d57cf08b3f59f3ac3c07f37a10e24cc9356610949306a1712ed'
STAGE_SHA='97a4d8548e404cc276b8eb43e365727680283930e18a96c3c28c388e338e5bd3'
DATA_MANIFEST_SHA='8d9da253a79ea335eb6a574a5a355c712edea24c1bd20b7ce9fab0ee39585267'
UPSTREAM_MANIFEST_SHA='612e47babd61490f6dd7c294d1adb874304d90df6ab6048d0e35a7e6807245ef'
UPSTREAM_COMMIT='3b17d2a67bd2565b9fbda495fd16a485406aa954'


def native_check(native, text, expected_input, tokenizer, max_new, sample, strip=False):
    require(native['input_ids']==[expected_input], 'native generation input differs')
    output=native['output_ids']; require(len(output)==1 and output[0][:len(expected_input)]==expected_input,'generation prefix differs')
    completion=output[0][len(expected_input):]
    require(0<len(completion)<max_new,'generation cap or empty output')
    config=native['generation_config']
    require(config['max_new_tokens']==max_new and config['do_sample']==sample,'generation config differs')
    eos=config['eos_token_id']; eos=[eos] if isinstance(eos,int) else eos
    require(eos and completion[-1] in eos and not any(x in eos for x in completion[:-1]),'native EOS termination differs')
    decoded=tokenizer.decode(completion,skip_special_tokens=True)
    require((decoded.strip() if strip else decoded)==text,'native decoded text differs')
    require(config.get('num_beams',1)==1 and config.get('repetition_penalty',1.)==1.,'unexpected decoding constraints')
    return len(completion)


def replay_judge(comparison, judge):
    scores=comparison['AB_BA_scores']; require(len(scores)==2,'missing AB/BA judge scores')
    decisions=[]
    for score in scores:
        require(set(score)=={'A','B','C'},'judge score keys')
        arrays={k:torch.tensor(v,dtype=torch.float32) for k,v in score.items()}
        require(all(v.shape==(1,) and torch.isfinite(v).all() and (v<=1e-5).all() for v in arrays.values()),'invalid judge scores')
        decisions.append(judge._decide_from_scores(arrays)[0])
    inverted=judge._invert_ab([decisions[1]])[0]
    expected=decisions[0] if decisions[0]==inverted else -1
    require(comparison['decision']==expected,'symmetric judge decision mismatch')
    return dict(decision=expected,AB=decisions[0],BA=decisions[1],order_consistent=decisions[0]==inverted)


def classes(root,tokenizer):
    user={'__name__':'frozen_user_simulator','__file__':str(root/'upstream/auxiliary/user_simulator.py')}
    exec(compile((root/'upstream/auxiliary/user_simulator.py').read_bytes(),'frozen_user','exec'),user)
    tree=ast.parse((root/'upstream/auxiliary/style_judge.py').read_bytes())
    relative=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.level]
    require(len(relative)==1 and relative[0].module=='user_simulator','unexpected judge dependency')
    tree.body.remove(relative[0])
    judge_namespace={'__name__':'frozen_style_judge','STYLE_PERSONAS':user['STYLE_PERSONAS']}
    exec(compile(ast.fix_missing_locations(tree),'frozen_judge','exec'),judge_namespace)
    model=SimpleNamespace();model.eval=lambda:model
    kwargs=dict(model=model,tokenizer=tokenizer,device=torch.device('cpu'),style='concise_casual_beginner')
    return user['StyleUserSimulator'](**kwargs),judge_namespace['StyleJudge'](**kwargs)


def verify(root,assets,tokenizer_path):
    root,assets,tokenizer_path=map(Path,(root,assets,tokenizer_path))
    manifest=manifest_check(root); freeze=read(root/'freeze.json'); status=read(root/'status.json')
    require(freeze['phase']=='calibration','not calibration')
    require(sha(root/'runner_source.py')==RUNNER_SHA==freeze['source_sha256'],'runner not frozen revision')
    require(sha(root/'PINNED_SOURCE.json')==UPSTREAM_MANIFEST_SHA==freeze['upstream_manifest_sha256'],'upstream not frozen')
    upstream=read(root/'PINNED_SOURCE.json');require(upstream['commit']==UPSTREAM_COMMIT==freeze['upstream_commit'],'upstream commit differs')
    for name,value in upstream['sha256'].items():require(sha(root/'upstream'/name)==value,'upstream dependency differs')
    require(sha(root/'data_manifest.json')==DATA_MANIFEST_SHA==sha(assets/'data_manifest.json'),'data manifest differs')
    dm=read(root/'data_manifest.json');require(dm['dataset_revision']=='b8f7d168b6f4e95b2a92e84768bd6c955bed2f29','dataset loader revision differs')
    for name,receipt in dm['files'].items():require(sha(assets/'raw'/name)==receipt['sha256'],'raw data bytes differ')
    # Execute only the separately pinned preprocessing helper, not unbound code.
    stage=Path(__file__).with_name('stage_sdpo_single_profile.py');require(sha(stage)==STAGE_SHA,'staging helper changed')
    namespace={'__name__':'frozen_stage','__file__':str(stage)};exec(compile(stage.read_bytes(),'frozen_stage','exec'),namespace)
    recreated,counts=namespace['prepare']([assets/'raw'/f for f in namespace['BATCHES']])
    require(counts==dm['eligible_counts'],'eligible pool counts differ')
    for split,n in [('train',64),('eval',32),('calibration',16)]:
        require(sha(root/(split+'.json'))==dm['splits'][split]==freeze['data_sha256'][split],'split bytes differ')
        require(read(root/(split+'.json'))==recreated[split] and len(recreated[split])==n,'frozen source selection differs')
    config=freeze['config']
    expected={'use_lora':True,'loss_mode':'full_distillation','learning_rate':5e-6,'lora_r':256,'lora_alpha':512,
              'lora_target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
              'signal_clip':0.,'distillation_topk':20,'distillation_add_tail':True,'adam_epsilon':1e-6,
              'max_grad_norm':1.,'weight_decay':0.,'train_steps_per_example':1,'lora_dropout':0.,
              'use_vllm':False,'async_training':False,'checkpoint_every_n_steps':0,'attn_implementation':'sdpa',
              'max_new_tokens':2048,'max_context_length':4096,'ignore_first_k':0}
    require(all(config.get(k)==v for k,v in expected.items()),'released recipe mismatch')
    require(freeze['profile']=='concise_casual_beginner' and freeze['seed']==9047801,'profile/seed changed')
    require(status['status'] in {'CALIBRATION_REQUIRES_ROOT_REVIEW','CALIBRATION_TIME_CAP_PARTIAL','FAILED_REPRODUCTION_INFRASTRUCTURE_OR_VALIDITY'},'unexpected phase status')
    require(not any(p.suffix in {'.pt','.safetensors'} for p in root.rglob('*')),'unexpected training checkpoint in calibration')
    if status['status']=='FAILED_REPRODUCTION_INFRASTRUCTURE_OR_VALIDITY':
        return dict(status=status['status'],verified_receipts=True,scientific_decision=None,scope='failure receipts only; no qualification')
    hashes=read(root/'model_hashes.json')
    for role,revision in [('policy','1cfa9a7208912126459214e8b04321603b3df60c'),('simulator','b968826d9c46dd6066d109eabc6255188de91218')]:
        receipt=read(root/f'expected_{role}_model.json');require(receipt['revision']==revision,'model revision differs')
        require(all(hashes[role].get(k)==v for k,v in receipt['sha256'].items()),'model receipt differs')
        for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
            require(sha(tokenizer_path/name)==hashes[role][name],'portable tokenizer differs')
    for index in range(2):
        info=read(root/f'model_loading_{index}.json')['info']
        require(not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),'model loader errors')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    simulator,judge=classes(root,tokenizer)
    require(simulator.system_persona==freeze['persona'],'persona differs')
    recorded=rows(root/'calibration_records.jsonl')
    require(len(recorded)<=16 and [r['id'] for r in recorded]==[r['id'] for r in recreated['calibration'][:len(recorded)]],'case identity/order differs')
    require(status.get('updates')==0,'calibration updated model')
    if status['status']=='CALIBRATION_REQUIRES_ROOT_REVIEW':require(len(recorded)==16,'incomplete calibration')
    findings=[]
    for rec,datum in zip(recorded,recreated['calibration']):
        require(rec['prompt']==datum['prompt'],'prompt differs')
        base=[{'role':'user','content':datum['prompt']}]
        teacher=copy.deepcopy(base);teacher[0]['content']+=config['hindsight_block_template'].format(follow_up=rec['feedback'].strip())
        explicit=[{'role':'system','content':freeze['explicit_preference']},*base]
        lengths={}
        for condition,messages in [('ordinary',base),('explicit',explicit),('teacher',teacher)]:
            text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(text,add_special_tokens=False)
            require(len(ids)<=4096,'policy input truncated')
            lengths[condition]=native_check(rec[condition]['native'],rec[condition]['text'],ids,tokenizer,2048,False)
        simtext=simulator._build_prompt_text(datum['prompt'],rec['ordinary']['text'])
        simids=tokenizer.encode(simtext,add_special_tokens=True)
        require(len(simids)<=simulator.max_input_tokens,'simulator input truncated')
        lengths['feedback']=native_check(rec['simulator_native'],rec['feedback'],simids,tokenizer,128,True,strip=True)
        require(rec['simulator_native']['generation_config']['temperature']==.7,'simulator temperature changed')
        decisions={}
        for key,a,b in [('explicit_vs_ordinary','explicit','ordinary'),('teacher_vs_ordinary','teacher','ordinary'),('teacher_vs_explicit','teacher','explicit')]:
            for left,right in [(a,b),(b,a)]:
                require(len(tokenizer.encode(judge._build_prompt_text(datum['prompt'],rec[left]['text'],rec[right]['text']),add_special_tokens=True))<=judge.max_input_tokens,'judge input truncated')
            decisions[key]=replay_judge(rec[key],judge)
        findings.append(dict(id=rec['id'],token_lengths=lengths,judge=decisions))
    return dict(status=status['status'],verified_receipts=True,calibration_manifest_sha256=sha(root/'MANIFEST.json'),
                cases=len(recorded),updates=0,scientific_decision=None,approval=False,records=findings,
                verifier_sha256=sha(Path(__file__)),scope='Source/data selection, hashes, model receipts, native text and token-context replay, AB/BA decision arithmetic only. No neural probabilities or model weights independently replayed; factual/style audit remains model-assisted/manual and cannot be inferred from pairwise scores. No automatic training approval.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('assets');p.add_argument('tokenizer');p.add_argument('report');args=p.parse_args()
    require(not Path(args.report).resolve().is_relative_to(Path(args.root).resolve()),'report inside evidence')
    torch.set_num_threads(4); result=verify(args.root,args.assets,args.tokenizer)
    with open(args.report,'x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
