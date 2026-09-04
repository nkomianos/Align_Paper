"""Read-only, CPU verifier for the post-hoc 192-forward SDPO diagnostic."""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.sdpo_local_gradient_audit_v2 import local_statistics, released_full_loss
from scripts.sdpo_format_verifier_common_v3 import load_apparatus, hindsight

ORIGINAL_MANIFEST = "3765136f5452a3831c9a5620938a8b20bd28d0ad18c59465179b5f2bebc0fd41"
APPARATUS = "097c3b4ab4edc0c99b696cf91de717bfe9f811126b564dd7a5a9bc7791b54604"
CHECKER = "140e93dd3b6b04d98b64723735ee20886bd554b2c54d8784e714cfeb26bab297"
CHECKPOINTS = ("initial_adapter.pt", "adapter_step_16.pt", "final_adapter.pt")
CONTEXTS = ("base", "empty", "feedback", "explicit")
PUNCTUATION = set(":=;|{}[]*-\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, name, tolerance=2e-5):
    if actual is None or expected is None:
        require(actual is expected, "undefined " + name)
    else:
        require(np.isfinite(actual) and np.isfinite(expected) and
                np.isclose(actual, expected, rtol=2e-5, atol=tolerance), "numeric " + name)


def manifest_check(root):
    manifest = read(root / "MANIFEST.json")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    require(set(manifest) == actual, "manifest file inventory differs")
    for name, value in manifest.items():
        target = (root / name).resolve()
        require(target.is_relative_to(root.resolve()) and not (root/name).is_symlink(), "unsafe manifest path")
        require(sha(target) == value, "manifest checksum: " + name)
    return manifest


def selected_cases(originals):
    require(len(originals) == len({r['id'] for r in originals}) == 32, "original calibration IDs")
    good = sorted((r for r in originals if r['score']['joint']), key=lambda r: r['id'])
    require(len(good) == 8, "original correct stratum")
    selected = good[:]
    for style, n in (("json", 3), ("table", 3), ("bullets", 2)):
        pool = sorted((r for r in originals if not r['score']['joint'] and r['preference'] == style), key=lambda r: r['id'])
        require(len(pool) >= n, "original incorrect stratum")
        selected.extend(pool[:n])
    return selected


def offsets(tokenizer, ids):
    for i in range(1, len(ids)):
        if ids[i] not in tokenizer.all_special_ids and set(tokenizer.decode([ids[i]], skip_special_tokens=False)) & PUNCTUATION:
            return [0, i]
    raise ValueError("no frozen punctuation offset")


def checkpoint_receipt(receipt, checkpoint_path):
    require(receipt['checkpoint_sha256'] == sha(checkpoint_path) and receipt['exact_equal'] is True,
            'checkpoint load receipt identity')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(isinstance(state, dict) and bool(state), 'empty checkpoint')
    require(set(state) == set(receipt['tensor_sha256']) == set(receipt['tensor_shapes']), 'checkpoint tensor keys')
    for name, value in state.items():
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and torch.isfinite(value).all(),
                'invalid adapter tensor')
        require(list(value.shape) == receipt['tensor_shapes'][name], 'adapter shape differs')
        require(hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == receipt['tensor_sha256'][name],
                'loaded adapter tensor receipt differs')
    return len(state)


def verify(root, original, tokenizer_path, expected_runner_sha):
    root, original, tokenizer_path = map(Path, (root, original, tokenizer_path))
    require(sha(original/'MANIFEST.json') == ORIGINAL_MANIFEST, "wrong source study")
    original_manifest = manifest_check(original)
    manifest = manifest_check(root)
    freeze = read(root/'freeze.json')
    require(sha(root/'runner_source.py') == expected_runner_sha, "diagnostic runner not externally pinned")
    require(freeze['sources']['runner_source.py'] == expected_runner_sha, "runner freeze inconsistency")
    for name, value in freeze['sources'].items():
        require(sha(root/name) == value, "source archive mismatch")
    for name, value in freeze['input_sha256'].items():
        require(sha(original/name) == value, "source input mismatch: " + name)
    required_inputs = {'MANIFEST.json', 'calibration_original.jsonl', 'calibration_explicit.jsonl',
                       'calibration_teacher.jsonl', 'PREPARATION_MANIFEST.json', 'model.json',
                       'lora_source.py', 'upstream_loss_source.py', *CHECKPOINTS}
    require(set(freeze['input_sha256']) == required_inputs, "missing frozen dependency")
    require(sha(root/'lora_source.py') == sha(original/'lora_source.py'), "LoRA helper changed")
    require(sha(root/'upstream_loss_source.py') == sha(original/'upstream_loss_source.py'), "upstream changed")
    require(freeze['checkpoints'] == list(CHECKPOINTS) and freeze['contexts'] == list(CONTEXTS), "factorial design changed")
    require(freeze['calls'] == 192 and freeze['layout_delimiters'] == sorted(PUNCTUATION), "case count/positions changed")
    require(freeze['time_cap_seconds'] == 2700, "runtime cap changed")
    require(freeze['prepared_calibration_sha256'] == sha(original/'calibration.json'), "calibration bytes changed")
    old_model, model_meta = read(original/'model.json'), read(root/'model.json')
    require(model_meta['revision'] == old_model['revision'], "model revision changed")
    require(all(model_meta['sha256'].get(k) == v for k,v in old_model['sha256'].items()), "model identity differs")
    for path in tokenizer_path.iterdir():
        if path.is_file() and path.suffix in {'.json', '.txt', '.jinja'} and path.name not in {'weight_receipt.json'}:
            require(model_meta['sha256'].get(path.name) == sha(path), "tokenizer file not model-bound: " + path.name)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
        require((tokenizer_path/name).is_file(), "missing portable tokenizer file")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    apparatus = load_apparatus(original, APPARATUS, CHECKER)
    prepared = {r['id']: r for r in read(original/'calibration.json')}
    originals = rows(original/'calibration_original.jsonl')
    chosen = selected_cases(originals)
    require(freeze['selected_ids'] == [r['id'] for r in chosen], "post-hoc selection changed")
    views = {kind: {r['id']: r for r in rows(original/file)} for kind,file in
             (('explicit','calibration_explicit.jsonl'), ('feedback','calibration_teacher.jsonl'))}
    expected_cases = []
    for case in chosen:
        raw = prepared[case['id']]
        require(apparatus['score'](case['text'], raw) == case['score'], "frozen score replay mismatch")
        def encode(messages):
            return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        actual_feedback = apparatus['feedback'](case['text'], raw)
        contexts = dict(base=encode(raw['prompt']), empty=encode(hindsight(raw['prompt'], '')),
                        feedback=encode(hindsight(raw['prompt'], actual_feedback)), explicit=encode(raw['calibration_prompt']))
        require(contexts['base'] == case['prompt_ids'], "original base tokenization differs")
        for kind in views:
            require(contexts[kind] == views[kind][case['id']]['prompt_ids'], "frozen teacher context differs")
        expected_cases.append(dict(id=case['id'], preference=case['preference'], original_joint=case['score']['joint'],
            contexts=contexts, completion_ids=case['completion_ids'], positions=offsets(tok, case['completion_ids']),
            original_logps=case['token_logprobs'], original_text=case['text']))
    require(read(root/'cases.json') == expected_cases, "saved cases/indices/contexts differ")
    status = read(root/'status.json')
    allowed = {'COMPLETE_FIXED_FORWARD_DIAGNOSTIC','TIME_CAP_PRELOAD','TIME_CAP_PARTIAL','TIME_PROJECTION_STOP','INVALID_MODEL_LOADING'}
    require(status['status'] in allowed, "unknown termination status")
    if status['status'] != 'TIME_CAP_PRELOAD':
        info = read(root/'loading_info.json')
        loading_failed = any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        require(loading_failed == (status['status']=='INVALID_MODEL_LOADING'), "loading status inconsistency")
    recorded = rows(root/'records.jsonl') if (root/'records.jsonl').exists() else []
    require(len(recorded) == status.get('forwards',0), "forward count mismatch")
    require(len(recorded) <= 192, 'excess forwards')
    if status['status'] == 'COMPLETE_FIXED_FORWARD_DIAGNOSTIC':
        require(len(recorded) == 192, "incomplete factorial")
    planned = list(itertools.product(CHECKPOINTS, range(16), CONTEXTS))
    receipts = [name for name in CHECKPOINTS if (root/(name+'.loaded.json')).exists()]
    require(receipts == list(CHECKPOINTS[:len(receipts)]), 'non-prefix checkpoint load receipts')
    for name in receipts:
        checkpoint_receipt(read(root/(name+'.loaded.json')),original/name)
    require(all(row['checkpoint'] in receipts for row in recorded), 'missing checkpoint load receipt')
    if status['status']=='COMPLETE_FIXED_FORWARD_DIAGNOSTIC':
        require(len(receipts)==3, 'incomplete checkpoint receipts')
    loss_class = released_full_loss((root/'upstream_loss_source.py').read_bytes())
    audits, logits_files = [], set()
    cosine_discrepancies = []
    original_null_convention_discrepancies = []
    base_logits = None
    for index, row in enumerate(recorded):
        checkpoint, case_index, context = planned[index]
        case = expected_cases[case_index]
        require((row['checkpoint'],row['case_index'],row['context']) == (checkpoint,case_index,context), "forward order/identity differs")
        require(row['call'] == index+1 and row['id'] == case['id'] and row['positions'] == case['positions'], "forward case/offset differs")
        require(row['checkpoint_sha256'] == original_manifest[checkpoint] and row['checkpoint_loaded_exact'] is True, "checkpoint identity differs")
        expected_path = f"logits/{index+1:03d}.pt"
        require(row['logits_path'] == expected_path and row['logits_sha256'] == manifest[expected_path], "logit receipt mismatch")
        logits_files.add(expected_path)
        logits = torch.load(root/expected_path, map_location='cpu', weights_only=True)
        require(isinstance(logits,torch.Tensor) and logits.ndim==2 and logits.shape[0]==2 and logits.shape[1]>20, "not two full-vocabulary rows")
        require(logits.dtype == torch.float32 and torch.isfinite(logits).all(), "invalid logits")
        require(row['logits_shape'] == list(logits.shape) and row['logits_dtype']=='torch.float32', "logit metadata differs")
        require(logits.shape[1] == read(tokenizer_path/'config.json')['vocab_size'], "not full vocabulary")
        logps = row['completion_logps']
        require(len(logps)==len(case['completion_ids']) and all(np.isfinite(v) and v<=1e-5 for v in logps), "invalid completion log probabilities")
        require(np.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>=0, "invalid timing")
        if context=='base': base_logits=logits
        require(len(row['fixed_feedback_local_moments'])==2, "missing local moments")
        for slot, offset in enumerate(case['positions']):
            token = case['completion_ids'][offset]
            near(float(logits[slot].log_softmax(-1)[token]),logps[offset], 'selected completion logp')
            audit = local_statistics(base_logits[slot],logits[slot],token,loss_class)
            saved = row['fixed_feedback_local_moments'][slot]
            mapping = {'selected_advantage':'observed_advantage', 'fixed_feedback_expected_ascent_norm':'expected_gradient_norm',
                       'fixed_feedback_variance_trace':'sampled_gradient_trace_variance', 'sampled_ascent_norm':'observed_gradient_norm',
                       'sampled_vs_expected_cosine':'observed_expected_gradient_cosine', 'top20_tail_reverse_kl':'released_topk_tail_reverse_kl',
                       'top20_tail_descent_norm':'topk_gradient_norm', 'top20_tail_vs_expected_cosine':'topk_full_gradient_cosine'}
            for key, independent in mapping.items():
                if 'cosine' not in key:
                    near(saved[key],audit[independent],key)
                    continue
                actual, expected = saved[key], audit[independent]
                if actual is not None:
                    require(np.isfinite(actual) and -1.00000001 <= actual <= 1.00000001, 'invalid saved cosine range')
                old_null = audit['sampled_cosine_numerically_small' if key=='sampled_vs_expected_cosine' else 'topk_cosine_numerically_small']
                if actual is not None and old_null:
                    original_null_convention_discrepancies.append(dict(call=index+1,id=case['id'],checkpoint=checkpoint,
                        context=context,offset=offset,metric=key,saved=actual,raw_recomputed=expected,
                        original_verifier_convention=None,sampled_gradient_norm=audit['observed_gradient_norm'],
                        expectation_gradient_norm=audit['expected_gradient_norm'],topk_gradient_norm=audit['topk_gradient_norm'],
                        autograd_analytic_max_error=audit['categorical_expectation_gradient_max_error']))
                match = ((actual is None and expected is None) or
                         (actual is not None and expected is not None and np.isclose(actual,expected,rtol=2e-5,atol=2e-5)))
                if not match:
                    cosine_discrepancies.append(dict(call=index+1,id=case['id'],checkpoint=checkpoint,context=context,
                        offset=offset,metric=key,saved=actual,recomputed=expected,
                        sampled_gradient_norm=audit['observed_gradient_norm'],
                        expectation_gradient_norm=audit['expected_gradient_norm'],
                        topk_gradient_norm=audit['topk_gradient_norm'],
                        autograd_analytic_max_error=audit['categorical_expectation_gradient_max_error']))

            near(saved['student_selected_logp'],float(base_logits[slot].double().log_softmax(-1)[token]), 'student selected')
            near(saved['teacher_selected_logp'],float(logits[slot].double().log_softmax(-1)[token]), 'teacher selected')
            audits.append(dict(call=index+1,id=case['id'],checkpoint=checkpoint,context=context,offset=offset,**audit))
        if checkpoint==CHECKPOINTS[0] and context=='base':
            near(row['original_base_max_logprob_gap'], max(abs(a-b) for a,b in zip(logps,case['original_logps'])), 'original base gap')
    require({k for k in manifest if k.startswith('logits/')} == logits_files, 'extra/missing logit files')
    return dict(status=status['status'], verified=False, scalar_arithmetic_verified=True, directional_metrics_verified=False,
                verification_status='PARTIAL_ARITHMETIC_REPLAY_DIRECTIONAL_METRICS_UNRESOLVED',
                original_null_convention_discrepancy_count=len(original_null_convention_discrepancies),
                original_null_convention_discrepancies=original_null_convention_discrepancies,
                cosine_discrepancy_count=len(cosine_discrepancies), cosine_discrepancies=cosine_discrepancies, forwards=len(recorded), original_manifest_sha256=ORIGINAL_MANIFEST,
                diagnostic_manifest_sha256=sha(root/'MANIFEST.json'), runner_sha256=expected_runner_sha,
                verifier_sha256=sha(Path(__file__)), local_math_sha256=sha(Path(__file__).with_name('sdpo_local_gradient_audit_v2.py')),
                scientific_decision=None, local_position_audits=audits,
                verifier_revision_note='Forensic metricwise replay: retains every non-directional check and all tolerances. Records cosine discrepancies instead of accepting or hiding them. Raw nonzero cosine convention and centered onehot arithmetic are as described in V2. No directional inference is certified.',
                verification_scope='Checks bytes, frozen cases/contexts, identities, selected-logit arithmetic and released top20-tail local gradients. No neural forward, optimizer, checkpoint-execution, or full-completion-logprob replay; model-weight hashes are remote receipts, not locally recomputed weights. Fixed feedback/prefix only, not an end-to-end unbiased gradient or semantic correctness measure.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('original'); p.add_argument('report')
    p.add_argument('--tokenizer-path',required=True); p.add_argument('--expected-runner-sha',required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    report_path=Path(args.report).resolve()
    require(not report_path.is_relative_to(Path(args.root).resolve()) and not report_path.is_relative_to(Path(args.original).resolve()), 'report must be outside evidence')
    result=verify(args.root,args.original,args.tokenizer_path,args.expected_runner_sha)
    with report_path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='local_position_audits'},indent=2))
