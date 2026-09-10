"""Finite-grammar neural gradient apparatus; no learning/paper claim."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from action_marginal_policy import initial_offsets, policy, exact_gradient_moments
from latent_contract.sender_update import LoRALinear
from run_unexplored_screens import dump, sha


def contexts():
    directions = ('north', 'south', 'east', 'west')
    return [{'id': f'move_{i}', 'target': i,
             'prompt': f'Use the move tool to travel one cell {direction}. Return only a JSON tool call.',
             'aliases': [[json.dumps({'tool': 'move', 'direction': d}, separators=(',', ':')),
                          json.dumps({'direction': d, 'tool': 'move'}, separators=(',', ':'))]
                         for d in directions]}
            for i, direction in enumerate(directions)]


def encode_choices(tok, row, canonical):
    rendered = tok.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix = tok.encode(rendered, add_special_tokens=False)
    choices = [(a, s) for a, aliases in enumerate(row['aliases'])
               for s in (aliases[:1] if canonical else aliases)]
    suffixes = [tok.encode(s, add_special_tokens=False) + [tok.eos_token_id] for _, s in choices]
    return rendered, prefix, choices, suffixes


def score_choices(model, prefix, suffixes, pad_id, device):
    # Right padding permits direct positions in the original token stream.
    length = len(prefix) + max(map(len, suffixes))
    ids = torch.tensor([prefix + s + [pad_id] * (length-len(prefix)-len(s)) for s in suffixes], device=device)
    mask = torch.tensor([[1] * (len(prefix)+len(s)) + [0] * (length-len(prefix)-len(s)) for s in suffixes], device=device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    values = []
    for i, suffix in enumerate(suffixes):
        selected = logits[i, len(prefix)-1:len(prefix)+len(suffix)-1].float()
        values.append(selected.log_softmax(-1).gather(1, torch.tensor(suffix, device=device)[:, None]).sum())
    return torch.stack(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; no launch')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(2026091044)
    tok = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.snapshot, local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    # Fixed restricted adapter organism; do not generalize to full-model gradients.
    attention = model.model.layers[-1].self_attn
    for leaf in ('q_proj', 'v_proj'):
        setattr(attention, leaf, LoRALinear(getattr(attention, leaf), rank=2, alpha=4))
    parameters = [p for p in model.parameters() if p.requires_grad]
    data = contexts()
    for row in data:
        for aliases in row['aliases']:
            assert json.loads(aliases[0]) == json.loads(aliases[1])
    dump(args.out/'INPUTS.json', data)
    dump(args.out/'PROTOCOL.json', {'scope': 'synthetic gradient apparatus, no learning',
        'model_snapshot': str(args.snapshot), 'seed': 2026091044,
        'trainable': 'last attention q/v rank2 LoRA only', 'initial_mass': 'uniform fixed offsets',
        'gate': 'all mean gradients relative error <=.02 (BF16 backward); mass error <=1e-5; median alias variance reduction >=20%',
        'sources': {name: sha(Path(__file__).parent/name) for name in
                    ('run_action_gradient_diagnostic.py', 'action_marginal_policy.py')},
        'lora_source': sha(Path(__import__('latent_contract.sender_update', fromlist=['x']).__file__))})
    dump(args.out/'MODEL.json', {'weights': {p.name: sha(p) for p in args.snapshot.glob('*.safetensors')},
        'eos_token_id': tok.eos_token_id, 'torch': torch.__version__})
    results = []
    started = time.monotonic()
    for row in data:
        for canonical in (False, True):
            rendered, prefix, choices, suffixes = encode_choices(tok, row, canonical)
            torch.cuda.synchronize(); began = time.monotonic()
            scores = score_choices(model, prefix, suffixes, tok.eos_token_id, 'cuda')
            action_ids = torch.tensor([a for a, _ in choices], device='cuda')
            offsets = initial_offsets(scores, action_ids)
            strings, actions = policy(scores, action_ids, offsets)
            rewards = torch.tensor([float(a == row['target']) for a in range(4)], device='cuda')
            moments = exact_gradient_moments(scores, action_ids, offsets, rewards, parameters)
            reference_parts = torch.autograd.grad(-(actions.exp()*rewards).sum(), parameters)
            reference = torch.cat([x.reshape(-1) for x in reference_parts]).double()
            torch.cuda.synchronize()
            label = row['id'] + ('_canonical' if canonical else '_aliases')
            torch.save({'reference': reference.cpu(), 'moments': {method: {k: v.cpu() for k, v in value.items()}
                       for method, value in moments.items()}}, args.out/(label+'.pt'))
            errors = {method: float((value['mean']-reference).norm()/reference.norm().clamp_min(1e-12))
                      for method, value in moments.items()}
            variance = {method: float(value['variance_trace']) for method, value in moments.items()}
            record = {'id': row['id'], 'canonical': canonical, 'rendered': rendered,
                'prefix_ids': prefix, 'choices': choices, 'suffix_ids': suffixes,
                'scores': scores.detach().tolist(), 'offsets': offsets.tolist(),
                'action_mass': actions.detach().exp().tolist(), 'relative_mean_error': errors,
                'reference_norm': float(reference.norm()), 'variance': variance,
                'reduction': 1-variance['marginal']/max(variance['spelling'], 1e-30),
                'seconds': time.monotonic()-began}
            results.append(record); dump(args.out/'RESULTS.partial.json', results)
            print(json.dumps({'completed': label, 'seconds': time.monotonic()-started}), flush=True)
            del moments, scores, strings, actions, reference_parts, reference
    valid = all(max(r['relative_mean_error'].values()) <= .02 and
                max(abs(x-.25) for x in r['action_mass']) <= 1e-5 for r in results)
    median = float(torch.tensor([r['reduction'] for r in results if not r['canonical']]).quantile(.5))
    dump(args.out/'SUMMARY.json', {'classification': 'DEVELOPMENTAL', 'valid': valid,
        'median_variance_reduction': median, 'route': 'INVALID_GRADIENT_APPARATUS' if not valid else
        ('LEARNING_PILOT_MAY_BE_DESIGNED' if median >= .2 else 'STOP_SMALL_VARIANCE_EFFECT'),
        'results': results, 'seconds': time.monotonic()-started})
    dump(args.out/'MANIFEST.json', {p.name: sha(p) for p in args.out.iterdir() if p.is_file()})
    print('ACTION_GRADIENT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
