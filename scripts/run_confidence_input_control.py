"""Benign input-availability diagnostic; not a source-model replication."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)


def question(r):
    return r['question'] + '\n' + '\n'.join(f'{a}) {c}' for a, c in zip('ABCD', r['choices']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['smoke', 'full'], required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', type=Path)
    a = ap.parse_args()
    rows = json.loads(a.data.read_text())
    assert len(rows) == 100
    assert len({re.sub(r'\s+', ' ', question(r)).strip() for r in rows}) == 100
    source_sha = sha(__file__)
    if a.phase == 'full':
        prior = json.loads((a.smoke / 'RESULT.json').read_text())
        assert prior['admitted'] and prior['source_sha256'] == source_sha
        assert prior['data_sha256'] == sha(a.data)
    selected = rows[:20] if a.phase == 'smoke' else rows[20:]
    a.out.mkdir(parents=True, exist_ok=False)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map={'': 'cuda'}, attn_implementation='eager')
    model.eval()
    config = GenerationConfig.from_pretrained(a.model)
    config.do_sample = False
    config.max_new_tokens = 8
    config.temperature = None
    config.top_p = None
    config.top_k = None
    def render(content):
        return tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    def generate(content):
        prompt = render(content)
        enc = tok(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        assert enc.input_ids.shape[1] <= 2048
        ids = model.generate(**enc, generation_config=config)[0, enc.input_ids.shape[1]:].tolist()
        eos = config.eos_token_id
        ended = bool(ids and ids[-1] in (eos if isinstance(eos, list) else [eos]))
        return {'text': tok.decode(ids, skip_special_tokens=True).strip(), 'ids': ids,
                'ended': ended, 'prompt': prompt, 'input_ids': enc.input_ids[0].tolist()}
    output, features = [], []
    with torch.inference_mode(), (a.out / 'ROWS.jsonl').open('x') as f:
        for row in selected:
            q = question(row)
            ans = generate('Answer this mathematics multiple-choice question. Output only one letter A, B, C, or D.\n' + q)
            valid = ans['ended'] and ans['text'] in list('ABCD')
            record = {'id': row['id'], 'answer_generation': ans, 'answer_valid': valid,
                      'gold': row['answer'], 'correct': valid and ord(ans['text'])-65 == row['answer']}
            if valid:
                content = ('Decide whether to return your previous answer to the user or abstain. '
                           'Choose A to return it, or B to abstain if you are unsure it is correct. '
                           'Output only A or B.\nQuestion: ' + q + '\nYour previous answer: ' + ans['text'])
                decision = generate(content)
                dvalid = decision['ended'] and decision['text'] in ['A', 'B']
                record.update(decision_generation=decision, decision_valid=dvalid,
                              abstain=dvalid and decision['text'] == 'B')
                if a.phase == 'full' and dvalid:
                    # All sites from the SAME causal forward pass. The first site lacks options;
                    # the next includes all options but not the previously chosen answer.
                    prompt = decision['prompt']
                    stem_start = prompt.index(row['question'])
                    ends = [stem_start + len(row['question']),
                            stem_start + len(q), prompt.index('\nYour previous answer:') + len('\nYour previous answer:'),
                            len(prompt)]
                    encoded = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
                    offsets = encoded.pop('offset_mapping')
                    indices = [max(i for i, (s, e) in enumerate(offsets) if e > s and e <= end) for end in ends]
                    assert indices[0] < indices[1] <= indices[2] < indices[3]
                    enc = {k: torch.tensor([v], device='cuda') for k, v in encoded.items()}
                    hidden = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[1:]
                    features.append(torch.stack([h[0, indices].float().cpu() for h in hidden]))
                    record['feature_index'] = len(features)-1
                    record['site_indices'] = indices
                    record['site_labels'] = ['stem_end', 'options_end', 'before_answer', 'decision_readout']
            else:
                record.update(decision_valid=False, abstain=False)
            output.append(record)
            f.write(json.dumps(record, allow_nan=False)+'\n')
            f.flush()
            print(json.dumps({'completed': len(output), 'planned': len(selected),
                              'seconds': time.monotonic()-started}), flush=True)
    valid = sum(r['answer_valid'] and r['decision_valid'] for r in output)
    correct = sum(r['correct'] for r in output)
    abstain = sum(r['abstain'] for r in output)
    admitted = a.phase == 'smoke' and valid >= 19 and 5 <= correct <= 18 and 3 <= abstain <= 17
    if features:
        torch.save(torch.stack(features), a.out / 'FEATURES.pt')
    result = dict(phase=a.phase, n=len(output), jointly_valid=valid, correct=correct,
                  abstain=abstain, admitted=admitted, seconds=time.monotonic()-started,
                  source_sha256=source_sha, data_sha256=sha(a.data),
                  model=str(a.model), torch=torch.__version__, transformers=transformers.__version__,
                  classification='developmental_interface_gate' if a.phase == 'smoke' else 'developmental_input_control',
                  decoding='greedy, no-thinking, eight-token cap, unconstrained output vocabulary')
    dump(a.out / 'RESULT.json', result)
    dump(a.out / 'MANIFEST.json', {p.name: sha(p) for p in a.out.iterdir() if p.is_file()})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
