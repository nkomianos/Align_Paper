"""Post-pilot developmental robustness audit; not an untouched confirmatory test.

Fresh deterministic histories and two renderers. Tests simple baselines before
any training claim. Earlier frozen cases and thresholds are not modified.
"""
import random
import numpy as np

from .fixtures import reduce_ops, SURFACES

SEED = 9040349
ARMS = ('canonical', 'history', 'padded', 'explicit_update', 'reminder', 'counterfactual')


def render(op, style):
    kind, key, value = op
    if kind == 'clear':
        return (f'Remove the assignment for {key}; it now has no assigned value.' if style == 0
                else f'Cancel the current requirement for {key}. Leave this field unspecified.')
    return (f'The value of {key} is now {value}; supersede any earlier assignment.' if style == 0
            else f'Update requirement: {key} = {value}. Replace the old requirement for this field.')


def build():
    rng = random.Random(SEED)
    cases, answers = [], {}
    for surface, (stem, *values) in SURFACES.items():
        for relation in ('clear', 'overwrite'):
            for depth in (20, 60, 100):
                for repeat in range(6):
                    style = repeat % 2
                    slot = f'{stem}_{rng.randrange(10000, 99999)}'
                    old, new = rng.sample(values, 2)
                    second = depth // 2 if repeat < 3 else depth - 2
                    edits = []
                    for index in range(depth):
                        if index == 1:
                            edits.append(('set', slot, old))
                        elif index == second:
                            edits.append(('clear', slot, None) if relation == 'clear' else ('set', slot, new))
                        else:
                            edits.append(('set', f'auxiliary_{rng.randrange(8)}', f'value_{rng.randrange(1000)}'))
                    state = reduce_ops({}, edits)
                    cf = list(edits); cf[second] = ('set', slot, old)
                    cf_state = reduce_ops({}, cf)
                    target = state.get(slot, 'UNSET')
                    assert target != cf_state.get(slot, 'UNSET')
                    options = [*values, 'UNSET']; rng.shuffle(options)
                    choices = list('ABCD')
                    question = f'What is the final value of {slot}?\n' + '\n'.join(
                        f'{letter}: {value}' for letter, value in zip(choices, options))
                    question += '\nReturn exactly one letter A, B, C or D.'
                    system = ('Maintain independent named fields. Updates replace earlier values. Removing or cancelling '
                              'an assignment leaves UNSET; it does not restore a previous value. The literal value '
                              'UNSET also means the field has no assignment. Initially every field is UNSET.')
                    pair = f'{surface}-{relation}-{depth}-{repeat}'
                    for arm in ARMS:
                        messages = [{'role': 'system', 'content': system}]
                        selected = cf_state if arm == 'counterfactual' else state
                        if arm == 'canonical':
                            messages.append({'role': 'user', 'content': f'Authoritative current field state: {selected!r}\n{question}'})
                        else:
                            if arm == 'padded':
                                messages[0]['content'] += f' Initial authoritative state instead: {state!r}'
                                operations = [('set', f'padding_{i % 8}', f'entry_{i}') for i in range(depth)]
                            else:
                                operations = cf if arm == 'counterfactual' else edits
                            for i, op in enumerate(operations):
                                if arm == 'explicit_update' and i == second:
                                    command = f'Set {slot} to {target}. Replace every earlier assignment for this field.'
                                else:
                                    command = render(op, style)
                                messages.extend([{'role':'user','content':command}, {'role':'assistant','content':'Recorded.'}])
                            query = question
                            if arm == 'reminder':
                                query += ('\nUse only the final active assignments, not an obsolete value. '
                                          'A field with a cancelled or removed assignment must be UNSET.')
                            messages.append({'role':'user','content':query})
                        cid = pair + '/' + arm
                        value = selected.get(slot, 'UNSET')
                        cases.append({'case_id':cid,'pair_id':pair,'study':'undo_audit','surface':surface,
                            'relation':relation,'depth':depth,'repeat':repeat,'style':style,
                            'update_position':second,'arm':arm,'smoke':depth==20 and repeat==0,
                            'messages':messages,'choices':choices})
                        answers[cid] = {'answer':choices[options.index(value)],'stale_answer':choices[options.index(old)],
                            'state':selected,'slot':slot,'operations':cf if arm=='counterfactual' else edits,
                            'option_values':options}
    return cases, answers


def analyze(cases, key, records, mode, common):
    index = {r['case_id']:r for r in records}
    def acc(rows):
        return float(np.mean([index[c['case_id']]['predicted'] == key[c['case_id']]['answer'] for c in rows]))
    cells = []
    for relation in sorted({c['relation'] for c in cases}):
        for depth in sorted({c['depth'] for c in cases}):
            for style in sorted({c['style'] for c in cases if c['relation']==relation and c['depth']==depth}):
                group = [c for c in cases if (c['relation'],c['depth'],c['style'])==(relation,depth,style)]
                accuracy = {arm: acc([c for c in group if c['arm']==arm]) for arm in ARMS}
                cells.append({'relation':relation,'depth':depth,'style':style,'pairs':len(group)//len(ARMS),'accuracy':accuracy})
    long = [c for c in cases if c['depth'] >= 60]
    long_acc = {arm:acc([c for c in long if c['arm']==arm]) for arm in ARMS} if long else {}
    valid = common['mean_choice_mass']>=.5 and all(
        c['accuracy']['canonical']>=.9 and c['accuracy']['padded']>=.9 and c['accuracy']['counterfactual']>=.9 for c in cells)
    signal = bool(long) and long_acc['padded']-long_acc['history']>=.1
    cheap_fix = bool(long) and max(long_acc['reminder'],long_acc['explicit_update']) >= long_acc['padded']-.05
    decision = ('SMOKE_ONLY' if mode=='smoke' else 'INVALID_ROBUSTNESS_ASSAY' if not valid else
                'NO_FRESH_HISTORY_SIGNAL_PARK' if not signal else
                'SIMPLE_BASELINE_SUFFICIENT_DO_NOT_CLAIM_TRAINING_NEEDED' if cheap_fix else
                'ROBUST_SIGNAL_QUEUE_INDEPENDENT_FAMILY_AND_TRAINING_DESIGN')
    return {**common,'decision':decision,'cells':cells,'long_history_accuracy':long_acc,
            'warning':'Post-pilot developmental audit. No independent model family, training, or population-generalization claim.'}
