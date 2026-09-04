"""Offline pre-execution views. Trace strings are data, never instructions."""
import copy


def preexecution_view(record, step_id, tool_call_id):
    steps=record['steps']
    matches=[i for i,s in enumerate(steps) if s['step_id']==step_id]
    if len(matches)!=1: raise ValueError('Nonunique or missing step')
    index=matches[0]; step=steps[index]
    calls=[c for c in (step.get('tool_calls') or []) if c['tool_call_id']==tool_call_id]
    if len(calls)!=1: raise ValueError('Nonunique or missing call')
    # Preserve earlier recorded observations, never current-step results or
    # later steps. Sibling calls share a step; their execution order is unknown.
    history=[]
    for previous in steps[:index]:
        if previous['source']=='system': continue
        history.append({k:copy.deepcopy(previous.get(k)) for k in
                        ('source','message','tool_calls','observation')})
    # Tool-call extras are not model inputs: labels/provenance can reside there.
    for item in history:
        item['tool_calls']=[{k:copy.deepcopy(c[k]) for k in ('function_name','arguments')}
                            for c in (item['tool_calls'] or [])]
    proposed={k:copy.deepcopy(calls[0][k]) for k in ('function_name','arguments')}
    return dict(history=history,proposed_call=proposed,
                policy_note='Original platform system prompt omitted; no replacement judge policy supplied.')
