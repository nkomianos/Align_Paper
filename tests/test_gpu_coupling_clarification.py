import pytest
import importlib.util
import json
from pathlib import Path
from interaction_sprint.gpu_coupling_clarification_prepare import complement


def test_full_complement_preserves_order_excludes_all_old_articles():
    cases=[{'id':str(i),'title':'article'+str(i),'question':'q'} for i in range(48)]
    old=cases[:8]
    assert complement(cases,old)==cases[8:]
    changed=[dict(x) for x in old];changed[0]['question']='different'
    with pytest.raises(AssertionError):complement(cases,changed)
    with pytest.raises(AssertionError):complement(cases[:-1],old)


def test_clarification_scorer_differs_only_in_scope_label():
    scripts=Path(__file__).parents[1]/'scripts'
    original=(scripts/'analyze_squad_coupling.py').read_text().strip()
    new=(scripts/'analyze_gpu_coupling_clarification.py').read_text().strip()
    assert original.replace('Eight public DEV questions, 16 seeds per question; descriptive pilot only; no automatic expansion',
                            'Forty article-disjoint clarification questions, 32 new seeds; post-DEV protocol amendment, not paper expansion')==new


def test_combined_summary_requires_both_registered_pairs(tmp_path):
    path=Path(__file__).parents[1]/'scripts/summarize_coupling_clarification_pairs.py'
    spec=importlib.util.spec_from_file_location('combined_for_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def report(names):
        policies={p:{'by_model':{n:{'n':1280} for n in names},'f1':{'covariance':0.,'variance_a':.2,'variance_b':.3}}
                  for p in ['independent','token_clock','byte_clock','byte_hierarchical']}
        return {'runtime':{'outputs':10240},'per_case':[{'id':str(i)} for i in range(40)],
                'policies':policies,'comparisons':{'f1':{'hierarchical vs '+p:{'ratio':1.} for p in ['independent','token_clock','byte_clock']}},'manifest_sha':'x'}
    small=tmp_path/'small.json';large=tmp_path/'large.json'
    small.write_text(json.dumps(report(['qwen3_06b','smollm2_360m'])))
    large.write_text(json.dumps(report(['qwen3_4b','smollm2_17b'])))
    module.summarize(small,large,tmp_path/'valid.json')
    with pytest.raises(AssertionError):module.summarize(small,small,tmp_path/'invalid.json')
