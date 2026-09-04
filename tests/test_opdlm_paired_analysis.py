import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('paired_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_opdlm_onpolicy.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

def test_discordance_preserves_direction():
    a=mod.paired([1,1,1,0],[0,0,1,1])
    assert a['a_only']==2 and a['b_only']==1 and a['difference_pp']==25
    assert a['unadjusted_exact_discordance_p']==1

def test_identical_is_not_evidence_of_equivalence():
    a=mod.paired([1,0,1],[1,0,1])
    assert a['a_only']==a['b_only']==0 and a['unadjusted_exact_discordance_p']==1

def test_last_number_span_not_echoed_operands():
    assert mod.last_numeric_span('3 + 4 = **7**.')==(10,11)
    assert mod.last_numeric_span('No answer') is None
    assert mod.last_numeric_span('The answer is 15.2') is None
