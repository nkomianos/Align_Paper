from interaction_sprint.squad_coupling import normalize,score,prompt
from interaction_sprint.squad_coupling_runner import sample_seed


def test_standard_squad_normalization():
    assert normalize('  The U.S.   Army! ')== 'us army'
    assert score('The Denver Broncos',['Denver Broncos','Broncos'])=={'em':1.,'f1':1.}
    assert score('Denver',['Denver Broncos'])=={'em':0.,'f1':2/3}
    assert score('',['Denver'])=={'em':0.,'f1':0.}
    assert score('Answer: Denver',['Denver'])=={'em':0.,'f1':2/3}


def test_sampler_seed_alignment():
    assert sample_seed(1,'x','byte_clock','a')==sample_seed(1,'x','token_clock','b')
    assert sample_seed(1,'x','independent','a')!=sample_seed(1,'x','independent','b')
    assert sample_seed(1,'x','byte_clock','a')!=sample_seed(1,'y','byte_clock','a')


def test_prompt_cannot_include_gold_field():
    c={'context':'The city is Paris.','question':'Which city?','answers':['secret-marker']}
    assert 'secret-marker' not in prompt(c) and 'The city is Paris.' in prompt(c)
