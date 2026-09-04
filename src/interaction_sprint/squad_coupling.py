"""Public SQuAD DEV preparation and standard answer scoring for coupling tests."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
from .byte_clock_coupling import digest

SOURCE_SHA='95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9'
SOURCE_URL='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
SALT='byte-coupling-squad-dev-20260904-v1'


def rank(s):
    return hashlib.sha256((SALT+':'+s).encode()).hexdigest()


def normalize(s):
    # SQuAD v1.1's published order: lowercase, remove ASCII punctuation,
    # remove English articles, collapse whitespace. No answer extraction.
    s=''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def score(prediction, answers):
    p=normalize(prediction); pwords=p.split(); em=0.; best=0.
    for answer in answers:
        a=normalize(answer); awords=a.split()
        em=max(em,float(p==a))
        overlap=sum((Counter(pwords)&Counter(awords)).values())
        f1=2*overlap/(len(pwords)+len(awords)) if overlap else 0.
        best=max(best,f1)
    return {'em':em,'f1':best}


def prepare(source,root,n=8):
    assert digest(source)==SOURCE_SHA
    data=json.loads(source.read_text(encoding='utf-8')); eligible=[]
    for article in data['data']:
        choices=[]
        for para in article['paragraphs']:
            context=para['context']
            if not 60<=len(context.split())<=180: continue
            for q in para['qas']:
                answers=q['answers']
                if len(q['question'].split())>30 or not answers: continue
                if any(len(a['text'].split())>8 or len(a['text'])>60 for a in answers): continue
                assert all(context[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in answers)
                choices.append({'id':q['id'],'title':article['title'],'context':context,'question':q['question'],
                                'answers':list(dict.fromkeys(a['text'] for a in answers))})
        if choices: eligible.append(min(choices,key=lambda q:rank(q['id'])))
    chosen=sorted(eligible,key=lambda q:rank(q['title']))[:n]
    assert len(chosen)==n and len(set(c['title'] for c in chosen))==n
    root.mkdir(parents=True,exist_ok=False)
    key={c['id']:c['answers'] for c in chosen}
    public=[{k:v for k,v in c.items() if k!='answers'} for c in chosen]
    (root/'cases.json').write_text(json.dumps(public,indent=2),encoding='utf-8')
    (root/'answer_key.json').write_text(json.dumps(key,indent=2),encoding='utf-8')
    meta={'source_url':SOURCE_URL,'source_sha':SOURCE_SHA,'source_path':str(source),'license':'CC BY-SA 4.0',
          'salt':SALT,'eligible_articles':len(eligible),'chosen_articles':n,'max_answer_words':8,'max_answer_chars':60,
          'context_words':[60,180],'max_question_words':30,'source_sha_code':digest(__file__),
          'scope':'Small public validation DEV, one item/article; not private or uncontaminated TEST'}
    (root/'preparation.json').write_text(json.dumps(meta,indent=2))
    (root/'MANIFEST.json').write_text(json.dumps({p.name:digest(p) for p in root.iterdir()},indent=2))
    return meta


def prompt(c):
    return ('Answer the question using the passage. Return only the shortest answer span, '
            'copied exactly from the passage. Do not explain.\n\nPassage:\n'+c['context']+
            '\n\nQuestion: '+c['question'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,default=Path('artifacts/byte_coupling_squad_source_v1/dev-v1.1.json'))
    a=p.parse_args();print(json.dumps(prepare(a.source,a.root),indent=2))
