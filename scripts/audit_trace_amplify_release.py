"""Audit released trajectory identity and split overlap without neural scoring."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('fresh report required')
    files=[];groups=defaultdict(lambda:defaultdict(set))
    with zipfile.ZipFile(a.archive) as z:
        for name in z.namelist():
            if not name.endswith('.jsonl'):continue
            labels=Counter();types=Counter();prompts=defaultdict(set);responses=set();lengths=[];count=0
            label_response=defaultdict(set);missing=[]
            with z.open(name) as f:
                for line in f:
                    r=json.loads(line);count+=1
                    if not isinstance(r['prompt'],str) or not isinstance(r['response'],str):
                        raise ValueError('unexpected prompt/response type')
                    if type(r['hacking_label']) is not int or r['hacking_label'] not in (0,1):
                        raise ValueError('bad label')
                    ph=hashlib.sha256(r['prompt'].encode()).hexdigest()
                    rh=hashlib.sha256(r['response'].encode()).hexdigest()
                    labels[r['hacking_label']]+=1;types[r['hacking_type']]+=1
                    prompts[ph].add(r['hacking_label']);responses.add(rh);label_response[rh].add(r['hacking_label'])
                    lengths.append(len(r['prompt'])+len(r['response']))
                    if not r['prompt'] or not r['response']:missing.append(count)
            path=Path(name);split=path.name.split('_')[0]
            if split in ('train','test'):groups[str(path.parent)][split].update(prompts)
            files.append({'file':name,'rows':count,'labels':dict(labels),'types':dict(types),
                'unique_prompts':len(prompts),'prompts_with_both_labels':sum(len(v)==2 for v in prompts.values()),
                'unique_responses':len(responses),'responses_with_conflicting_labels':sum(len(v)==2 for v in label_response.values()),
                'min_characters':min(lengths),'max_characters':max(lengths),'empty_rows':missing})
    overlap=[{'group':g,'train_prompts':len(v['train']),'test_prompts':len(v['test']),
              'exact_prompt_overlap':len(v['train'] & v['test'])} for g,v in groups.items()]
    with a.archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'archive_sha256':sha,'files':files,'train_test_overlap':overlap,
        'scope':'Source identity audit only. Hash overlap does not replay labels or establish semantic task independence.',
        'neural_scores_computed':False,'paper_green_light':False}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps({'files':len(files),'file_rows_including_duplicates':sum(f['rows'] for f in files),'train_test_overlap':overlap},indent=2))


if __name__=='__main__':main()
