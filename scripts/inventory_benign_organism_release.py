"""Read release metadata for the Italian-food preference family only."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Align-Paper-benign-release-audit'})
    with urllib.request.urlopen(request,timeout=60) as response:return response.read()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);receipts={}
    def save(name,url):
        raw=get(url);(a.out/name).write_bytes(raw)
        receipts[name]={'url':url,'sha256':hashlib.sha256(raw).hexdigest()}
        return json.loads(raw)
    models=save('model_list.json','https://huggingface.co/api/models?author=model-organisms-for-real&search=italian&limit=100&full=true')
    assert len(models)<100, 'listing may be truncated; paginate before interpreting'
    selected=[r for r in models if 'italian' in r['id'].lower()]
    records=[]
    for index,row in enumerate(selected):
        info=save(f'model_{index:02d}.json','https://huggingface.co/api/models/'+row['id'])
        records.append({'id':info['id'],'revision':info.get('sha'),'gated':info.get('gated'),
                        'files':[r['rfilename'] for r in info.get('siblings',[])],
                        'pipeline_tag':info.get('pipeline_tag'),'tags':info.get('tags',[])})
    repo=save('repository.json','https://api.github.com/repos/model-organisms-for-real/model-organism-lottery')
    revision=save('repository_commit.json','https://api.github.com/repos/model-organisms-for-real/model-organism-lottery/commits/'+repo['default_branch'])['sha']
    tree=save('repository_tree.json','https://api.github.com/repos/model-organisms-for-real/model-organism-lottery/git/trees/'+revision+'?recursive=1')
    report={'classification':'BENIGN_RELEASE_METADATA_INVENTORY','selected_family':'Italian food preference',
            'listed_models':len(models),'italian_models':records,'repository_revision':revision,
            'tree_truncated':tree.get('truncated'),'repository_entries':len(tree.get('tree',[])),
            'scope':'Public metadata only. No model weights, training samples, evaluation prompts or non-benign families downloaded. Availability is not a qualified assay or causal transfer result.'}
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipts,indent=2))
    (a.out/'REPORT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({**{k:v for k,v in report.items() if k!='italian_models'},'italian_models':[{k:r[k] for k in ['id','revision','gated']} for r in records]},indent=2))


if __name__=='__main__':main()
