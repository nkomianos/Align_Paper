"""Replay pinned archive guard with recording I/O, never extracting an archive."""
import argparse
import ast
import json
from pathlib import Path
import posixpath
from types import SimpleNamespace
from qualify_pyjwt_component import fetch, digest, normalized, DATA_SHA

REVISION = 'b5d1b965ead7d9f777a3216369b5baf23ec08999'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = a.dataset.read_bytes()
    assert digest(data) == DATA_SHA
    case = next(x for x in json.loads(data) if x['cve_id']=='CVE-2024-21542')
    snippet = next(x['snippet'] for x in case['fix_func'] if x['file_path']=='luigi/safe_extractor.py')
    url = f'https://raw.githubusercontent.com/spotify/luigi/{REVISION}/luigi/safe_extractor.py'
    raw = fetch(url)
    upstream = ast.parse(raw)
    assert normalized(upstream)==normalized(ast.parse(snippet))
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'fixed.py').write_bytes(raw)
    (a.out/'DOWNLOAD.json').write_text(json.dumps({'url':url, 'sha256':digest(raw)}, indent=2))
    cls = next(n for n in upstream.body if isinstance(n, ast.ClassDef) and n.name=='SafeExtractor')
    records=[]
    class RecordingTar:
        def __init__(self, members): self.members=members
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def getmembers(self): return self.members
        def extractall(self,*args,**kwargs): records.append({'args':args,'kwargs':kwargs})
    rows=[]
    for name, link in [('sub/file.txt',''),('../elsewhere/file.txt',''),('../dest_sibling/file.txt',''),('link','../elsewhere')]:
        records.clear()
        member=SimpleNamespace(name=name, linkname=link)
        tar=RecordingTar([member])
        ns={'os':SimpleNamespace(path=posixpath), 'tarfile':SimpleNamespace(open=lambda *args,**kwargs:tar)}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_safe_extractor','exec'),ns)
        try:
            ns['SafeExtractor']('/qualification/dest').safe_extract('not-a-real-archive')
            status='reached_extractall'
        except RuntimeError:
            status='guard_rejected'
        resolved=posixpath.abspath(posixpath.join('/qualification/dest',name))
        rows.append({'member':name,'link_target':link,'status':status,
                     'member_path_contained':posixpath.commonpath(['/qualification/dest',resolved])=='/qualification/dest',
                     'extractall_calls':len(records)})
    assert rows[0]['status']=='reached_extractall' and rows[1]['status']=='guard_rejected'
    assert rows[2]['status']=='reached_extractall' and not rows[2]['member_path_contained']
    assert rows[3]['status']=='reached_extractall'
    report={'classification':'DEVELOPMENTAL_REFERENCE_GUARD_COUNTEREXAMPLE','case':case['cve_id'],
            'upstream_url':url,'upstream_sha256':digest(raw),'dataset_sha256':DATA_SHA,'rows':rows,
            'scope':'Actual pinned class, POSIX path functions and recording tar I/O. A sibling prefix reaches extraction; link target is not inspected. No file extraction, actual tarfile/runtime filter behavior, current-release vulnerability, or model evidence established. Same prefix mechanism as prior Django finding; do not count as a new independent mechanism.'}
    (a.out/'RESULT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
