"""Fetch official benign script dataset and inventory it without task selection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    url='https://keisuke-sakaguchi.github.io/proscript/proscript_v1a.zip'
    with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
    (a.out/'proscript_v1a.zip').write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        inventory=[{'name':i.filename,'bytes':i.file_size,'crc32':i.CRC} for i in archive.infolist()]
        assert sum(i['bytes'] for i in inventory)<100_000_000
        metadata={}
        for name in archive.namelist():
            if Path(name).name.lower() in {'readme','readme.md','readme.txt','license','license.txt'}:
                metadata[name]=archive.read(name).decode('utf-8')
    report={'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'inventory':inventory,
            'metadata':metadata,'scope':'Official version-named URL downloaded and content hashed, not an immutable URL attestation. No train/dev/test selection or neural evaluation.'}
    (a.out/'DOWNLOAD.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
