"""Export exact pinned upstream Python source for the single-profile wrapper."""
import argparse,hashlib,json,subprocess
from pathlib import Path
COMMIT="3b17d2a67bd2565b9fbda495fd16a485406aa954"
FILES=("online_sdpo_updater.py","online_sdpo_updater_config.py","auxiliary/user_simulator.py","auxiliary/style_judge.py")
def main():
    p=argparse.ArgumentParser();p.add_argument("repo",type=Path);p.add_argument("out",type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);hashes={}
    for name in FILES:
        data=subprocess.check_output(["git","show",f"{COMMIT}:{name}"],cwd=a.repo)
        target=a.out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    (a.out/"PINNED_SOURCE.json").write_text(json.dumps(dict(commit=COMMIT,sha256=hashes),indent=2))
if __name__=="__main__":main()
