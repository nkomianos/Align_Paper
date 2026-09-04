"""Exercise native upstream templates on CPU without model loading/generation."""
import argparse,hashlib,inspect,json,os,subprocess,sys
from pathlib import Path
os.environ["USE_TF"]="0"
os.environ["HF_HUB_OFFLINE"]="1"

def main():
    p=argparse.ArgumentParser();p.add_argument("assets",type=Path);p.add_argument("upstream",type=Path)
    p.add_argument("root",type=Path);p.add_argument("--policy",required=True);p.add_argument("--simulator",required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(a.upstream))
    import jinja2,torch,transformers
    from transformers import AutoTokenizer,GenerationConfig
    from transformers.generation import GenerationMixin
    from online_sdpo_updater import OnlineSDPOUpdater
    from online_sdpo_updater_config import OnlineSDPOConfig
    from auxiliary.user_simulator import StyleUserSimulator
    from auxiliary.style_judge import StyleJudge
    cases=json.loads((a.assets/"calibration.json").read_text())
    results=[]
    class Stub:
        def eval(self):return self
    policy=AutoTokenizer.from_pretrained(a.policy,local_files_only=True)
    simtok=AutoTokenizer.from_pretrained(a.simulator,local_files_only=True)
    for tok in (policy,simtok):
        if tok.pad_token is None:tok.pad_token=tok.eos_token
    simulator=StyleUserSimulator(Stub(),simtok,torch.device("cpu"),"concise_casual_beginner")
    judge=StyleJudge(Stub(),simtok,torch.device("cpu"),"concise_casual_beginner")
    updater=object.__new__(OnlineSDPOUpdater);updater.config=OnlineSDPOConfig()
    explicit="The user prefers concise, casual, and beginner-friendly responses: short, clear answers rather than long, formal, or technically dense answers. Honor that preference while accurately summarizing the supplied text."
    for row in cases:
        messages=[dict(role="user",content=row["prompt"])]
        prompts={"policy":policy.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),
            "explicit":policy.apply_chat_template([dict(role="system",content=explicit)]+messages,tokenize=False,add_generation_prompt=True,enable_thinking=False),
            "hindsight":policy.apply_chat_template(updater._build_hindsight_messages(messages,"Please make the summary concise, casual, and beginner-friendly."),tokenize=False,add_generation_prompt=True,enable_thinking=False),
            "simulator":simulator._build_prompt_text(row["prompt"],"TL;DR: Template-preflight placeholder, not a generated answer."),
            "judge":judge._build_prompt_text(row["prompt"],"TL;DR: Placeholder A.","TL;DR: Placeholder B.")}
        encoded={k:(simtok if k in ("simulator","judge") else policy)(v,add_special_tokens=False,return_tensors="pt")["input_ids"][0].tolist() for k,v in prompts.items()}
        results.append(dict(id=row["id"],rendered=prompts,token_ids=encoded,lengths={k:len(v) for k,v in encoded.items()}))
    generation=GenerationConfig(max_new_tokens=2048,do_sample=True,temperature=1.,top_p=1.,eos_token_id=policy.eos_token_id,pad_token_id=policy.pad_token_id)
    generation.validate()
    report=dict(status="NATIVE_TEMPLATE_PREFLIGHT_PASSED",case_count=len(results),model_loads=0,generation_calls=0,
        jinja2=jinja2.__version__,transformers=transformers.__version__,torch=torch.__version__,
        generation_config=generation.to_dict(),generate_signature=str(inspect.signature(GenerationMixin.generate)),
        results=results,scope="Native template rendering/tokenization plus config validation only; no CUDA kernels or model generation")
    (a.root/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (a.root/"package_freeze.txt").write_bytes(subprocess.check_output([sys.executable,"-m","pip","freeze"]))
    (a.root/"preflight_source.py").write_bytes(Path(__file__).read_bytes())
    (a.root/"MANIFEST.json").write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.root.iterdir() if f.is_file()},indent=2))
    print(json.dumps({k:report[k] for k in ("status","case_count","model_loads","generation_calls","jinja2")}))
if __name__=="__main__":main()
