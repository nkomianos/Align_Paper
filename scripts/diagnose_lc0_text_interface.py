"""Bounded post-smoke text capability diagnosis, not latent-channel validation.

Eight existing DEV cases x three arms. No labels are read by the runner.
Never changes or overwrites previous scientific inputs, thresholds or evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from latent_contract.channel import MODEL, REVISION, plan, receiver_text, sender_text


ARMS = ("ids_no_thinking", "embeds_no_thinking", "ids_thinking")


def final_answer(completion, thinking):
    if thinking:
        if "</think>" not in completion:
            return None
        completion = completion.rsplit("</think>", 1)[1]
    answer = completion.strip()
    return answer if answer in ("A", "B", "C", "D") else None


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def run(prepared, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = plan(json.loads((prepared / "cases.json").read_text()), "smoke")
    assert len(cases) == 8
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "runner.py")
    write(output / "cases.json", cases)
    write(output / "plan.json", {"scope": "post_hoc_text_interface_diagnostic_not_gate",
        "cases": [c["case_id"] for c in cases], "arms": ARMS, "forwards": 24,
        "model": MODEL, "revision": REVISION,
        "max_new_tokens": {"ids_no_thinking": 8, "embeds_no_thinking": 8, "ids_thinking": 256},
        "prepared_manifest_sha256": hashlib.sha256((prepared / "MANIFEST.json").read_bytes()).hexdigest(),
        "interpretation": "test embedding API equivalence and reasoning-budget sensitivity; no latent result"})
    try:
        assert torch.cuda.is_available()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}).eval()
        embedding = model.get_input_embeddings()
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "resolved_commit": model.config._commit_hash})
        def ids(text):
            return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        def encode(case, thinking):
            marker = "LC0_CHANNEL_SLOT_81ca2"
            chat = tokenizer.apply_chat_template([
                {"role": "system", "content": "Answer the relation query using supplied information. Output one answer letter only."},
                {"role": "user", "content": marker + receiver_text(case)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
            assert chat.count(marker) == 1
            before, after = chat.split(marker)
            return torch.cat((ids(before), ids(sender_text(case)), ids(after)), dim=1)
        with (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for case in cases:
                for arm in ARMS:
                    thinking = arm == "ids_thinking"
                    tokens = encode(case, thinking)
                    mask = torch.ones_like(tokens)
                    limit = 256 if thinking else 8
                    start = time.monotonic()
                    with torch.inference_mode():
                        inputs = ({"inputs_embeds": embedding(tokens)} if arm.startswith("embeds") else {"input_ids": tokens})
                        outputs = model.generate(**inputs, attention_mask=mask, do_sample=False,
                            max_new_tokens=limit, pad_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    generated = outputs[0] if arm.startswith("embeds") else outputs[0, tokens.shape[1]:]
                    completion = tokenizer.decode(generated, skip_special_tokens=True)
                    row = {"case_id": case["case_id"], "arm": arm, "completion": completion,
                           "input_ids": tokens[0].tolist(), "generated_ids": generated.tolist(),
                           "final_answer": final_answer(completion, thinking),
                           "hit_token_limit": len(generated) == limit, "seconds": time.monotonic()-start}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        write(output / "COMPLETE.json", {"records": 24})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.output)
