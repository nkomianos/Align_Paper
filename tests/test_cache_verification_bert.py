import ast
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from interaction_sprint.cache_verification_bert import Instrument, summarize, ARMS, sha


def tiny():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    config = BertConfig(vocab_size=128, hidden_size=32, num_hidden_layers=3,
                        num_attention_heads=4, intermediate_size=48,
                        hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
    config._attn_implementation = "eager"
    return BertForMaskedLM(config).eval()


def test_native_equivalence_and_restoration():
    model = tiny()
    ids = torch.tensor([[4, 5, 6, 7, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    with torch.no_grad():
        native = model(ids, attention_mask=mask).logits
        with Instrument(model):
            got = model(ids, attention_mask=mask).logits
        restored = model(ids, attention_mask=mask).logits
    torch.testing.assert_close(got, native, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(native, restored, atol=0, rtol=0)


def test_last_layer_is_clean_but_first_layer_can_feedback():
    model = tiny()
    ids = torch.tensor([[4, 5, 6, 7]])
    draft = ids.clone(); draft[0, 1] = 99
    with torch.no_grad():
        native = model(ids).logits[0, 1]
        with Instrument(model) as inst:
            inst.position, inst.mode = 1, "collect"
            model(draft)
            inst.cache = inst.saved
            inst.mode = "last"
            last = model(ids).logits[0, 1]
            inst.mode = "first"
            first = model(ids).logits[0, 1]
    torch.testing.assert_close(last, native, atol=1e-6, rtol=1e-5)
    assert (first-native).abs().max() > 1e-7


def test_summary_refuses_broken_native_control():
    logits = np.zeros((1, len(ARMS), 3))
    logits[:, :, 1] = 1
    rows = [{"id": "a", "gold_id": 1, "wrong_id": 2}]
    config = {"cases": [{"id": "a"}], "equivalence_tolerance": 1e-4,
              "minimum_clean_correct_for_interpretation": 1}
    assert summarize(logits, rows, config)["clean_correct"] == 1
    logits[0, 1, 2] = 4
    with pytest.raises(AssertionError):
        summarize(logits, rows, config)


def test_reviewed_upstream_correction_matches_direct_rows():
    # Execute only the previously inspected standalone arithmetic function,
    # never import the repository's model, loaders, launchers, or evaluation.
    path = Path("artifacts/cover_source_audit_20260904/llada_ins/code/dllm_eval/models/cover/modeling_llada_kv_cover.py")
    if not path.exists():
        pytest.skip("Optional pinned upstream checkout absent")
    assert sha(path) == "b9bdff4b4fd93136e4046f4bef5e92e009afee9ea18f51af2685b3daea9ab4ca"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_post_hoc_diagonal_correction_exact")
    namespace = {"torch": torch, "math": math, "Optional": Optional, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    torch.manual_seed(241)
    q, k, v, ck, cv = [torch.randn(1, 4, 6, 8, dtype=torch.float64) for _ in range(5)]
    pos = torch.tensor([1, 3])
    out = (q @ k.transpose(-1, -2) / math.sqrt(8)).softmax(-1) @ v
    got = namespace[node.name](out.clone(), q, k, v, ck[:, :, pos], cv[:, :, pos], pos)
    reference = out.clone()
    for index in pos:
        ki, vi = k.clone(), v.clone()
        ki[:, :, index], vi[:, :, index] = ck[:, :, index], cv[:, :, index]
        reference[:, :, index:index+1] = (q[:, :, index:index+1] @ ki.transpose(-1, -2)/math.sqrt(8)).softmax(-1) @ vi
    torch.testing.assert_close(got, reference, atol=1e-12, rtol=1e-12)
