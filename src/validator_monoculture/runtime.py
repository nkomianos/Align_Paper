"""Pinned text-generation runtime shared by the G0 collection phases."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from typing import Any

from under_extinction.modeling import QWEN35_MODEL_ID, chat_prompt_text


GEMMA4_MODEL_ID = "google/gemma-4-12B-it"
FROZEN_MODELS = {QWEN35_MODEL_ID, GEMMA4_MODEL_ID}


def deterministic_seed(*parts: str | int) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


class TextRuntime:
    def __init__(self, model_id: str, revision: str, *, local_files_only: bool = False) -> None:
        import torch

        if model_id not in FROZEN_MODELS:
            raise ValueError("model is not part of the frozen validator-monoculture gate")
        self.model_id = model_id
        self.revision = revision
        if model_id == QWEN35_MODEL_ID:
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM

            self.processor = AutoTokenizer.from_pretrained(
                model_id, revision=revision, use_fast=True, local_files_only=local_files_only
            )
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
                use_kernels=False,
                local_files_only=local_files_only,
            ).eval()
        else:
            from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(
                model_id, revision=revision, local_files_only=local_files_only
            )
            self.model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ).eval()

    def provenance(self) -> dict[str, Any]:
        """Record prompt-template and library identity without model weights."""

        import torch
        import transformers

        template = getattr(self.processor, "chat_template", None)
        if template is None and hasattr(self.processor, "tokenizer"):
            template = getattr(self.processor.tokenizer, "chat_template", None)
        encoded = json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "chat_template_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "platform_system": platform.system(),
            "cuda_version": str(torch.version.cuda),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "device_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else 0
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else []
            ),
        }

    def _inputs(self, prompt: str) -> Any:
        if self.model_id == QWEN35_MODEL_ID:
            rendered = chat_prompt_text(self.processor, [{"role": "user", "content": prompt}])
            return self.processor(
                rendered, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch

        if max_new_tokens < 32:
            raise ValueError("max_new_tokens is too small")
        if do_sample and not (0 < temperature <= 2 and 0 < top_p <= 1):
            raise ValueError("invalid sampling parameters")
        inputs = self._inputs(prompt)
        input_length = int(inputs["input_ids"].shape[-1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
        generated = output[0, input_length:]
        return self.processor.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
