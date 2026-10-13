"""Experimental whole-action CAD. No model or torch import at module import time."""
from __future__ import annotations

import json
from pydantic import BaseModel, Field
from typing import Literal
from ..domain import Message


class ContextAwareDecodingConfig(BaseModel):
    alpha: float = Field(default=1.0, ge=0)
    reference_mode: Literal["empty-document-content-v1"] = "empty-document-content-v1"
    after_searches: tuple[int, ...] = (1, 2)


def search_count(messages: list[Message]) -> int:
    return sum(m.role == "tool" for m in messages)


def reference_messages(messages: list[Message]) -> list[Message]:
    """Remove only returned title/text values, preserving even repeated documents."""
    result = []
    for message in messages:
        if message.role != "tool":
            result.append(message.model_copy(deep=True))
            continue
        payload = json.loads(message.content)
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
            raise ValueError("CAD requires canonical search observations")
        for document in payload["result"]:
            if not isinstance(document, dict):
                raise ValueError("CAD document must be an object")
            for field in ("title", "text"):
                if field in document:
                    document[field] = ""
        result.append(message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)}, deep=True))
    return result


class ContextAwareLogitsProcessor:
    """Independent cached reference following the primary stream's actual prefix.

    Uses the installed Transformers 5.17 generation preparation/update helpers.
    Explicit text position IDs avoid Qwen multimodal model-level rope state.
    """
    def __init__(self, model, primary_length, reference_inputs, alpha):
        import torch
        from transformers.cache_utils import DynamicCache
        self.model, self.primary_length, self.alpha = model, primary_length, alpha
        self.ids = reference_inputs["input_ids"].clone()
        mask = reference_inputs.get("attention_mask", torch.ones_like(self.ids)).clone()
        self.kwargs = dict(attention_mask=mask, position_ids=mask.long().cumsum(-1) - 1,
                           past_key_values=DynamicCache(config=model.config.get_text_config()),
                           use_cache=True, logits_to_keep=1)
        self.steps = []
        self.reference_tokens = self.ids[0].tolist()

    def __call__(self, input_ids, scores):
        import torch
        step = len(self.steps)
        prefix = input_ids[:, self.primary_length:]
        if prefix.shape[1] != step:
            raise RuntimeError("CAD prefix alignment failure")
        if step:
            self.ids = torch.cat([self.ids, prefix[:, -1:]], dim=-1)
        prepared = self.model.prepare_inputs_for_generation(
            self.ids, next_sequence_length=None if step == 0 else 1,
            is_first_iteration=step == 0, **self.kwargs)
        output = self.model(**prepared, return_dict=True)
        reference = output.logits[:, -1, :].float()
        primary = scores.float()
        if not torch.isfinite(reference).all() or not torch.isfinite(primary).all():
            raise FloatingPointError("CAD nonfinite logits")
        combined = (1 + self.alpha) * primary - self.alpha * reference
        if not torch.isfinite(combined).all():
            raise FloatingPointError("CAD nonfinite combined logits")
        selected = int(combined.argmax(-1).item())
        plain = int(primary.argmax(-1).item())
        prior = int(reference.argmax(-1).item())
        candidates = sorted({selected, plain, prior})
        self.steps.append(dict(step=step, selected_token=selected, full_argmax=plain,
                               reference_argmax=prior, reference_cache_tokens=self.kwargs["past_key_values"].get_seq_length(),
                               logits={str(i):dict(full=float(primary[0,i]), reference=float(reference[0,i]),
                                                  cad=float(combined[0,i])) for i in candidates}))
        self.kwargs = self.model._update_model_kwargs_for_generation(output, self.kwargs)
        return combined

    def audit(self):
        return dict(enabled=True, alpha=self.alpha, reference_prompt_token_ids=self.reference_tokens,
                    reference_sequences=1, reference_forward_passes=len(self.steps),
                    primary_forward_passes=len(self.steps), independent_reference_cache=True,
                    explicit_reference_text_positions=True, token_scores=self.steps)
