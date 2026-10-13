"""Goal AR: lossless evidence addresses and syntax-only action constraints.

No source judgements or expected answers are used by this module. The address
mask guarantees only that selected spans are present, never that they entail
the question's relations.
"""
from __future__ import annotations

import hashlib
import json
import re

PRESENTATION = "evidence-addresses-v1"


def addressed_observation(raw: str, prefix: str):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), list):
        raise ValueError("addressed observations require a search result list")
    chunks = ["Search metadata (JSON):\n",
              json.dumps({k:v for k,v in payload.items() if k != "result"}, ensure_ascii=False), "\n"]
    spans = []
    cursor = sum(map(len, chunks))
    for rank, doc in enumerate(payload["result"], 1):
        body = doc.get("text")
        if not isinstance(body, str):
            raise ValueError("every document must have an original string body")
        # Breaks are positional and deterministic; no sentence or entity scoring.
        boundaries = [0] + [m.start() for m in re.finditer(r"(?<=[.!?])(?=\s+[A-Z])|(?<=\n)", body)] + [len(body)]
        boundaries = sorted(set(boundaries))
        header = "\nDocument metadata (JSON):\n" + json.dumps(
            {k:v for k,v in doc.items() if k != "text"}, ensure_ascii=False) + "\n"
        chunks.append(header); cursor += len(header)
        recovered = []
        for n, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            value = body[start:end]
            address = f"{prefix}-r{rank}-s{n}"
            label = f"\n[EVIDENCE {address} chars {start}:{end}]\n"
            chunks.append(label); cursor += len(label)
            rendered_start = cursor
            chunks.append(value); cursor += len(value)
            recovered.append(value)
            spans.append(dict(address=address, rank=rank, document_id=doc.get("id"),
                body_sha256=hashlib.sha256(body.encode()).hexdigest(), body_chars=len(value),
                body_char_span=[start,end], quote=value,
                rendered_char_span=[rendered_start,cursor]))
        if "".join(recovered) != body:
            raise RuntimeError("evidence indexing changed original body")
        chunks.append("\n[END DOCUMENT]\n"); cursor += len(chunks[-1])
    return "".join(chunks), spans


def action_grammar(mode: str, addresses: list[str]):
    if mode not in ("json-action-v1", "evidence-action-v1"):
        raise ValueError(mode)
    prefix = ""
    extra = ""
    if mode == "evidence-action-v1":
        prefix = 'evidence ws "," ws '
        if addresses:
            values = " | ".join(json.dumps(json.dumps(x)) for x in addresses)
            choices = " | ".join('address' + ' ws "," ws address' * n for n in range(4))
            extra = f'address ::= {values}\nchoices ::= {choices}\n'
            item = '(choices)?'
        else:
            item = ''
        extra += f'evidence ::= "\\\"evidence\\\"" ws ":" ws "[" ws {item} ws "]"\n'
    return ('root ::= "{" ws ' + prefix + 'action ws "}" ws\n'
        'action ::= "\\\"action\\\"" ws ":" ws "\\\"search\\\"" ws "," ws "\\\"query\\\"" ws ":" ws string | '
        '"\\\"action\\\"" ws ":" ws "\\\"answer\\\"" ws "," ws "\\\"answer\\\"" ws ":" ws string\n'
        'string ::= "\\\"" ([^"\\\\\\x00-\\x1f] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\\""\n'
        'ws ::= [ \\t\\n\\r]*\n' + extra)


def make_processor(client, inputs):
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor
    import torch
    config = client.config
    if config.recompute_old_logprobs:
        raise ValueError("constraint inference cannot expose unmasked logprobs as behaviour probabilities")
    if config.enable_thinking or inputs["input_ids"].shape[0] != 1:
        raise ValueError("constraints require single-sample, non-thinking generation")
    tokenizer = getattr(client._processor, "tokenizer", client._processor)
    if not hasattr(client, "_ar_compiler"):
        vocab = client._model.get_output_embeddings().weight.shape[0]
        from .local import generation_terminator_ids
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab,
            stop_token_ids=generation_terminator_ids(tokenizer))
        client._ar_compiler = xgr.GrammarCompiler(info)
    spans = inputs._context_usage["presentation"].get("document_bodies", [])
    addresses = [s["address"] for s in spans if "address" in s]
    grammar = action_grammar(config.action_constraint, addresses)
    compiled = client._ar_compiler.compile_grammar(grammar)
    processor = LogitsProcessor(compiled)

    class FiniteConstraint:
        def __call__(self, input_ids, scores):
            if torch.isnan(scores).any() or torch.isposinf(scores).any():
                raise FloatingPointError("non-finite policy logits before grammar mask")
            result = processor(input_ids, scores)
            if not torch.isfinite(result).any(dim=-1).all():
                raise FloatingPointError("grammar mask leaves no finite legal token")
            return result
    inputs._context_usage["action_constraint"] = config.action_constraint
    inputs._context_usage["grammar_sha256"] = hashlib.sha256(grammar.encode()).hexdigest()
    inputs._context_usage["available_evidence_addresses"] = addresses
    return FiniteConstraint()


def audit_selection(raw, spans):
    from .local import first_json_object_end
    end = first_json_object_end(raw)
    value = json.loads(raw[:end] if end is not None else raw)
    selected = value.get("evidence")
    by = {s["address"]:s for s in spans}
    valid = isinstance(selected, list) and len(selected) <= 4 and all(isinstance(x,str) and x in by for x in selected)
    return dict(addresses_valid=valid, selected_spans=[by[x] for x in selected] if valid else [],
        duplicate_addresses=len(selected) != len(set(selected)) if valid else None,
        semantic_support="requires_source_grounded_adjudication", affects_dispatch=False)
