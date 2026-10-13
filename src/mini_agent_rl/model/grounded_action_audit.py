"""Diagnostic-only grounded action checks; never dispatch or rewrite actions."""
import hashlib
import json
import re


def visible_documents(messages):
    documents = []
    for index, message in enumerate(messages):
        role, content = message.get("role"), message.get("content", "")
        if role == "user" and content.startswith("<tool_response>\n") and content.endswith("\n</tool_response>"):
            content = content[len("<tool_response>\n"):-len("\n</tool_response>")]
        elif role != "tool":
            continue
        payload = json.loads(content)
        for rank, document in enumerate(payload.get("result", []), 1):
            documents.append(dict(message_index=index, rank=rank, document_id=document.get("id"),
                text=document.get("text", ""), body_sha256=hashlib.sha256(document.get("text", "").encode()).hexdigest()))
    return documents


def map_documents(tokenizer, ids, messages):
    """Map actual serialized JSON bodies, including escapes, into actual input IDs."""
    rendered = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise RuntimeError("input token offsets cannot exactly reconstruct input")
    offsets = encoded["offset_mapping"]
    cursor = 0
    mapped = []
    for document in visible_documents(messages):
        matches = list(re.finditer(r'"text"\s*:\s*("(?:[^"\\]|\\.)*")', rendered[cursor:]))
        match = next((m for m in matches if json.loads(m.group(1)) == document["text"]), None)
        if match is None:
            raise RuntimeError("visible body missing from actual serialized input")
        start, end = cursor + match.start(1) + 1, cursor + match.end(1) - 1
        token_positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        document = dict(document, serialized_char_span=[start, end],
            input_token_span=[min(token_positions), max(token_positions)+1] if token_positions else [0, 0],
            serialized_body=rendered[start:end])
        mapped.append(document)
        cursor = end + 1
    return mapped, offsets


def audit_grounded_action(raw, documents, offsets=None):
    errors = []
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return dict(schema_valid=False, exact_quotes_valid=False, errors=["not_one_json_object"], links=[])
    if not isinstance(value, dict):
        return dict(schema_valid=False, exact_quotes_valid=False, errors=["not_an_object"], links=[])
    links = value.get("links")
    if not isinstance(links, list) or len(links) > 2:
        errors.append("links_must_be_list_of_zero_to_two")
        links = []
    if not isinstance(value.get("missing_relation"), str):
        errors.append("missing_relation_must_be_string")
    keys = list(value)
    if "action" in keys and any(k not in keys or keys.index(k) > keys.index("action") for k in ("links", "missing_relation")):
        errors.append("audit_fields_must_precede_action")
    if value.get("action") not in ("answer", "search"):
        errors.append("invalid_action_type")
    target = "answer" if value.get("action") == "answer" else "query"
    if not isinstance(value.get(target), str) or not value.get(target, "").strip():
        errors.append("missing_action_value")
    mapped_links = []
    for number, link in enumerate(links):
        fields = ("subject", "relation", "object", "document_id", "quote")
        if not isinstance(link, dict) or any(not isinstance(link.get(k), str) or not link[k].strip() for k in fields):
            errors.append(f"link_{number}_invalid_fields")
            mapped_links.append(dict(link=link, exact_quote=False, occurrences=[]))
            continue
        occurrences = []
        for document in documents:
            if document["document_id"] != link["document_id"]:
                continue
            position = document["text"].find(link["quote"])
            if position < 0:
                continue
            occurrence = {k: v for k, v in document.items() if k not in ("text", "serialized_body")}
            occurrence["body_char_span"] = [position, position + len(link["quote"])]
            if offsets is not None:
                body = document["serialized_body"]
                encoded_quote = next((json.dumps(link["quote"], ensure_ascii=ascii_)[1:-1]
                    for ascii_ in (False, True) if json.dumps(link["quote"], ensure_ascii=ascii_)[1:-1] in body), None)
                if encoded_quote is None:
                    raise RuntimeError("exact quote cannot be mapped into serialized body")
                start = document["serialized_char_span"][0] + body.index(encoded_quote)
                end = start + len(encoded_quote)
                positions = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
                occurrence["quote_input_token_span"] = [min(positions), max(positions)+1]
            occurrences.append(occurrence)
        mapped_links.append(dict(link=link, exact_quote=bool(occurrences), occurrences=occurrences))
    schema_valid = not errors
    return dict(schema_valid=schema_valid, exact_quotes_valid=schema_valid and all(x["exact_quote"] for x in mapped_links),
        errors=errors, links=mapped_links, missing_relation=value.get("missing_relation"),
        semantic_support="requires_source_grounded_adjudication", affects_dispatch=False)
