"""One-pass source-grounded reconciliation of Goal AP v2 probe outputs."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "runs/v133-goal-ap-evidence-presentation-v1"
V2 = ROOT / "runs/v133-goal-ap-evidence-presentation-v2"


def read(name: str, folder: Path = V2):
    return [json.loads(x) for x in (folder / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def norm(value):
    return " ".join((value or "").casefold().replace(".", "").replace(",", "").split())


def citations(model, arm, probe_id, requirements):
    item = next(x for x in read("probe-inputs.jsonl")
                if (x["model"], x["arm"], x["probe_id"]) == (model, arm, probe_id))
    body_rows = []
    for message in item["messages"]:
        if message["role"] != "tool":
            continue
        for doc in json.loads(message["content"]).get("result", []):
            body_rows.append(doc)
    mappings = item["presentation"].get("document_bodies", [])
    result = []
    for doc_id, quote in requirements:
        doc = next((x for x in body_rows if x.get("id") == doc_id and quote in x.get("text", "")), None)
        mapping = next((x for x in mappings if x.get("document_id") == doc_id and
                        x.get("body_sha256") == __import__("hashlib").sha256(doc["text"].encode()).hexdigest()), None) if doc else None
        if doc is None or mapping is None:
            raise RuntimeError(f"citation text/token mapping missing: {probe_id}/{doc_id}")
        start = doc["text"].index(quote)
        result.append({"document_id": doc_id, "title": doc.get("title"), "quote": quote,
                       "body_char_span": [start, start + len(quote)],
                       "document_input_token_span": mapping["input_token_span"],
                       "body_sha256": mapping["body_sha256"]})
    return result


def main():
    prior = read("probe-adjudications.jsonl", V1)
    prior_a = {(x["model"], x["probe_id"]): x for x in prior if x["arm"] == "A"}
    outputs = {}
    for model in ("AM", "Z"):
        for arm in ("A", "B"):
            for row in read(f"probe-{model}-{arm}.jsonl"):
                outputs[(model, arm, row["probe_id"])] = row
    inputs = {(x["model"], x["arm"], x["probe_id"]): x for x in read("probe-inputs.jsonl")}
    evidence = {x["probe_id"]: x for x in read("frozen-probes.jsonl")}

    result = []
    for model in ("AM", "Z"):
        for arm in ("A", "B"):
            for probe_id in [x["probe_id"] for x in read("frozen-probes.jsonl")]:
                current = outputs[(model, arm, probe_id)]
                source = dict(prior_a[(model, probe_id)])
                source.update({"arm": arm, "action": current["action"], "answer": current["answer"],
                               "query": current["query"], "raw": current["raw"],
                               "current_input_sha256": inputs[(model, arm, probe_id)]["input_sha256"],
                               "training_use_forbidden": True})
                if arm == "B":
                    a = outputs[(model, "A", probe_id)]
                    changed = (a["action"], norm(a["answer"]), norm(a["query"])) != (
                        current["action"], norm(current["answer"]), norm(current["query"]))
                    source["decision_reuse"] = "same underlying source ruling; current B text/token mapping rechecked"
                    if not changed:
                        source["adjudication_reason"] = (
                            "Reused the A source-grounded ruling: output action and normalized answer/query are equivalent; "
                            "CPU preflight verified identical tool JSON document bodies/order and current token spans."
                        )
                    elif model == "AM" and evidence[probe_id]["category"] == "complete":
                        if probe_id == "complete:2wiki:1dd33f080bde11eba7f7acde48001122":
                            source.update(supported_correct=True, binding_repair=True,
                                adjudication_reason=("Correct binding repair. The question asks for the birthplace of Johann "
                                "Nicolaus Bach's father. Current B input document 37f133d1958d1a8bc640b31d states Johann "
                                "Nicolaus was a son of Johann Christoph Bach and that Johann Christoph Bach was born at "
                                "Arnstadt. The citation body is preserved in full; B cites its current document token span."))
                            source["citation_refs"] = citations("AM", "B", probe_id, [
                                ("4daf04bbbc3ecccdf9a4fd0f", "Johann Nicolaus was the eldest son of Johann Christoph Bach"),
                                ("37f133d1958d1a8bc640b31d", "He was born at Arnstadt"),
                            ])
                        elif probe_id == "complete:2wiki:49d965aa0bde11eba7f7acde48001122":
                            source.update(supported_correct=True, binding_repair=False,
                                adjudication_reason=("Equivalent entity answer. B omits the extra geographic phrase 'in Manila' "
                                "but names the same Las Piñas Medical Center stated in the visible source; this is not a binding repair."))
                            source["citation_refs"] = citations("AM", "B", probe_id, [
                                ("4cbda48c1f51a26640d69cbf", "at the Las Piñas Medical Center in Manila")
                            ])
                        elif probe_id in ("complete:2wiki:764039340bde11eba7f7acde48001122",
                                          "complete:2wiki:f299dd600bdd11eba7f7acde48001122"):
                            source.update(supported_correct=False, binding_repair=False,
                                adjudication_reason=("The full needed relation chain is visible in the actual input, and A's "
                                "answer was source-supported. B emits another search action instead of answering from that "
                                "complete chain; it is therefore not a supported-correct action for this complete-evidence probe."))
                            if probe_id.endswith("764039340bde11eba7f7acde48001122"):
                                source["citation_refs"] = citations("AM", "B", probe_id, [
                                    ("56f3b855da277556ca88965f", "Abercrombie was the illegitimate son of James Hamilton, 4th Duke of Hamilton"),
                                    ("de579d6e8297b9ce0bbbf9a9", "He died on 15 November 1712 as the result of a celebrated duel in Hyde Park, Westminster"),
                                ])
                            else:
                                source["citation_refs"] = citations("AM", "B", probe_id, [
                                    ("9041c550da7e2c17d1e61654", "He was the son of Don Juan de la Cerda, 4th Duke of Medinaceli"),
                                    ("083960bde826bfe3d1196f74", "Juan de la Cerda, 4th Duke of Medinaceli (c. 1514 – 1575)"),
                                ])
                        else:
                            raise RuntimeError(f"unreviewed changed AM complete output: {probe_id}")
                    elif model == "AM" and evidence[probe_id]["category"] == "partial":
                        current_input = inputs[(model, arm, probe_id)]
                        visible = "\n".join(m["content"] for m in current_input["messages"])
                        query = current.get("query") or ""
                        if probe_id.endswith("862994_69048:step1"):
                            legal = all(term.casefold() in visible.casefold() for term in ("Gloria Kemasuode", "chief judge"))
                            reason = "The question names Gloria Kemasuode and asks for the chief judge at her birthplace; query keeps those visible entities/relations and does not add an answer."
                        elif probe_id.endswith("506035_325512:step1"):
                            legal = all(term.casefold() in visible.casefold() for term in ("Poplarville", "Mississippi", "county"))
                            reason = "The visible Larkin I. Smith result says born in Poplarville, Mississippi; the query targets the missing county relation with that visible bridge location."
                        elif probe_id.endswith("337223_56873:step1"):
                            legal = False
                            reason = ("The user asks when Mark Crossley's team won the FA Cup. Although a visible FA Cup result "
                                "mentions the 2015–16 UEFA Europa League season, that is an unrelated competition rule, not a "
                                "team-win date or requested season. Adding 2015 narrows the target relation using irrelevant evidence.")
                            source["citation_refs"] = citations("AM", "B", probe_id, [
                                ("dd90c27efd4ec176aa05ddb7", "From the 2015–16 UEFA Europa League season")
                            ])
                        elif probe_id.endswith("796221_18803:step1"):
                            legal = all(term.casefold() in visible.casefold() for term in ("MV Rapana", "April 2010"))
                            reason = "The query paraphrases the visible question's entity/time and seeks the missing announced action; no hidden answer is added."
                        else:
                            raise RuntimeError(f"unreviewed changed AM partial query: {probe_id}")
                        source.update(legal_bridge_search=legal,
                            premature_answer=current["action"] == "answer",
                            invalid_action=current.get("parse_error") is not None,
                            repeated_query=norm(query) in {norm(q) for q in evidence[probe_id].get("previous_queries", [])},
                            adjudication_reason=reason)
                    elif model == "Z":
                        source.update(supported_correct=None, binding_repair=None, legal_bridge_search=None,
                            premature_answer=None, invalid_action=current.get("parse_error") is not None,
                            repeated_query=None,
                            adjudication_reason=("Goal Z is the descriptive model-dependence probe, not the continuation gate. "
                                "Output difference is recorded, but it is not included in the primary correctness count."))
                    else:
                        raise RuntimeError(f"unexpected changed output: {model}/{probe_id}")
                result.append(source)

    path = V2 / "probe-adjudications.jsonl"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in result),
                    encoding="utf-8", newline="\n")
    print(json.dumps({"adjudications": len(result), "am_complete_A": sum(
        x["supported_correct"] is True for x in result if x["model"] == "AM" and x["arm"] == "A" and x["category"] == "complete"),
        "am_complete_B": sum(x["supported_correct"] is True for x in result if x["model"] == "AM" and x["arm"] == "B" and x["category"] == "complete"),
        "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
