"""Goal P candidate 1: general search text-field alias, with frozen baseline scoring."""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import run_v133_goal_p_system as p

OUT = p.OUT
RESULT = OUT / "p2-candidate-1.jsonl"


class SearchTextAliasClient:
    def __init__(self, base):
        self.base = base

    async def generate(self, messages, tools, *, temperature, seed):
        from mini_agent_rl.domain import ToolCall
        response = await self.base.generate(messages, tools, temperature=temperature, seed=seed)
        if not response.parse_error:
            return response
        raw = (response.raw_text or "").removesuffix("<|im_end|>").strip()
        try:
            value = json.loads(raw)
        except ValueError:
            return response
        if (isinstance(value, dict) and set(value) == {"action", "text"}
                and value.get("action") == "search" and isinstance(value.get("text"), str)
                and value["text"].strip()):
            usage = dict(response.usage)
            usage["protocol_recovery"] = "search_text_alias_v1"
            return response.model_copy(update={"content": "", "parse_error": None,
                "tool_calls": [ToolCall(name="search", arguments={"query": value["text"].strip()})],
                "usage": usage})
        return response

    def gpu_memory_peak_gb(self):
        return self.base.gpu_memory_peak_gb()


def run():
    c = p.config()
    if RESULT.exists():
        raise FileExistsError("candidate 1 already started")
    from evaluate_v133_goal_c_sft import normalize_answer, f1_score
    client, registry = p.make_components(c)
    client = SearchTextAliasClient(client)
    cases = [row for row in p.records(p.CASES) if row["split"] != "train"]
    if len(cases) != 32:
        raise RuntimeError("expected 32 development cases")
    started = time.monotonic()
    for row in cases:
        if time.monotonic()-started >= 900:
            raise TimeoutError("candidate 1 cap")
        try:
            item = asyncio.run(p.ask(row["question"], client, registry, c))
        except Exception as exc:
            item = {"question":row["question"],"answer":None,"reason":"model_error",
                    "error":str(exc),"trace":[],"model_calls":0,"searches":0,"elapsed_seconds":0}
        answer = item["answer"] or ""
        item.update(case_id=row["case_id"], split=row["split"],
                    em=int(bool(answer) and normalize_answer(answer)==normalize_answer(row["official_answer"])),
                    f1=f1_score(answer,row["official_answer"]) if answer else 0.0)
        observed = {doc["id"] for step in item["trace"] for doc in (step.get("observation",{}).get("result") or [])}
        item["retrieved_support_count"] = len(observed.intersection(row["support_ids"]))
        item["both_support_retrieved"] = item["retrieved_support_count"]==2
        p.append(RESULT,item)
        print(json.dumps({"case_id":row["case_id"],"reason":item["reason"],"em":item["em"]}),flush=True)
    values = p.records(RESULT)
    summary={"status":"PASSED","n":len(values),"em":sum(x["em"] for x in values),
             "f1":statistics.mean(x["f1"] for x in values),
             "both_support_retrieved":sum(x["both_support_retrieved"] for x in values),
             "invalid_action":sum(x["reason"]=="invalid_action" for x in values),
             "recovered_aliases":sum(step.get("context_usage",{}).get("protocol_recovery")=="search_text_alias_v1"
                                     for x in values for step in x["trace"]),
             "model_calls":sum(x["model_calls"] for x in values),"searches":sum(x["searches"] for x in values),
             "elapsed_seconds":round(time.monotonic()-started,3),
             "gpu_peak_allocated_gib":client.gpu_memory_peak_gb()}
    p.write(OUT/"p2-candidate-1-summary.json",summary)
    print(json.dumps(summary))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
