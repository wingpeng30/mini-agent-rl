"""Inspect the first reconstructed Goal R diagnostic prompt mismatch."""
import run_v133_goal_r_evidence_order as r
import run_v133_goal_q_search as q

client, _, _ = q.make_client(r.START, deterministic=True)
client._load()
for row in r.rows(r.OUT / "eval-incumbent.jsonl"):
    final = row["trace"][-1]
    if final.get("action") != "answer": continue
    messages = r.rebuild(row["question"], row["trace"], reverse=False)
    ids, _ = r.prompt_ids(client, messages)
    old = final["prompt_token_ids"]
    if ids != old:
        print("CASE", row["case_id"], "lengths", len(ids), len(old))
        idx = next((i for i, (a, b) in enumerate(zip(ids, old)) if a != b), min(len(ids), len(old)))
        print("FIRST_DIFF", idx)
        print("REBUILT", repr(client._processor.tokenizer.decode(ids[max(0, idx - 30):idx + 50])))
        print("ACTUAL", repr(client._processor.tokenizer.decode(old[max(0, idx - 30):idx + 50])))
        print("USAGE", final["usage"].get("prompt_truncated"))
        break
else:
    print("ALL_MATCH")
