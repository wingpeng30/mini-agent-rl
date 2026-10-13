import difflib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v133_goal_ac_failure_state_repair as ac
import run_v133_goal_ab_evidence_decision as ab
import run_v133_goal_z_multihop_sft as z
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(z.q.p.BASE, local_files_only=True)
client, _, _ = z.make_client(ac.SOURCE, z.OUT / "corpus-train.json")
client._load()
votes = {x["case_id"]: x for x in ac.rows(ac.AB / "train-decision-review.jsonl")}
cases = [x for x in ac.rows(ac.AB / "train-on-policy.jsonl") if votes[x["case_id"]]["on_policy_stop_error"]]
for case in cases:
    for i, step in enumerate(case["trace"]):
        if i and step["action"] in ("answer", "search"):
            messages = ac.messages_for({"question": case["question"], "trace": case["trace"]}, i, tokenizer)
            encoded = client._inputs(messages, "debug")
            got = encoded["input_ids"][0].tolist()
            old = step["prompt_token_ids"]
            if got != old:
                prefix = 0
                while prefix < min(len(got), len(old)) and got[prefix] == old[prefix]: prefix += 1
                print(json.dumps({"case_id": case["case_id"], "step": i,
                    "old_len": len(old), "got_len": len(got), "first_diff": prefix,
                    "old_tail": tokenizer.decode(old[max(0,prefix-8):prefix+24], skip_special_tokens=False),
                    "got_tail": tokenizer.decode(got[max(0,prefix-8):prefix+24], skip_special_tokens=False),
                    "old_prompt": tokenizer.decode(old, skip_special_tokens=False),
                    "got_prompt": tokenizer.decode(got, skip_special_tokens=False)}, ensure_ascii=False))
                sys.exit(0)
print("no mismatch")
