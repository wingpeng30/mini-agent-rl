"""Read-only AL failure diagnosis. No stage resume, GPU model, search or output file."""
import json
import run_v133_goal_al_wide_context as al
from transformers import AutoProcessor


def main():
    processor = AutoProcessor.from_pretrained(str(al.assets.BASE), local_files_only=True)
    cpu = {arm: al.renderer(pc, processor) for arm, pc in al.ARMS.items()}
    originals, archived = al.source_history()
    count, longest, old_failures = 0, 0, 0
    for row in originals:
        saved = archived[row["case_id"]]
        for i, step in enumerate(row["trace"]):
            source_messages = al.prefix(row, i)
            a = cpu["A"]._inputs(source_messages, "failure_diagnosis")["input_ids"][0].tolist()
            b = cpu["B"]._inputs(source_messages, "failure_diagnosis")["input_ids"][0].tolist()
            raw = cpu["B"]._render_chat(source_messages)["input_ids"][0].tolist()
            assert a == step["prompt_token_ids"] == saved["trace"][i]["prompt_token_ids"]
            assert b == raw
            old_reconstruction = cpu["A"]._inputs(al.prefix(saved, i), "sorted_archive_diagnosis")["input_ids"][0].tolist()
            old_failures += old_reconstruction != a
            count += 1
            longest = max(longest, len(b))
    print(json.dumps(dict(cause="AK archive sorts JSON keys; production AH archive preserves key order",
        original_source_replay_matches=count, sorted_archive_replay_mismatches=old_failures,
        B_original_messages_retained=count, longest_B_input=longest,
        gpu_generations=0, new_searches=0, failed_stage_resumed=False)))


if __name__ == "__main__":
    main()
