"""Verify selected search top3 documents against the frozen candidate corpus."""
from __future__ import annotations

import json

from mini_agent_rl.tools.search import _BM25Index
from scripts.prepare_v117 import OUT
from scripts.prepare_v117_review import loadl, sha, write


def project(documents):
    return [{key: doc[key] for key in ("id", "title", "text")} for doc in documents]


def main():
    frozen = OUT / "frozen"
    selected = set(json.loads((frozen / "internal-private.json").read_text(encoding="utf-8"))["case_ids"])
    corpus = json.loads((frozen / "internal-corpus.json").read_text(encoding="utf-8"))
    index = _BM25Index(corpus)
    packets = [p for p in loadl(OUT / "review/internal-packets.jsonl") if p["case_id"] in selected]
    searches = [p for p in packets if p["proposed_action"]["action"] == "search"]
    if len(packets) != 100 or len(searches) != 50:
        raise ValueError("selected_state_count_mismatch")
    wrong = [p["case_id"] for p in searches
             if project(index.search(p["proposed_action"]["query"], top_k=3)) != p["retrieved_after_search"]]
    if wrong:
        raise RuntimeError("retrieval_top3_drift:" + ",".join(wrong[:5]))
    report = {"version": "v117-selected-retrieval-replay-v1", "search_states": 50,
              "corpus_sha256": sha(frozen / "internal-corpus.json"), "packet_file_sha256":
              sha(OUT / "review/internal-packets.jsonl"), "top3_documents_exact": True,
              "runtime_score_field_omitted_from_blind_packets": True}
    write(OUT / "retrieval-replay.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
