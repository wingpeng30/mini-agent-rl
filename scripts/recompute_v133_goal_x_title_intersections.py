"""Recompute Goal X post-G historical title intersections from the frozen union.

The first X0 index extended its title union after remote Goal J assets arrived
but retained earlier per-split intersection lists. This addendum leaves that
index intact and supplies corrected counts with an auditable source hash.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-x-audit-and-transfer"
W = ROOT / "runs/v133-goal-w-transfer/data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def main():
    target = OUT / "x0-title-intersections-corrected.json"
    if target.exists():
        raise FileExistsError(target)
    source = OUT / "x0-history-exposure-index.json"
    index = json.loads(source.read_text(encoding="utf-8"))
    titles = set(index["post_g_title_union"]["normalized_titles"])
    result = {"status": "corrected_intersection_lists_only", "historical_index_sha256": sha(source),
              "historical_title_count": len(titles), "splits": {}}
    for split in ("train", "dev", "confirm"):
        cases_path = W / f"candidate-{split}.jsonl"
        corpus_path = W / f"corpus-{split}.json"
        cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
        docs = {doc["id"]: doc for doc in json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]}
        support = sorted({norm(docs[doc_id]["title"]) for case in cases for doc_id in case["support_ids"]
                          if norm(docs[doc_id]["title"]) in titles})
        corpus = sorted({norm(doc["title"]) for doc in docs.values() if norm(doc["title"]) in titles})
        result["splits"][split] = {"candidate_file_sha256": sha(cases_path),
                                    "corpus_file_sha256": sha(corpus_path),
                                    "historical_candidate_support_title_keys": support,
                                    "historical_corpus_title_keys": corpus,
                                    "support_title_count": len(support), "corpus_title_count": len(corpus)}
    target.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: {k: value[k] for k in ("support_title_count", "corpus_title_count")}
                      for split, value in result["splits"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
