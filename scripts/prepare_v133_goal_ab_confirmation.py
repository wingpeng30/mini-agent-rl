"""Freeze a fresh two-hop confirmation set before Goal AB candidate evaluation."""
from __future__ import annotations

import json
from collections import Counter

import curate_v133_goal_z_multihop as curator
import prepare_v133_goal_z_multihop_data as source
import run_v133_goal_ab_evidence_decision as ab


def entities(case: dict) -> set[str]:
    values = []
    if case["source"] == "musique":
        values.extend(str(x.get("answer", "")) for x in case.get("chain", [])[:-1])
    elif case["source"] == "2wiki":
        for x in case.get("chain", []):
            values.extend((str(x[0]), str(x[2])))
    values.extend(d["title"] for d in case.get("docs", []) if d.get("support"))
    values.extend(case.get("support_titles", []))
    return {source.norm(v) for v in values if len(source.norm(v)) >= 3}


def historical() -> tuple[set, set, set, set]:
    ids, questions, titles, _ = source.w.history_exposure()
    ids, questions, titles = set(ids), set(questions), set(titles)
    named = set()
    for split in ("train", "dev", "confirm"):
        path = source.OUT / ("selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl")
        for case in ab.z.rows(path):
            ids.update(str(x) for x in case["source_ids"])
            questions.add(source.w.qhash(case["question"]))
            named.update(entities(case))
            titles.update(entities(case) & {source.norm(x) for x in case.get("support_titles", [])})
    for split in ("train", "dev", "confirm"):
        for doc in ab.z.read(source.OUT / f"corpus-{split}.json"):
            titles.add(source.norm(doc["title"]))
    return ids, questions, titles, named


def prepare() -> dict:
    if ab.z.read(ab.STATE)["status"] != "awaiting_train_material":
        raise RuntimeError("Diagnostic gate has not passed")
    target = ab.OUT / "fresh-confirm-cases.jsonl"
    if target.exists() or (ab.OUT / "fresh-confirm-corpus.json").exists():
        raise FileExistsError("Confirmation material already frozen")
    ids, questions, titles, named = historical()
    titles.update(source.norm(d["title"]) for d in ab.z.read(ab.z.y.OUT / "dev-corpus.json"))
    titles.update(source.norm(d["title"]) for d in ab.z.read(ab.z.q.p.CORPUS))
    pools, rejection = {}, Counter()
    for name, loader in (("musique", source.load_musique), ("2wiki", source.load_wiki)):
        ranked = sorted((v for v in loader("dev") if source.source_eligible(name, v)),
                        key=lambda v: source.stable("goal-ab-confirm:" + name + ":" +
                                                  str(v.get("id", v.get("_id")))))
        pools[name] = [source.canonical(name, v) for v in ranked[:256]]
        if len(pools[name]) != 256:
            raise RuntimeError(f"Candidate pool shortfall: {name}")
        pool_path = ab.OUT / f"fresh-confirm-pool-{name}.jsonl"
        if pool_path.exists():
            if ab.z.rows(pool_path) != pools[name]:
                raise RuntimeError("Frozen confirmation candidate pool changed")
        else:
            source.jsonl(pool_path, pools[name])
    accepted = []
    reserved_support = set()
    for name in ("musique", "2wiki"):
        count = 0
        for case in pools[name]:
            reason = source.audit_case(case, (ids, questions, titles))
            if not reason and entities(case) & named:
                reason = "historical_entity"
            case_titles = {source.norm(x["title"]) for x in case["docs"] if x["support"]}
            if not reason and case_titles & reserved_support:
                reason = "cross_confirmation_title"
            if reason:
                rejection[reason] += 1
                continue
            accepted.append(case)
            reserved_support.update(case_titles)
            ids.update(case["source_ids"])
            questions.add(source.w.qhash(case["question"]))
            named.update(entities(case))
            count += 1
            if count == 32:
                break
        if count != 32:
            ab.z.write(ab.OUT / "fresh-confirm-shortfall.json", {
                "source": name, "found": count, "required": 32, "rejections": dict(rejection)})
            raise RuntimeError(f"Fresh confirmation shortfall: {name} {count}/32")
    reserved = {source.norm(d["title"]) for split in ("train", "dev", "confirm")
                for d in ab.z.read(source.OUT / f"corpus-{split}.json")}
    reserved.update(source.norm(d["title"]) for d in ab.z.read(ab.z.y.OUT / "dev-corpus.json"))
    reserved.update(source.norm(d["title"]) for d in ab.z.read(ab.z.q.p.CORPUS))
    corpus, _ = curator.build_corpus("confirm", accepted, reserved, titles)
    corpus_titles = {source.norm(d["title"]) for d in corpus}
    for case in accepted:
        if not {source.norm(d["title"]) for d in case["docs"] if d["support"]}.issubset(corpus_titles):
            raise RuntimeError("Chosen support not in actual confirmation library")
    if len(corpus) != 2048 or len(accepted) != 64:
        raise RuntimeError("Confirmation set size invalid")
    # All checks finish before either released artifact is written.
    for case in accepted:
        ab.z.append(target, {k: v for k, v in case.items() if k != "docs"} |
                    {"support_titles": [d["title"] for d in case["docs"] if d["support"]]})
    ab.z.write(ab.OUT / "fresh-confirm-corpus.json", corpus)
    manifest = {"pool": {name: ab.digest(ab.OUT / f"fresh-confirm-pool-{name}.jsonl")
                         for name in pools}, "case_sha256": ab.digest(target),
                "corpus_sha256": ab.digest(ab.OUT / "fresh-confirm-corpus.json"),
                "counts": dict(Counter(x["source"] for x in accepted)),
                "rejections": dict(rejection), "confirmation_accessed": False}
    ab.z.write(ab.OUT / "fresh-confirm-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False))
