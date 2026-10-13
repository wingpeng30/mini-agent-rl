"""Replace source-quality failures within Goal Z's originally frozen pools.

This is a pre-model-output review amendment. It never changes the candidate
pools, source archives, corpora, target counts, or evaluation rules.
"""
from __future__ import annotations

import json
import shutil
import argparse
from pathlib import Path

from mini_agent_rl.tools.search import SearchTool

import curate_v133_goal_z_multihop as curate
import prepare_v133_goal_z_multihop_data as z


FLAGS = {
    "train": {
        "musique:2hop__720263_161223", "musique:2hop__746150_556534",
        "musique:2hop__234880_46285", "musique:2hop__131821_51968",
        "musique:2hop__808685_126084", "musique:2hop__635362_91717",
        "musique:2hop__602786_3814", "musique:2hop__104019_23142",
        "musique:2hop__24685_41437", "musique:2hop__248482_59201",
        "musique:2hop__410137_47336", "musique:2hop__388207_24352",
        "musique:2hop__581063_4037", "musique:2hop__89253_49700",
        "musique:2hop__598306_65075", "musique:2hop__208855_694982",
        "musique:2hop__484115_81963", "musique:2hop__61258_43178",
        "musique:2hop__544332_224276", "musique:2hop__136500_92216",
        "musique:2hop__742976_61463", "musique:2hop__296886_160137",
        "musique:2hop__561974_160137", "musique:2hop__80539_74402",
    },
    "dev": {
        "musique:2hop__107548_124896", "musique:2hop__30741_85990",
        "musique:2hop__46550_85990", "musique:2hop__90773_715759",
        "musique:2hop__496732_658428",
    },
    "confirm": {
        "musique:2hop__58939_189318", "musique:2hop__91713_289940",
        "musique:2hop__491195_265959", "musique:2hop__62020_158262",
        "musique:2hop__663184_57816", "2wiki:f4885f140bda11eba7f7acde48001122",
    },
}

FLAGS_2 = {
    "train": set(),
    "dev": {"musique:2hop__648648_173389", "musique:2hop__226204_69048",
            "musique:2hop__747758_58067", "musique:2hop__373759_136043"},
    "confirm": {"musique:2hop__53175_548781", "2wiki:9481ffc00bdb11eba7f7acde48001122"},
}

FLAGS_3 = {
    "train": {"musique:2hop__529677_107048", "musique:2hop__833234_55369",
              "musique:2hop__146174_8682", "musique:2hop__152980_52794",
              "musique:2hop__139721_256194", "musique:2hop__581005_53519",
              "musique:2hop__118995_825694", "musique:2hop__26953_85063",
              "musique:2hop__782642_52667"},
    "dev": {"musique:2hop__460091_405771"},
    "confirm": set(),
}

FLAGS_4 = {
    "train": {"musique:2hop__713250_159417", "musique:2hop__37161_24960",
              "musique:2hop__558996_529125"},
    "dev": set(),
    "confirm": set(),
}

FLAGS_5 = {
    "train": {"musique:2hop__104341_47939", "musique:2hop__593379_42328"},
    "dev": set(), "confirm": set(),
}

FLAGS_6 = {
    "train": {"musique:2hop__856348_124042"},
    "dev": set(), "confirm": set(),
}


def main(round_number: int) -> None:
    out = z.OUT
    flags = {1: FLAGS, 2: FLAGS_2, 3: FLAGS_3, 4: FLAGS_4, 5: FLAGS_5, 6: FLAGS_6}[round_number]
    report = out / ("replacement-review.json" if round_number == 1 else f"replacement-review-{round_number}.json")
    if report.exists():
        raise FileExistsError("Replacement pass already executed")
    history = z.w.history_exposure()
    manifest = z.json.loads((out / "curation.json").read_text(encoding="utf-8"))
    replacement = {}
    for split in ("train", "dev", "confirm"):
        path = out / ("selected-confirm.jsonl" if split == "confirm" else f"selected-{split}.jsonl")
        original = curate.rows(path)
        if not flags[split].issubset({v["id"] for v in original}):
            raise RuntimeError(f"Review ID absent from selected {split}")
        keep = [v for v in original if v["id"] not in flags[split]]
        corpus = z.json.loads((out / f"corpus-{split}.json").read_text(encoding="utf-8"))
        titles = {z.norm(d["title"]) for d in corpus}
        tool = SearchTool(corpus, "bm25-v2")
        used = {v["id"] for v in original}
        if round_number >= 2:
            for previous_round in range(1, round_number):
                prior = out / ("replacement-review.json" if previous_round == 1 else f"replacement-review-{previous_round}.json")
                previous = z.json.loads(prior.read_text(encoding="utf-8"))
                used.update(previous[split]["excluded_ids"])
        added = []
        for source in ("musique", "2wiki"):
            need = z.TARGETS[split][source] - sum(v["source"] == source for v in keep + added)
            for case in curate.rows(out / f"candidates-{split}-{source}.jsonl"):
                if need == 0:
                    break
                if case["id"] in used or z.audit_case(case, (history[0], history[1], history[2])):
                    continue
                support = {z.norm(d["title"]) for d in case["docs"] if d["support"]}
                if not support.issubset(titles):
                    continue
                if split == "train":
                    trajectory, reason = curate.teacher(case, tool)
                    if reason:
                        continue
                    selected = {k: v for k, v in case.items() if k != "docs"} | {"teacher": trajectory}
                else:
                    selected = {k: v for k, v in case.items() if k != "docs"} | {
                        "support_titles": [d["title"] for d in case["docs"] if d["support"]]}
                added.append(selected)
                used.add(case["id"])
                need -= 1
            if need:
                raise RuntimeError(f"Replacement shortfall: {split}/{source}/{need}")
        revised = keep + added
        if len(revised) != len(original):
            raise RuntimeError("Replacement altered count")
        backup = path.with_suffix(path.suffix + (".initial" if round_number == 1 else f".before-review{round_number}"))
        shutil.copyfile(path, backup)
        temporary = path.with_suffix(path.suffix + ".reviewed")
        z.jsonl(temporary, revised)
        temporary.replace(path)
        manifest["selected_sha256"][split] = z.digest(path)
        replacement[split] = {"excluded_ids": sorted(flags[split]),
                              "replacement_ids": [v["id"] for v in added],
                              "initial_sha256": z.digest(backup), "revised_sha256": z.digest(path)}
        print(json.dumps({"split": split, "excluded": len(flags[split]),
                          "replaced": len(added)}), flush=True)
    (out / "curation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.write_text(json.dumps(replacement, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, choices=(1, 2, 3, 4, 5, 6), default=1)
    main(parser.parse_args().round)
