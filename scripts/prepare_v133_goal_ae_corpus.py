"""Freeze title inventories and build revision-pinned, paragraphized article text.

This fetches pages by their already-frozen titles through the Wikimedia content
API. It does not issue search queries and never reads the fresh Goal AB
confirmation inventory. The resulting corpus files are immutable run artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

try:
    import mwparserfromhell
except ImportError as exc:  # pragma: no cover - installed in the preparation environment
    raise SystemExit(
        "Goal AE preparation requires mwparserfromhell==0.7.2; install it in the "
        "data-preparation environment, not the product runtime."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ae-corpus-repair"
Z = ROOT / "runs/v133-goal-z-multihop-sft"
P = ROOT / "runs/v133-goal-p-system"
Y = ROOT / "runs/v133-goal-y-three-class-transfer"
AA = ROOT / "runs/v133-goal-aa-audit-confirmation"
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Goal-AE corpus reconstruction/1.0 (research evaluation)"
BATCH_SIZE = 20


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", decomposed.casefold()).strip()


def stable(value: str) -> str:
    return hashlib.sha256(("goal-ae-v1\0" + value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_once(path: Path, values: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def inventory_sources() -> dict[str, Path]:
    # Goal Z confirm was already exposed by Goal AA and is used here only as a
    # regression inventory. The fresh Goal AB confirmation inventory is omitted.
    return {
        "goal_z_train": Z / "corpus-train.json",
        "goal_z_dev": Z / "corpus-dev.json",
        "old_regression": P / "corpus.json",
        "goal_y_regression": Y / "dev-corpus.json",
        "aa_regression": Z / "corpus-confirm.json",
    }


def freeze_inputs() -> dict:
    sources = inventory_sources()
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing corpus inventories: " + ", ".join(missing))
    inventories: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    all_titles: dict[str, str] = {}
    for label, path in sources.items():
        rows = read_json(path)
        title_map: dict[str, str] = {}
        for row in rows:
            title = str(row.get("title", "")).strip()
            if title:
                title_map.setdefault(norm(title), title)
                all_titles.setdefault(norm(title), title)
        inventories[label] = sorted(title_map.values(), key=lambda x: (stable(label + ":" + norm(x)), norm(x)))
        hashes[label] = sha_file(path)
    ordered = sorted(all_titles.values(), key=lambda x: (stable("title:" + norm(x)), norm(x)))
    return {
        "version": "goal-ae-corpus-inputs-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "pre-existing frozen run corpora only; no case/question/answer files read",
        "fresh_goal_ab_confirmation_accessed": False,
        "corpora": {label: {"path": str(path.relative_to(ROOT)), "sha256": hashes[label],
                            "title_count": len(inventories[label])}
                    for label, path in sources.items()},
        "inventories": inventories,
        "unique_title_count": len(ordered),
        "unique_titles": ordered,
    }


def title_key(value: str) -> str:
    return norm(value.replace("_", " "))


def page_revision(page: dict) -> dict:
    revision = (page.get("revisions") or [{}])[0]
    slot = (revision.get("slots") or {}).get("main") or {}
    wikitext = slot.get("content") or slot.get("*") or ""
    if not wikitext and not page.get("missing"):
        raise ValueError(f"revision content absent for page {page.get('title')!r}")
    return {
        "pageid": page.get("pageid"),
        "canonical_title": page.get("title", ""),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "missing": bool(page.get("missing")),
        "wikitext": wikitext,
        "wikitext_sha256": sha_bytes(wikitext.encode("utf-8")) if wikitext else None,
    }


def fetch_batch(session: requests.Session, titles: list[str]) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "redirects": 1,
        "maxlag": 5,
        "titles": "|".join(titles),
    }
    response = session.get(API, params=params, timeout=(15, 90))
    if response.status_code != 200:
        raise RuntimeError(f"Wikimedia API HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    if payload.get("error"):
        raise RuntimeError("Wikimedia API error: " + json.dumps(payload["error"], ensure_ascii=True))
    query = payload.get("query")
    if not isinstance(query, dict):
        raise RuntimeError("Wikimedia API response has no query object")
    pages = [page_revision(page) for page in query.get("pages", [])]
    return {
        "input_titles": titles,
        "normalization": query.get("normalized", []),
        "redirects": query.get("redirects", []),
        "pages": pages,
        "continue": payload.get("continue"),
    }


def parse_article(page: dict) -> dict:
    text = mwparserfromhell.parse(page.pop("wikitext")).strip_code(normalize=True, collapse=True)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paras: list[str] = []
    for raw in re.split(r"\n\s*\n+", text):
        paragraph = re.sub(r"\s+", " ", raw).strip()
        if paragraph and (not paras or paragraph != paras[-1]):
            paras.append(paragraph)
    return {
        **page,
        "paragraphs": paras,
        "text": "\n\n".join(paras),
        "text_sha256": sha_bytes("\n\n".join(paras).encode("utf-8")),
        "source_url": (f"https://en.wikipedia.org/w/index.php?oldid={page['revision_id']}"
                       if page.get("revision_id") else None),
    }


def build_index() -> dict:
    if not OUT.exists():
        raise FileNotFoundError("run prepare first")
    frozen_path = OUT / "frozen-inputs.json"
    frozen = read_json(frozen_path)
    batches_dir = OUT / "api-batches"
    batch_paths = sorted(batches_dir.glob("batch-*.json"))
    pages_by_requested: dict[str, dict] = {}
    for path in batch_paths:
        batch = read_json(path)
        transforms: dict[str, str] = {}
        for item in batch.get("normalization", []) + batch.get("redirects", []):
            transforms[title_key(item["from"])] = item["to"]
        canonical_pages = {title_key(page["canonical_title"]): page for page in batch.get("pages", [])}
        for requested in batch["input_titles"]:
            key = title_key(requested)
            target = key
            seen: set[str] = set()
            while target in transforms and target not in seen:
                seen.add(target)
                target = title_key(transforms[target])
            page = canonical_pages.get(target) or canonical_pages.get(key)
            if page is not None:
                pages_by_requested[key] = page
    unique_pages: dict[tuple[int | None, int | None, str], dict] = {}
    for page in pages_by_requested.values():
        key = (page.get("pageid"), page.get("revision_id"), page.get("wikitext_sha256") or "missing")
        unique_pages.setdefault(key, page)
    articles = [parse_article(dict(page)) for page in unique_pages.values()]
    by_canonical = {title_key(x["canonical_title"]): x for x in articles}
    requested_map: dict[str, dict] = {}
    for requested_key, page in pages_by_requested.items():
        art = by_canonical.get(title_key(page["canonical_title"]))
        if art:
            requested_map[requested_key] = art
    missing_rows = []
    for label, titles in frozen["inventories"].items():
        for title in titles:
            article = requested_map.get(title_key(title))
            if article is None or article.get("missing") or not article.get("paragraphs"):
                missing_rows.append({"corpus": label, "requested_title": title,
                                     "reason": "page_missing_or_no_paragraphs"})
    write_jsonl_once(OUT / "articles.jsonl", articles)
    write_jsonl_once(OUT / "missing-pages.jsonl", missing_rows)
    corpus_summary = {}
    paragraph_rows = []
    for label, titles in frozen["inventories"].items():
        docs = []
        seen_paragraphs: set[str] = set()
        for requested_title in titles:
            article = requested_map.get(title_key(requested_title))
            if article is None or article.get("missing"):
                continue
            for idx, paragraph in enumerate(article["paragraphs"]):
                pid = sha_bytes(f"{label}\0{article['pageid']}\0{article['revision_id']}\0{idx}\0{paragraph}".encode())[:24]
                if pid in seen_paragraphs:
                    continue
                seen_paragraphs.add(pid)
                row = {"id": pid, "title": article["canonical_title"], "requested_title": requested_title,
                       "pageid": article["pageid"], "revision_id": article["revision_id"],
                       "paragraph_index": idx, "text": paragraph, "source_url": article["source_url"]}
                docs.append(row)
                paragraph_rows.append({"corpus": label, **row})
        corpus_path = OUT / f"paragraphs-{label}.json"
        write_json_once(corpus_path, docs)
        corpus_summary[label] = {"titles": len(titles), "articles": len({d['pageid'] for d in docs}),
                                 "paragraphs": len(docs), "sha256": sha_file(corpus_path)}
    write_jsonl_once(OUT / "paragraph-index.jsonl", paragraph_rows)
    result = {
        "status": "data_prepared" if not missing_rows else "data_incomplete",
        "article_count": len(articles),
        "missing_page_count": len(missing_rows),
        "corpora": corpus_summary,
        "paragraph_index_sha256": sha_file(OUT / "paragraph-index.jsonl"),
        "parser": {"package": "mwparserfromhell", "version": mwparserfromhell.__version__,
                   "transform": "strip_code(normalize=True, collapse=True), split on blank lines"},
    }
    write_json_once(OUT / "data-summary.json", result)
    return result


def prepare() -> dict:
    if OUT.exists() and not (OUT / "state.json").exists():
        raise FileExistsError(f"run directory exists without Goal AE state; refusing to overwrite: {OUT}")
    if not OUT.exists():
        OUT.mkdir(parents=True)
        frozen = freeze_inputs()
        write_json_once(OUT / "frozen-inputs.json", frozen)
        (OUT / "api-batches").mkdir()
        write_json_once(OUT / "state.json", {"status": "fetching_articles", "next_batch": 0,
                                               "created_at": datetime.now(timezone.utc).isoformat(),
                                               "fresh_goal_ab_confirmation_accessed": False})
    frozen = read_json(OUT / "frozen-inputs.json")
    if frozen.get("fresh_goal_ab_confirmation_accessed"):
        raise RuntimeError("fresh confirmation inventory must stay sealed until the dev gate")
    titles = frozen["unique_titles"]
    batches_dir = OUT / "api-batches"
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    total = (len(titles) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_no, start in enumerate(range(0, len(titles), BATCH_SIZE)):
        path = batches_dir / f"batch-{batch_no:04d}.json"
        if path.exists():
            continue
        batch_titles = titles[start:start + BATCH_SIZE]
        batch = fetch_batch(session, batch_titles)
        write_json_once(path, batch)
        if (batch_no + 1) % 10 == 0 or batch_no + 1 == total:
            state = read_json(OUT / "state.json")
            state.update({"status": "fetching_articles", "next_batch": batch_no + 1,
                          "completed_batches": batch_no + 1, "total_batches": total,
                          "retrieved_pages": sum(len(read_json(p).get("pages", [])) for p in batch_paths_now(batches_dir))})
            (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(0.25)
    result = build_index()
    state = read_json(OUT / "state.json")
    state.update({"status": result["status"], "data_summary_sha256": sha_file(OUT / "data-summary.json"),
                  "completed_at": datetime.now(timezone.utc).isoformat()})
    (OUT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def batch_paths_now(path: Path) -> list[Path]:
    return sorted(path.glob("batch-*.json"))


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
