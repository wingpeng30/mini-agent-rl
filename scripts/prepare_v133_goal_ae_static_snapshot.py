"""Fetch only frozen-title rows from the pinned static Wikimedia parquet dump.

This is a bulk-data path: no search API, no live page/API calls, and no fresh
Goal AB confirmation inventory. Each shard is downloaded, filtered, and removed
before the next one is processed; only matching article text is retained.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ae-corpus-repair"
FROZEN = OUT / "frozen-inputs.json"
Z = ROOT / "runs/v133-goal-z-multihop-sft"
P = ROOT / "runs/v133-goal-p-system"
Y = ROOT / "runs/v133-goal-y-three-class-transfer"
SNAPSHOT_REPO = "wikimedia/wikipedia"
SNAPSHOT_CONFIG = "20231101.en"
SNAPSHOT_COMMIT = "e6057dc557255a03c9c3c47ceab0eb44353b1bc5"
SHARDS = 41
BASE_URL = f"https://huggingface.co/datasets/{SNAPSHOT_REPO}/resolve/{SNAPSHOT_COMMIT}/{SNAPSHOT_CONFIG}"
USER_AGENT = "Goal-AE corpus reconstruction/1.0 (https://github.com/wingpeng30/mini-agent-rl)"
MAX_PARAGRAPH_CHARS = 1800


def norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl_once(path: Path, values: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl_deterministic(path: Path, values: list[dict]) -> None:
    rendered = "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"existing artifact differs from deterministic output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def write_json_list_once(path: Path, values: list[dict]) -> None:
    if path.exists():
        expected = json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if path.read_text(encoding="utf-8") == expected:
            return
        raise FileExistsError(f"existing artifact differs from deterministic output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def chunks(text: str) -> list[str]:
    paras = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    result: list[str] = []
    for paragraph in paras:
        if len(paragraph) <= MAX_PARAGRAPH_CHARS:
            result.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            while len(sentence) > MAX_PARAGRAPH_CHARS:
                if current:
                    result.append(current)
                    current = ""
                result.append(sentence[:MAX_PARAGRAPH_CHARS])
                sentence = sentence[MAX_PARAGRAPH_CHARS:]
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > MAX_PARAGRAPH_CHARS:
                result.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            result.append(current)
    return [p for p in result if p]


def download(url: str, path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
        if offset and response.status_code == 200:
            offset = 0
        elif response.status_code not in (200, 206):
            raise RuntimeError(f"static dump download HTTP {response.status_code} for {url}")
        mode = "ab" if response.status_code == 206 and offset else "wb"
        total = response.headers.get("Content-Length")
        expected = int(total) + offset if total and response.status_code == 206 else int(total) if total else None
        h = hashlib.sha256()
        if mode == "ab":
            with part.open("rb") as existing:
                for block in iter(lambda: existing.read(1 << 20), b""):
                    h.update(block)
        received = offset
        last_log = received
        with part.open(mode) as stream:
            for block in response.iter_content(chunk_size=1 << 20):
                if not block:
                    continue
                stream.write(block)
                h.update(block)
                received += len(block)
                if received - last_log >= (64 << 20):
                    print(json.dumps({"download": path.name, "bytes": received,
                                      "expected_bytes": expected}, ensure_ascii=True), flush=True)
                    last_log = received
    if expected is not None and received != expected:
        raise RuntimeError(f"short shard download {path.name}: {received}/{expected}")
    if received < 16:
        raise RuntimeError(f"download too small: {path.name}")
    with part.open("rb") as stream:
        head = stream.read(4)
        stream.seek(-4, 2)
        tail = stream.read(4)
    if head != b"PAR1" or tail != b"PAR1":
        raise RuntimeError(f"invalid parquet framing: {path.name}")
    part.replace(path)
    return received, h.hexdigest()


def target_titles() -> dict[str, list[str]]:
    if not FROZEN.is_file():
        raise FileNotFoundError("run the title freeze step first")
    frozen = read_json(FROZEN)
    if frozen.get("fresh_goal_ab_confirmation_accessed"):
        raise RuntimeError("fresh Goal AB confirmation must remain sealed before development gate")
    return frozen["inventories"]


def original_corpora() -> dict[str, Path]:
    return {
        "goal_z_train": Z / "corpus-train.json",
        "goal_z_dev": Z / "corpus-dev.json",
        "old_regression": P / "corpus.json",
        "goal_y_regression": Y / "dev-corpus.json",
        "aa_regression": Z / "corpus-confirm.json",
    }


def scan_shard(path: Path, title_keys: set[str], shard_index: int) -> list[dict]:
    parquet = pq.ParquetFile(path)
    schema = set(parquet.schema.names)
    required = {"title", "text"}
    if not required.issubset(schema):
        raise RuntimeError(f"snapshot schema missing columns: {sorted(required - schema)}")
    columns = [name for name in ("id", "url", "title", "text") if name in schema]
    matches: dict[str, dict] = {}
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            key = norm(row.get("title", ""))
            text = str(row.get("text") or "").strip()
            if key not in title_keys or not text:
                continue
            value = {"snapshot": f"{SNAPSHOT_CONFIG}@{SNAPSHOT_COMMIT}",
                     "source_shard": shard_index, "source_id": row.get("id"),
                     "title": row["title"], "url": row.get("url"), "text": text}
            if key not in matches or len(text) > len(matches[key]["text"]):
                matches[key] = value
    return list(matches.values())


def assemble() -> dict:
    data_dir = OUT / "static-data"
    shard_files = sorted(data_dir.glob("matches-*.jsonl"))
    by_title: dict[str, dict] = {}
    for file in shard_files:
        for page in read_jsonl(file):
            key = norm(page["title"])
            if key not in by_title or len(page["text"]) > len(by_title[key]["text"]):
                by_title[key] = page
    write_jsonl_deterministic(data_dir / "articles-snapshot.jsonl", list(by_title.values()))
    corpora = original_corpora()
    inventories = target_titles()
    summaries = {}
    all_paragraph_records = []
    missing = []
    for label, source_path in corpora.items():
        original = json.loads(source_path.read_text(encoding="utf-8"))
        original_by_title = {norm(row["title"]): row for row in original}
        target = inventories[label]
        docs = []
        full_page_titles = 0
        for requested in target:
            page = by_title.get(norm(requested))
            text = page["text"] if page else str(original_by_title.get(norm(requested), {}).get("text", ""))
            if page:
                full_page_titles += 1
            else:
                missing.append({"corpus": label, "requested_title": requested,
                                "fallback": "frozen original corpus passage" if text else "none"})
            for index, paragraph in enumerate(chunks(text)):
                pid = digest_bytes(f"{SNAPSHOT_COMMIT}\0{label}\0{norm(requested)}\0{index}\0{paragraph}".encode())[:24]
                row = {"id": pid, "title": page["title"] if page else requested,
                       "text": paragraph, "requested_title": requested,
                       "source_id": page.get("source_id") if page else None,
                       "source_url": page.get("url") if page else None,
                       "snapshot": page.get("snapshot") if page else "original-corpus-fallback",
                       "paragraph_index": index}
                docs.append(row)
                all_paragraph_records.append({"corpus": label, **row})
        dest = data_dir / f"paragraphs-{label}.json"
        write_json_list_once(dest, docs)
        summaries[label] = {"titles": len(target), "full_page_matches": full_page_titles,
                            "fallback_titles": len(target) - full_page_titles,
                            "paragraphs": len(docs), "sha256": digest_file(dest)}
    write_jsonl_deterministic(data_dir / "paragraph-index.jsonl", all_paragraph_records)
    write_jsonl_deterministic(data_dir / "missing-title-fallbacks.jsonl", missing)
    manifest = {
        "version": "goal-ae-static-corpus-v1",
        "dataset": SNAPSHOT_REPO,
        "config": SNAPSHOT_CONFIG,
        "commit": SNAPSHOT_COMMIT,
        "license": "CC-BY-SA-3.0 and GFDL per Wikimedia dataset card; attribution is retained per page URL/title",
        "source_download": "pinned public Parquet files; no search API and no page API",
        "fresh_goal_ab_confirmation_accessed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parser": "Wikimedia provided plain text; split on blank lines; long paragraphs sentence-chunked at 1800 characters",
        "corpora": summaries,
        "missing_page_count": len(missing),
        "articles_snapshot_sha256": digest_file(data_dir / "articles-snapshot.jsonl"),
        "paragraph_index_sha256": digest_file(data_dir / "paragraph-index.jsonl"),
    }
    write_json(data_dir / "manifest.json", manifest)
    return manifest


def prepare() -> dict:
    if not OUT.is_dir() or not FROZEN.is_file():
        raise FileNotFoundError("Goal AE title inventory is not frozen")
    data_dir = OUT / "static-data"
    data_dir.mkdir(exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        run_state_path = OUT / "state.json"
        if run_state_path.exists():
            run_state = read_json(run_state_path)
            run_state.update({"status": "static_snapshot_prepared",
                              "static_snapshot_manifest_sha256": digest_file(manifest_path),
                              "fresh_goal_ab_confirmation_accessed": False})
            write_json(run_state_path, run_state, replace=True)
        return read_json(manifest_path)
    titles = target_titles()
    frozen = read_json(FROZEN)
    for label, source_path in original_corpora().items():
        expected_hash = frozen["corpora"][label]["sha256"]
        if digest_file(source_path) != expected_hash:
            raise RuntimeError(f"frozen corpus inventory changed: {label}")
    title_keys = {norm(title) for values in titles.values() for title in values}
    status_path = data_dir / "state.json"
    state = read_json(status_path) if status_path.exists() else {
        "status": "static_snapshot_downloading", "next_shard": 0,
        "dataset": SNAPSHOT_REPO, "config": SNAPSHOT_CONFIG, "commit": SNAPSHOT_COMMIT,
        "title_count": len(title_keys), "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(status_path, state, replace=status_path.exists())
    run_state_path = OUT / "state.json"
    if run_state_path.exists():
        run_state = read_json(run_state_path)
        run_state.update({"status": "static_snapshot_preparing", "api_attempt": "stopped_after_http_429",
                          "api_batches_saved": len(list((OUT / "api-batches").glob("batch-*.json"))),
                          "static_snapshot": f"{SNAPSHOT_CONFIG}@{SNAPSHOT_COMMIT}",
                          "fresh_goal_ab_confirmation_accessed": False})
        write_json(run_state_path, run_state, replace=True)
    for index in range(SHARDS):
        result = data_dir / f"matches-{index:05d}.jsonl"
        if result.exists():
            continue
        shard_name = f"train-{index:05d}-of-{SHARDS:05d}.parquet"
        shard_path = data_dir / "cache" / shard_name
        url = f"{BASE_URL}/{shard_name}"
        if shard_path.is_file():
            byte_count, checksum = shard_path.stat().st_size, digest_file(shard_path)
        else:
            byte_count, checksum = download(url, shard_path)
        matches = scan_shard(shard_path, title_keys, index)
        write_jsonl_once(result, matches)
        shard_path.unlink()
        manifest_path = data_dir / "download-ledger.json"
        ledger = read_json(manifest_path) if manifest_path.exists() else {"files": []}
        ledger["files"].append({"name": shard_name, "bytes": byte_count, "sha256": checksum,
                                 "matched_pages": len(matches)})
        write_json(manifest_path, ledger, replace=manifest_path.exists())
        state.update({"status": "static_snapshot_downloading", "next_shard": index + 1,
                      "completed_shards": index + 1, "total_shards": SHARDS,
                      "matched_pages": sum(len(read_jsonl(data_dir / f"matches-{i:05d}.jsonl"))
                                           for i in range(index + 1))})
        write_json(status_path, state, replace=True)
        print(json.dumps({"shard": index + 1, "total": SHARDS, "matched_pages": len(matches),
                          "matched_titles_total": state["matched_pages"]}, ensure_ascii=True), flush=True)
    manifest = assemble()
    state.update({"status": "static_snapshot_prepared", "manifest_sha256": digest_file(data_dir / "manifest.json"),
                  "completed_at": datetime.now(timezone.utc).isoformat()})
    write_json(status_path, state, replace=True)
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
