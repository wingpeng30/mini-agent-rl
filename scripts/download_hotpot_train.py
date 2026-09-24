"""Download the published HotpotQA distractor training shards with a speed floor."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "data/hotpot-source-v110/hf-distractor-tree.json"
DEST = ROOT / "data/hotpot-full-train-v114"
BASE = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
MIN_BYTES_PER_SECOND = 1_000_000
WINDOW_SECONDS = 5


def download(entry: dict) -> dict:
    path = DEST / Path(entry["path"]).name
    partial = path.with_name(path.name + ".part")
    expected_size = entry["size"]
    expected_sha = entry["lfs"]["oid"]
    url = BASE + entry["path"] + "?download=true"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing file: {path}")
    digest = hashlib.sha256()
    bytes_written = partial.stat().st_size if partial.exists() else 0
    if bytes_written > expected_size:
        raise RuntimeError(f"Partial file is larger than expected: {bytes_written}")
    if bytes_written:
        print(f"RESUME {path.name} offset={bytes_written}", flush=True)
        with partial.open("rb") as existing:
            for old_chunk in iter(lambda: existing.read(1024 * 1024), b""):
                digest.update(old_chunk)
    started = None
    window_started = None
    window_bytes = 0
    print(f"START {path.name} expected={expected_size} source={url}", flush=True)
    headers = {"Range": f"bytes={bytes_written}-"} if bytes_written else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 8)) as response:
        response.raise_for_status()
        if bytes_written and response.status_code != 206:
            raise RuntimeError(f"Server did not honor resume range (HTTP {response.status_code})")
        if response.headers.get("Content-Length"):
            actual_length = int(response.headers["Content-Length"])
            if actual_length != expected_size - bytes_written:
                raise RuntimeError(f"Unexpected response length: {actual_length}")
        output_mode = "ab" if bytes_written else "xb"
        with partial.open(output_mode) as output:
            iterator = response.iter_content(chunk_size=256 * 1024)
            while True:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except requests.exceptions.RequestException as exc:
                    elapsed = time.monotonic() - window_started if window_started else 0
                    speed = window_bytes / elapsed if elapsed else 0
                    if window_started and elapsed >= WINDOW_SECONDS and speed < MIN_BYTES_PER_SECOND:
                        raise RuntimeError(
                            f"SPEED_BELOW_1MBPS {speed / 1_000_000:.2f} MB/s; "
                            f"download stopped; partial={partial}"
                        ) from exc
                    raise
                if not chunk:
                    continue
                now = time.monotonic()
                if started is None:
                    started = window_started = now
                output.write(chunk)
                digest.update(chunk)
                bytes_written += len(chunk)
                window_bytes += len(chunk)
                elapsed = now - window_started
                if elapsed >= WINDOW_SECONDS:
                    speed = window_bytes / elapsed
                    print(
                        f"PROGRESS {path.name} bytes={bytes_written} "
                        f"speed_MBps={speed / 1_000_000:.2f}",
                        flush=True,
                    )
                    if speed < MIN_BYTES_PER_SECOND:
                        raise RuntimeError(
                            f"SPEED_BELOW_1MBPS {speed / 1_000_000:.2f} MB/s; "
                            f"download stopped; partial={partial}"
                        )
                    window_started = now
                    window_bytes = 0

    if bytes_written != expected_size:
        raise RuntimeError(f"Truncated download: {bytes_written} != {expected_size}")
    sha = digest.hexdigest()
    if sha != expected_sha:
        raise RuntimeError(f"SHA256 mismatch: {sha} != {expected_sha}")
    partial.replace(path)
    rows = pq.ParquetFile(path).metadata.num_rows
    duration = time.monotonic() - started if started else 0
    print(f"DONE {path.name} rows={rows} sha256={sha} MBps={bytes_written / duration / 1_000_000:.2f}", flush=True)
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "source": url,
        "bytes": bytes_written,
        "sha256": sha,
        "rows": rows,
        "download_seconds": round(duration, 3),
    }


def main() -> None:
    entries = json.loads(TREE.read_text(encoding="utf-8"))
    train = [item for item in entries if item["path"].startswith("distractor/train-")]
    if len(train) != 2:
        raise RuntimeError(f"Expected two train shards; got {len(train)}")
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in train:
        records.append(download(entry))
    total_rows = sum(record["rows"] for record in records)
    if total_rows != 90_447:
        raise RuntimeError(f"Unexpected training row count: {total_rows}")
    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "rows": total_rows,
        "files": records,
        "source_metadata": str(TREE.relative_to(ROOT)).replace("\\", "/"),
        "speed_floor_bytes_per_second": MIN_BYTES_PER_SECOND,
    }
    (DEST / "download-provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"COMPLETE rows={total_rows} manifest={DEST / 'download-provenance.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr, flush=True)
        sys.exit(20 if "SPEED_BELOW_1MBPS" in str(exc) else 1)
